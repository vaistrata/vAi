#include "tensor.h"
#include "neural-net.h"   // getVaiDevice(), getVaiPool()
#include <atomic>
#include <bit>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>


namespace vai {




// IEEE-754 binary16 (F16) <-> binary32 (F32) bit conversion helpers.
// Written without branches so the loops over them vectorize; each was verified
// against its branching form over every input.

// F16 -> F32. A subnormal is normalized by one float subtract rather than a shift
// loop: biasing by 113<<23 puts the value where subtracting that same constant
// leaves it correctly normalized.
static inline float halfToFloat(uint16_t h)
{
    constexpr uint32_t magic      = 113u << 23;
    constexpr uint32_t shiftedExp = 0x7C00u << 13;

    const uint32_t o0   = uint32_t(h & 0x7FFFu) << 13;
    const uint32_t exp  = shiftedExp & o0;
    const uint32_t base = o0 + ((127u - 15u) << 23);

    const uint32_t infNan = base + ((128u - 16u) << 23);
    const uint32_t sub    = std::bit_cast<uint32_t>(
        std::bit_cast<float>(base + (1u << 23)) - std::bit_cast<float>(magic));

    uint32_t o = base;
    o = (exp == shiftedExp) ? infNan : o;
    o = (exp == 0)          ? sub    : o;
    return std::bit_cast<float>(o | (uint32_t(h & 0x8000u) << 16));
}


// F32 -> F16, round-to-nearest-even, NaN/Inf preserved.
// Written without branches so the loop over it can vectorize: every case is
// computed and then selected. Verified against the branching form over all 2^32
// inputs.
static inline uint16_t floatToHalf(float f)
{
    const uint32_t b      = std::bit_cast<uint32_t>(f);
    const uint32_t sign   = (b >> 16) & 0x8000u;
    const uint32_t expRaw = (b >> 23) & 0xFFu;
    const uint32_t mant   = b & 0x7FFFFFu;

    // Normal case with round-to-nearest-even folded in. A rounding carry lands in
    // bit 10 of mant10, and adding that to exp*1024 carries into the exponent on
    // its own, so a mantissa that overflowed needs no test of its own.
    const int32_t  exp    = int32_t(expRaw) - 112;                 // 127 - 15
    const uint32_t rest   = mant & 0x1FFFu;
    uint32_t mant10       = mant >> 13;
    mant10 += (rest > 0x1000u) | ((rest == 0x1000u) & (mant10 & 1u));
    const uint32_t normal = uint32_t(exp) * 0x400u + mant10;

    const uint32_t nanInf = 0x7C00u | (mant != 0 ? 0x0200u : 0u);

    uint32_t out = normal;
    out = (normal >= 0x7C00u) ? 0x7C00u : out;   // overflow saturates to inf
    out = (exp <= 0)          ? 0u      : out;   // underflow flushes to zero
    out = (expRaw == 0xFFu)   ? nanInf  : out;   // NaN / Inf pass through
    return uint16_t(sign | out);
}


// BF16 -> F32 bit conversion helper
static inline float bfloatToFloat(uint16_t b) {
    return std::bit_cast<float>(uint32_t(b) << 16);
}

// F32 -> BF16, round-to-nearest-even
static inline uint16_t floatToBfloat(float f)
{
    uint32_t b = std::bit_cast<uint32_t>(f);
    if (((b >> 23) & 0xFFu) == 0xFFu && (b & 0x7FFFFFu))     // NaN
        return uint16_t((b >> 16) | 0x40u);                  // canonical quiet NaN
    uint32_t rnd = b + 0x7FFFu + ((b >> 16) & 1u);
    return uint16_t(rnd >> 16);
}


// F16 -> BF16, round-to-nearest-even. F16 widens to F32 exactly, so the rounding
// is done there. The NaN case is not delegated to floatToBfloat: this one
// canonicalizes every NaN to 0x7FC0 where that one keeps the payload.
static inline uint16_t halfToBfloat(uint16_t h)
{
    const uint32_t f    = std::bit_cast<uint32_t>(halfToFloat(h));
    const uint32_t rnd  = (f + 0x7FFFu + ((f >> 16) & 1u)) >> 16;
    const uint32_t sign = h & 0x8000u;
    const uint32_t mag  = h & 0x7FFFu;

    uint32_t out = rnd;
    out = (mag == 0x7C00u) ? (sign | 0x7F80u) : out;         // inf
    out = (mag >  0x7C00u) ? (sign | 0x7FC0u) : out;         // NaN, canonicalized
    return uint16_t(out);
}


// BF16 -> F16, NaN/Inf preserved, overflow -> inf, underflow -> 0.
// BF16 is the top half of an F32, so widening costs a shift and the narrowing
// rules are floatToHalf's already.
static inline uint16_t bfloatToHalf(uint16_t b)
{
    return floatToHalf(bfloatToFloat(b));
}


// A load-time pass touches every element of a tensor, and a checkpoint holds
// billions of them. Splitting [0, count) across cores pays once a tensor is large
// enough that starting the threads disappears next to the work they do.
template <typename Fn>
static void parallelChunks(size_t count, Fn fn)
{
    constexpr size_t kMinPerThread = 1u << 20;

    const unsigned cores   = std::thread::hardware_concurrency();
    const size_t   threads = std::min<size_t>(cores ? cores : 1, count / kMinPerThread);

    if (threads <= 1)
    {
        fn(size_t(0), count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);

    const size_t chunk = count / threads;
    for (size_t t = 1; t < threads; ++t)
    {
        const size_t begin = t * chunk;
        const size_t end   = (t + 1 == threads) ? count : begin + chunk;
        workers.emplace_back([=] { fn(begin, end); });
    }
    fn(size_t(0), chunk);   // this thread takes the first chunk

    for (std::thread& worker : workers)
        worker.join();
}

template <typename Src, typename Dst, typename Fn>
static void transformParallel(const Src* src, Dst* dst, size_t count, Fn fn)
{
    parallelChunks(count, [=](size_t begin, size_t end) {
        std::transform(src + begin, src + end, dst + begin, fn);
    });
}


// fp32 -> Q4 (layout at vai::Dtype). Each group's largest magnitude maps to
// q = 0 (-8 * scale), as ggml's Q4_0 does; the scale is rounded to fp16 first,
// so q is chosen against the scale the GPU will read.
static void quantizeQ4(const float* src, void* dst, size_t count)
{
    auto words  = static_cast<uint32_t*>(dst);
    auto scales = reinterpret_cast<uint16_t*>(static_cast<char*>(dst) + count / 2);
    parallelChunks(count / 32, [=](size_t begin, size_t end) {
        for (size_t g = begin; g < end; ++g)
        {
            const float* x = src + g * 32;
            // |peak| has its own variable: testing fabs(x[i]) > fabs(peak) makes
            // NDK r30's clang 21 at -O3 skip a negative peak at x[0].
            float peak = 0.0f, mag = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                const float a = std::fabs(x[i]);
                if (a > mag) { mag = a; peak = x[i]; }
            }

            const uint16_t scale = floatToHalf(peak / -8.0f);
            const float d = halfToFloat(scale);
            const float inv = (d != 0.0f) ? 1.0f / d : 0.0f;
            scales[g] = scale;
            for (int w = 0; w < 4; ++w)
            {
                uint32_t word = 0;
                for (int i = 0; i < 8; ++i)
                {
                    // In double the product is exact, so q is the same whether or
                    // not the compiler fuses this into an FMA (clang does on ARM).
                    const int q = std::clamp(int(std::floor(double(x[w * 8 + i]) * inv + 8.5)), 0, 15);
                    word |= uint32_t(q) << (4 * i);
                }
                words[g * 4 + w] = word;
            }
        }
    });
}

static void dequantizeQ4(const void* src, float* dst, size_t count)
{
    auto words  = static_cast<const uint32_t*>(src);
    auto scales = reinterpret_cast<const uint16_t*>(static_cast<const char*>(src) + count / 2);
    parallelChunks(count / 8, [=](size_t begin, size_t end) {
        for (size_t w = begin; w < end; ++w)
        {
            const float d = halfToFloat(scales[w / 4]);
            for (int i = 0; i < 8; ++i)
                dst[w * 8 + i] = (float((words[w] >> (4 * i)) & 0xFu) - 8.0f) * d;
        }
    });
}


// dtype-tagged element-wise conversion of `count` elements from src to dst.
// Handles all fp32 / fp16 combinations and Q4 both ways; same-dtype is a memcpy.
void convertDtype(const void* src, Dtype srcDtype,
                  void* dst, Dtype dstDtype, size_t count)
{
    if (srcDtype == dstDtype)
    {
        std::memcpy(dst, src, byteSize(srcDtype, count));
        return;
    }
    if (srcDtype == Dtype::Q4 || dstDtype == Dtype::Q4)
    {
        VAI_ASSERT(count % 32 == 0);
        // Q4 goes through fp32 on the side that is not Q4.
        const bool f32 = (srcDtype == Dtype::Fp32 || dstDtype == Dtype::Fp32);
        std::vector<float> wide(f32 ? 0 : count);
        if (dstDtype == Dtype::Q4)
        {
            if (!f32) convertDtype(src, srcDtype, wide.data(), Dtype::Fp32, count);
            quantizeQ4(f32 ? static_cast<const float*>(src) : wide.data(), dst, count);
        }
        else
        {
            dequantizeQ4(src, f32 ? static_cast<float*>(dst) : wide.data(), count);
            if (!f32) convertDtype(wide.data(), Dtype::Fp32, dst, dstDtype, count);
        }
        return;
    }
    auto sF = static_cast<const float*>(src);
    auto sU = static_cast<const uint16_t*>(src);
    auto dF = static_cast<float*>(dst);
    auto dU = static_cast<uint16_t*>(dst);

    constexpr auto K = [](Dtype s, Dtype d) { return (unsigned(s) << 2) | unsigned(d); };
    using D = Dtype;
    switch (K(srcDtype, dstDtype))
    {
        case K(D::Fp32, D::Fp16): transformParallel(sF, dU, count, floatToHalf);   break;
        case K(D::Fp16, D::Fp32): transformParallel(sU, dF, count, halfToFloat);   break;
    }
}


// Converts count elements stored as storedType (a file dtype, BF16 included) into
// dstDtype on the calling thread. BF16 is a file format only, not a tensor dtype:
// it widens losslessly to fp32 unless fp16 is asked for.
static void convertStored(const char* src, const std::string& storedType, char* dst, Dtype dstDtype, size_t count)
{
    const Dtype srcDtype = (storedType == "BF16") ? Dtype::None : dtypeFromString(storedType);
    auto sU = reinterpret_cast<const uint16_t*>(src);
    auto sF = reinterpret_cast<const float*>(src);
    auto dU = reinterpret_cast<uint16_t*>(dst);
    auto dF = reinterpret_cast<float*>(dst);
    if (srcDtype == dstDtype)                                   std::memcpy(dst, src, count * bytesPerElement(dstDtype));
    else if (srcDtype == Dtype::None && dstDtype == Dtype::Fp32) std::transform(sU, sU + count, dF, bfloatToFloat);
    else if (srcDtype == Dtype::None)                           std::transform(sU, sU + count, dU, bfloatToHalf);
    else if (dstDtype == Dtype::Fp16)                           std::transform(sF, sF + count, dU, floatToHalf);
    else                                                        std::transform(sU, sU + count, dF, halfToFloat);
}


// Reads srcBytes of ref from byteOffset into dst as dstDtype. Worker threads take
// parts of it in turn and each converts the part it just read, so the drive keeps
// several reads in flight and conversion runs while other parts are being read.
// The per-worker staging buffers belong to the calling thread and are reused
// across loads, so their pages fault in once.
static void loadInto(const SafetensorRef& ref, size_t byteOffset, size_t srcBytes,
                     const std::string& storedType, char* dst, Dtype dstDtype)
{
    constexpr size_t kPartBytes = size_t(8) << 20;
    constexpr unsigned kWorkers = 8;

    const bool same = storedType != "BF16" && dtypeFromString(storedType) == dstDtype;
    const size_t srcElem = (storedType == "BF16") ? 2 : bytesPerElement(dtypeFromString(storedType));
    const size_t dstElem = bytesPerElement(dstDtype);
    const size_t parts = (srcBytes + kPartBytes - 1) / kPartBytes;
    const unsigned workers = unsigned(std::min<size_t>(kWorkers, parts));

    thread_local HostBuffer staging[kWorkers];
    char* buf[kWorkers];
    for (unsigned w = 0; w < workers; ++w)
    {
        if (!same && staging[w].size() < kPartBytes)
            staging[w].resize(kPartBytes);
        buf[w] = staging[w].data();
    }

    std::atomic<size_t> next{0};
    auto work = [&](unsigned w)
    {
        for (size_t i; (i = next++) < parts;)
        {
            const size_t pos = i * kPartBytes;
            const size_t n = std::min(kPartBytes, srcBytes - pos);
            if (same)
                ref.readInto(dst + pos, byteOffset + pos, n);
            else
            {
                ref.readInto(buf[w], byteOffset + pos, n);
                convertStored(buf[w], storedType, dst + pos / srcElem * dstElem, dstDtype, n / srcElem);
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned w = 1; w < workers; ++w)
        threads.emplace_back(work, w);
    work(0);
    for (std::thread& t : threads)
        t.join();
}


Tensor::Tensor(const SafetensorRef& ref, Dtype targetDtype)
: data_(std::make_shared<TensorData>(targetDtype==Dtype::None ? dtypeFromString(ref.getDataType()) : targetDtype))
{
    shape_ = ref.getShape();

    if (dtype() == Dtype::Q4)
    {
        setFloats(static_cast<const float*>(Tensor(ref, Dtype::Fp32).hostData()));
        return;
    }

    const std::string storedType = ref.getDataType();
    const size_t count    = numElements();
    const size_t srcBytes = ref.byteSize();
    const size_t dstBytes = count * bytesPerElement(dtype());

    data_->hostData_.resize(dstBytes);
    loadInto(ref, 0, srcBytes, storedType, data_->hostData_.data(), dtype());
}


// Checkpoint tensors stacked along the leading axis, the inverse of split().
// Each is read into its own part of one buffer, so the bytes are written once.
Tensor concat(std::initializer_list<SafetensorRef> refs, Dtype targetDtype)
{
    VAI_ASSERT(refs.size() > 0);
    VAI_ASSERT(targetDtype != Dtype::None);

    if (targetDtype == Dtype::Q4)
    {
        Tensor wide = concat(refs, Dtype::Fp32);
        return Tensor(wide.shape(), Dtype::Q4).setFloats(static_cast<const float*>(wide.hostData()));
    }

    const std::string storedType = refs.begin()->getDataType();

    std::vector<uint32_t> shape = refs.begin()->getShape();
    VAI_ASSERT(!shape.empty());
    shape[0] = 0;

    // Sizes are kept from this pass: byteSize() costs a map lookup and a shape
    // product, and the read below would otherwise ask for them a second time.
    std::vector<size_t> sizes;
    sizes.reserve(refs.size());

    for (const SafetensorRef& ref : refs)
    {
        VAI_ASSERT(ref.getDataType() == storedType);

        const std::vector<uint32_t> part = ref.getShape();
        VAI_ASSERT(part.size() == shape.size());
        for (size_t d = 1; d < part.size(); ++d)   // only the leading axis adds up
            VAI_ASSERT(part[d] == shape[d]);

        shape[0] += part[0];
        sizes.push_back(ref.byteSize());
    }

    Tensor out(shape, targetDtype);
    HostBuffer& host = out.data_->hostData_;
    host.resize(out.numElements() * bytesPerElement(targetDtype));

    // Each part lands at its own offset of the one buffer.
    const size_t srcElem = (storedType == "BF16") ? 2 : bytesPerElement(dtypeFromString(storedType));
    char* dst = host.data();
    size_t i = 0;
    for (const SafetensorRef& ref : refs)
    {
        loadInto(ref, 0, sizes[i], storedType, dst, targetDtype);
        dst += sizes[i++] / srcElem * bytesPerElement(targetDtype);
    }

    return out;
}



Tensor slice(const SafetensorRef& ref, uint32_t begin, uint32_t end, Dtype targetDtype)
{
    VAI_ASSERT(targetDtype != Dtype::None);

    if (targetDtype == Dtype::Q4)
    {
        Tensor wide = slice(ref, begin, end, Dtype::Fp32);
        return Tensor(wide.shape(), Dtype::Q4).setFloats(static_cast<const float*>(wide.hostData()));
    }

    std::vector<uint32_t> shape = ref.getShape();
    VAI_ASSERT(!shape.empty());
    VAI_ASSERT(begin < end && end <= shape[0]);

    const std::string storedType = ref.getDataType();
    const size_t rowBytes  = ref.byteSize() / shape[0];   // as stored
    const size_t srcOffset = size_t(begin) * rowBytes;
    const size_t srcBytes  = size_t(end - begin) * rowBytes;

    shape[0] = end - begin;
    Tensor out(shape, targetDtype);
    HostBuffer& host = out.data_->hostData_;
    host.resize(out.numElements() * bytesPerElement(targetDtype));
    loadInto(ref, srcOffset, srcBytes, storedType, host.data(), targetDtype);

    return out;
}


Tensor sum(const SafetensorRef& a, const SafetensorRef& b, Dtype targetDtype)
{
    VAI_ASSERT(targetDtype != Dtype::None);

    Tensor ta(a, Dtype::Fp32);
    Tensor tb(b, Dtype::Fp32);

    // Axes align from the back; each pair is equal or has a 1.
    const size_t rank = std::max(ta.shape().size(), tb.shape().size());
    std::vector<uint32_t> da(rank, 1), db(rank, 1), shape(rank);
    std::copy(ta.shape().begin(), ta.shape().end(), da.end() - ta.shape().size());
    std::copy(tb.shape().begin(), tb.shape().end(), db.end() - tb.shape().size());
    size_t count = 1;
    for (size_t d = 0; d < rank; ++d)
    {
        VAI_ASSERT(da[d] == db[d] || da[d] == 1 || db[d] == 1);
        shape[d] = std::max(da[d], db[d]);
        count *= shape[d];
    }

    // Leading 1s, then exactly the result's trailing axes: a flat index modulo
    // its size reaches the element.
    auto isSuffix = [&](const std::vector<uint32_t>& dims)
    {
        size_t d = 0;
        while (d < rank && dims[d] == 1) ++d;
        return std::equal(dims.begin() + d, dims.end(), shape.begin() + d);
    };

    float* pa = reinterpret_cast<float*>(ta.data_->hostData_.data());
    const float* pb = static_cast<const float*>(tb.hostData());
    const size_t countA = ta.numElements(), countB = tb.numElements();

    // When a has the result's shape, b is added where a was read.
    std::vector<float> fresh(da == shape ? 0 : count);
    float* out = fresh.empty() ? pa : fresh.data();

    if (isSuffix(da) && isSuffix(db))
    {
        parallelChunks(count, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                out[i] = pa[countA == count ? i : i % countA] + pb[i % countB];
        });
    }
    else
    {
        // A broadcast axis takes stride 0.
        std::vector<size_t> strideA(rank), strideB(rank);
        for (size_t d = rank, sA = 1, sB = 1; d-- > 0; sA *= da[d], sB *= db[d])
        {
            strideA[d] = (da[d] == 1) ? 0 : sA;
            strideB[d] = (db[d] == 1) ? 0 : sB;
        }
        parallelChunks(count, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                size_t rest = i, ia = 0, ib = 0;
                for (size_t d = rank; d-- > 0; rest /= shape[d])
                {
                    const size_t c = rest % shape[d];
                    ia += c * strideA[d];
                    ib += c * strideB[d];
                }
                out[i] = pa[ia] + pb[ib];
            }
        });
    }

    if (out == pa && targetDtype == Dtype::Fp32)
        return ta.reshape(shape);

    Tensor result(shape, targetDtype);
    VAI_ASSERT(targetDtype != Dtype::Q4 || shape.back() % 32 == 0);
    result.data_->hostData_.resize(byteSize(targetDtype, count));
    convertDtype(out, Dtype::Fp32, result.data_->hostData_.data(), targetDtype, count);
    return result;
}


Tensor& Tensor::setFloats(const std::vector<float>& inData)
{
    VAI_ASSERT(numElements() == inData.size());
    return setFloats(inData.data());
}


Tensor& Tensor::setFloats(std::vector<float>&& inData)
{
    const std::vector<float> consumed = std::move(inData);   // the caller's buffer is freed on return
    return setFloats(consumed);
}


Tensor& Tensor::setFloats(const float* inData)
{
    VAI_ASSERT(validShape() && baseOffset_ == 0);
    VAI_ASSERT(data_->dtype_ != Dtype::Q4 || shape_.back() % 32 == 0);   // a Q4 group stays in one row
    const size_t count = numElements();

    data_->lastSubmission_ = {nullptr, 0};
    data_->hostUploaded_ = false;

    data_->hostData_.resize(byteSize(data_->dtype_, count));
    convertDtype(inData, Dtype::Fp32, data_->hostData_.data(), data_->dtype_, count);
    return *this;
}


std::shared_ptr<BufferPool> Tensor::pool()
{
    static std::shared_ptr<BufferPool> p =
        std::make_shared<BufferPool>(BufferPoolConfig{getVaiDevice(), 0});
    return p;
}

static eva::CommandBuffer& downloadCmd()
{
    static eva::CommandBuffer cmd = getVaiDevice().newCommandBuffer(
        eva::queue_compute, eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);
    return cmd;
}

Tensor& Tensor::downloadToHost()
{
    VAI_ASSERT(data_ && data_->buffer_);
    VAI_ASSERT(data_->lastSubmission_.second && "downloadToHost(): tensor is not a forward output");

    if (data_->hostSyncedAt_ != data_->lastSubmission_)
    {
        const size_t bytes = numBytes();
        const size_t cap = nextPow2(bytes);

        // transient readback (exact alloc, reuse up to pow2)
        PooledBuffer readback = pool()->requestBuffer(
            eva::BUFFER_USAGE::TRANSFER_DST,
            eva::MEMORY_PROPERTY::HOST_CACHED | eva::MEMORY_PROPERTY::HOST_VISIBLE,
            bytes, cap);

        downloadCmd().begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
            .copyBuffer(data_->buffer_.buffer, readback.buffer, byteOffset(), 0, bytes)
        .end().submit().wait();

        data_->hostData_.resize(bytes);

        // HOST_CACHED: invalidate (after map) so the device write is visible to the CPU
        uint8_t* src = readback.buffer.map();
        readback.buffer.invalidate();
        std::memcpy(data_->hostData_.data(), src, bytes);

        data_->hostSyncedAt_ = data_->lastSubmission_;
    }
    return *this;
}


} // namespace vai
