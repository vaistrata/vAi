#ifndef TENSOR_H
#define TENSOR_H


#include "error.h"
#include "safetensors-parser.h"
#include "buffer-pool.h"
#include <array>
#include <memory>
#include <tuple>
#include <vector>
#include <cstring>
#include <utility>

namespace vai {

class NeuralNet;

// Q4 is a weight format: symmetric 4-bit values in groups of 32 consecutive
// elements (the last axis a multiple of 32), each group with an fp16 scale.
// The packed words come first, a uint32 per 8 elements with element i in bits
// 4i..4i+3 holding q for the value (q - 8) * scale; the scales follow, one per
// group. Linear reads it; on the host it is only converted to and from.
enum class Dtype { Fp32, Fp16, None, Q4 };

constexpr size_t bytesPerElement(Dtype dt) noexcept
{
    switch (dt) {
        case Dtype::Fp16:  return 2;
        case Dtype::Fp32:  return 4;
        case Dtype::None:  return 0;
        case Dtype::Q4:    return 0;   // packed; see byteSize()
    }
    return 4;
}

// Bytes that `count` elements of dt take, Q4's scales included.
constexpr size_t byteSize(Dtype dt, size_t count) noexcept
{
    return (dt == Dtype::Q4) ? count / 2 + count / 32 * 2 : count * bytesPerElement(dt);
}

// Tensor dtype a safetensors dtype string loads as. BF16 is a file format
// only, not a tensor dtype - it loads as fp32 (lossless) unless the caller
// requests a target dtype (see Tensor(SafetensorRef, Dtype)).
inline Dtype dtypeFromString(const std::string& s)
{
    if (s == "F32")  return Dtype::Fp32;
    if (s == "F16")  return Dtype::Fp16;
    if (s == "BF16") return Dtype::Fp32;
    VAI_ASSERT(false && "Unsupported dtype string");
    return Dtype::Fp32;
}


// Element-wise dtype conversion of `count` elements (host-side).
// Handles all fp32 / fp16 combinations, and to and from Q4 (count a multiple
// of 32, each group within one row); same-dtype is a memcpy.
void convertDtype(const void* src, Dtype srcDtype, void* dst, Dtype dstDtype, size_t count);


class Tensor;

// Checkpoint tensors stacked along the leading axis, the inverse of split().
// Each is read into its own part of one buffer, so the bytes are written once.
// Every ref must share a dtype and its trailing dimensions.
Tensor concat(std::initializer_list<SafetensorRef> refs, Dtype targetDtype);


// Rows [begin, end) of a checkpoint tensor's leading axis. Only those bytes
// leave the file, so the rows left out cost neither a read nor a conversion.
Tensor slice(const SafetensorRef& ref, uint32_t begin, uint32_t end, Dtype targetDtype);


// a + b under numpy broadcasting (e.g. a position table [N, D] plus a bias
// [D]). Summed in fp32 and rounded to targetDtype once.
Tensor sum(const SafetensorRef& a, const SafetensorRef& b, Dtype targetDtype);


// Smallest power of two >= n (with nextPow2(0) == 1). Used to bucket grow-only
// buffer capacities so they grow only occasionally and reuse across shapes.
inline size_t nextPow2(size_t n) noexcept
{
    if (n <= 1) return 1;
    n--;
    for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1)
        n |= n >> shift;
    return n + 1;
}


using HostBuffer = std::vector<char>;


class TensorData
{
    friend class Tensor;
    friend class NeuralNet;
    friend Tensor concat(std::initializer_list<SafetensorRef> refs, Dtype targetDtype);
    friend Tensor slice(const SafetensorRef& ref, uint32_t begin, uint32_t end, Dtype targetDtype);
    friend Tensor sum(const SafetensorRef& a, const SafetensorRef& b, Dtype targetDtype);

    HostBuffer hostData_;
    bool hostUploaded_ = false;     // hostData_ is already on the device; a setter clears it

    const Dtype dtype_;
    vai::PooledBuffer buffer_;

    std::pair<NeuralNet*, uint64_t> lastSubmission_{nullptr, 0};
    std::pair<NeuralNet*, uint64_t> hostSyncedAt_{nullptr, 0};

public:
    TensorData(Dtype dt = Dtype::Fp32) : dtype_(dt)
    {
        VAI_ASSERT(dt != Dtype::None);
    }
    ~TensorData()
    {
        buffer_.release();
    }
};


class Tensor
{
    friend class NeuralNet;
    friend Tensor concat(std::initializer_list<SafetensorRef> refs, Dtype targetDtype);
    friend Tensor slice(const SafetensorRef& ref, uint32_t begin, uint32_t end, Dtype targetDtype);
    friend Tensor sum(const SafetensorRef& a, const SafetensorRef& b, Dtype targetDtype);

    std::shared_ptr<TensorData> data_;
    std::vector<uint32_t> shape_;
    uint32_t baseOffset_ = 0;   // in elements, not bytes; != 0 means this is a slice()

    void bindBuffer(vai::PooledBuffer buf, uint32_t baseOffset = 0)
    {
        VAI_ASSERT(data_ && baseOffset_ == 0);
        data_->buffer_.release();
        data_->buffer_ = std::move(buf);
        baseOffset_ = baseOffset;
        VAI_ASSERT(!data_->buffer_ || byteOffset() + numBytes() <= data_->buffer_.buffer.size());
    }

    bool hostUploaded() const { return data_ && data_->hostUploaded_; }
    void markHostUploaded() { VAI_ASSERT(data_); data_->hostUploaded_ = true; }

    void clearHostData(bool shrink = false)
    {
        VAI_ASSERT(data_);
        data_->hostData_.clear();
        if (shrink)
            data_->hostData_.shrink_to_fit();
    }

public:
    Tensor(const std::vector<uint32_t>& shape, Dtype dtype = Dtype::Fp32)
    : data_(std::make_shared<TensorData>(dtype)), shape_(shape) {}

    Tensor(const SafetensorRef& ref, Dtype targetDtype = Dtype::None);

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    // Ingest float data, converting fp32 -> this tensor's dtype.
    Tensor& setFloats(const std::vector<float>& inData);
    Tensor& setFloats(std::vector<float>&& inData);
    Tensor& setFloats(const float* inData);

    // Raw typed write: memcpy src as-is (no dtype conversion). Mirror of getData<T>().
    // sizeof(T) must match this tensor's element size (e.g. uint32_t into an fp32 slot).
    template <typename T>
    Tensor& setData(const T* src)
    {
        VAI_ASSERT(validShape() && baseOffset_ == 0);
        VAI_ASSERT(sizeof(T) == bytesPerElement(data_->dtype_) && "setData<T>: sizeof(T) != element size");
        const size_t bytes = numBytes();

        data_->lastSubmission_ = {nullptr, 0};
        data_->hostUploaded_ = false;

        data_->hostData_.resize(bytes);
        std::memcpy(data_->hostData_.data(), src, bytes);
        return *this;
    }

    /**
     * The same data seen as `dims`, leaving this tensor untouched. The result
     * shares this tensor's data; only the handle's shape differs.
     *
     *   Tensor flat = t.reshape({M * N});   // t stays [M, N]
     */
    Tensor reshape(const std::vector<uint32_t>& dims) const
    {
        VAI_ASSERT(data_);
        size_t total = 1;
        for (auto d : dims) total *= d;
        VAI_ASSERT(numElements() == total);
        Tensor out = *this;
        out.shape_ = dims;
        return out;
    }

    /**
     * Reorder the dimensions, leaving this tensor untouched.
     *
     *   Tensor t = w.permute(1, 0);   // [in, out] -> [out, in]
     */
    template <typename... Ts>
    Tensor permute(Ts... Perms) const;

    /**
     * Cut the leading axis at each offset, yielding one more tensor than there
     * are offsets. Each piece owns its bytes — a tensor cannot alias another —
     * and keeps this tensor's dtype and trailing dimensions.
     *
     *   auto [q, k, v] = qkv.split(D, 2 * D);   // [3D, ...] -> three [D, ...]
     */
    template <typename... Ts>
    auto split(Ts... offsets) const;   // std::tuple, one Tensor per piece

    /**
     * Concatenate along the leading axis, the inverse of split(). Every tensor
     * must match this one's dtype and trailing dimensions; the result owns its
     * bytes.
     *
     *   Tensor gu = gate.concat(up);   // two [H, K] -> one [2H, K]
     */
    template <typename... Ts>
    Tensor concat(const Ts&... others) const;

    /**
     * Reinterpret the bound buffer as `dims` starting at element `baseOffset`,
     * leaving this tensor untouched. Unlike permute() / split(), the result
     * shares this tensor's buffer.
     *
     *   Tensor q = qkv.view(shape, 0);   // aliases qkv's buffer, no copy
     */
    Tensor view(const std::vector<uint32_t>& dims, uint32_t baseOffset) const
    {
        VAI_ASSERT(data_);

        Tensor out = *this;
        out.shape_ = dims;
        out.baseOffset_ = baseOffset;
        VAI_ASSERT(!data_->buffer_ || out.byteOffset() + out.numBytes() <= data_->buffer_.buffer.size());
        return out;
    }

    uint64_t byteOffset() const
    {
        return uint64_t(baseOffset_) * bytesPerElement(dtype());
    }

    const std::vector<uint32_t>& shape() const
    {
        static const std::vector<uint32_t> emptyShape{};
        return data_ ? shape_ : emptyShape;
    }

    template <typename... Ts>
    bool isShapeOf(Ts... dims) const
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        if (!data_ || shape_.size() != sizeof...(Ts))
            return false;

        const uint32_t dimArray[] = { uint32_t(dims)... };

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return ((dimArray[Is] == shape_[Is] || dimArray[Is] == uint32_t(-1)) && ...);
        }(std::index_sequence_for<Ts...>{});
    }

    Dtype dtype() const
    {
        return data_ ? data_->dtype_ : Dtype::None;
    }

    size_t numElements() const
    {
        if (!data_ || shape_.empty()) return 0;
        size_t sz = 1; for (uint32_t dim : shape_) sz *= dim;
        return sz;
    }

    size_t numBytes() const
    {
        return byteSize(dtype(), numElements());
    }

    bool validShape() const
    {
        if (!data_ || shape_.empty()) return false;
        for (uint32_t dim : shape_) if (dim == 0) return false;
        return true;
    }

    bool hasHostData() const
    {
        return data_ && !data_->hostData_.empty();
    }

    const void* hostData() const
    {
        if (!data_ || data_->hostData_.empty()) return nullptr;
        return data_->hostData_.data();
    }

    bool gpuBufferBound() const
    {
        return data_ && data_->buffer_;
    }

    uint32_t baseOffset() const
    {
        return baseOffset_;
    }

    eva::BufferRange buffer() const
    {
        VAI_ASSERT(data_ && data_->buffer_);
        return eva::BufferRange(data_->buffer_.buffer, byteOffset());
    }

    std::pair<NeuralNet*, uint64_t> lastSubmission() const
    {
        return data_ ? data_->lastSubmission_ : std::pair<NeuralNet*, uint64_t>{nullptr, 0};
    }

    // Copy device data into a host vector.
    Tensor& downloadToHost();
    
    template <typename T = float>
    const T* getData()   // downloadToHost() + typed pointer into the host copy
    {
        return reinterpret_cast<const T*>(downloadToHost().data_->hostData_.data());
    }

private:
    // Engine-default pool for downloadToHost() readback buffers — not tied to any NeuralNet.
    static std::shared_ptr<BufferPool> pool();
};


template <typename... Ts>
inline Tensor Tensor::permute(Ts... Perms) const
{
    static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
    VAI_ASSERT(data_ && !data_->hostData_.empty());

    const uint32_t rank = shape_.size();
    VAI_ASSERT(sizeof...(Perms) == rank);
    VAI_ASSERT(((Perms < rank) && ...)); // Ensure all Perms are valid indices

    const uint32_t permutation[] = { static_cast<uint32_t>(Perms)... };

    for (size_t i = 0; i < rank; ++i)
        for (size_t j = i + 1; j < rank; ++j)
            VAI_ASSERT(permutation[i] != permutation[j]);

    std::vector<uint32_t> dstShape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i)
        dstShape[i] = shape_[permutation[i]];

    std::vector<size_t> srcStrides(rank);
    srcStrides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
        srcStrides[i] = srcStrides[i + 1] * shape_[i + 1];

    std::vector<size_t> dstStrides(rank);
    dstStrides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
        dstStrides[i] = dstStrides[i + 1] * dstShape[i + 1];

    VAI_ASSERT(data_->dtype_ != Dtype::Q4);
    const size_t n = numElements();
    const size_t elemSize = bytesPerElement(data_->dtype_);
    const char* srcData = data_->hostData_.data();
    HostBuffer dstData(n * elemSize);
    std::vector<uint32_t> srcIndices(rank);
    std::vector<uint32_t> dstIndices(rank);
    for (size_t linear = 0; linear < n; ++linear)
    {
        // linear → multi-dim
        size_t remaining = linear;
        for (uint32_t i = 0; i < rank; ++i)
        {
            srcIndices[i] = remaining / srcStrides[i];
            remaining %= srcStrides[i];
        }

        // permute index
        for (uint32_t i = 0; i < rank; ++i)
            dstIndices[i] = srcIndices[permutation[i]];

        // multi-dim → linear (permuted)
        size_t dstLinear = 0;
        for (uint32_t i = 0; i < rank; ++i)
            dstLinear += dstIndices[i] * dstStrides[i];

        std::memcpy(&dstData[dstLinear * elemSize], &srcData[linear * elemSize], elemSize);
    }

    Tensor out(dstShape, data_->dtype_);
    out.data_->hostData_ = std::move(dstData);
    return out;
}


template <typename... Ts>
inline auto Tensor::split(Ts... offsets) const
{
    static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All offsets must be integral types");
    VAI_ASSERT(data_ && !data_->hostData_.empty());
    VAI_ASSERT(!shape_.empty() && data_->dtype_ != Dtype::Q4);

    constexpr size_t count = sizeof...(Ts) + 1;
    const uint32_t bounds[count] = { static_cast<uint32_t>(offsets)..., shape_[0] };

    const size_t stride = (numElements() / shape_[0]) * bytesPerElement(data_->dtype_);

    std::array<Tensor, count> parts;
    uint32_t begin = 0;
    for (size_t i = 0; i < count; ++i)
    {
        // Offsets must ascend and stay inside the axis; the last bound is its end.
        VAI_ASSERT(begin < bounds[i] && bounds[i] <= shape_[0]);

        std::vector<uint32_t> shape = shape_;
        shape[0] = bounds[i] - begin;

        parts[i] = Tensor(shape, data_->dtype_);
        parts[i].data_->hostData_.assign(data_->hostData_.begin() + begin * stride,
                                         data_->hostData_.begin() + bounds[i] * stride);
        begin = bounds[i];
    }
    return std::apply([](auto&... part) { return std::make_tuple(std::move(part)...); }, parts);
}


template <typename... Ts>
inline Tensor Tensor::concat(const Ts&... others) const
{
    static_assert((std::conjunction_v<std::is_same<Ts, Tensor>...>), "All arguments must be Tensors");
    VAI_ASSERT(data_ && !data_->hostData_.empty());
    VAI_ASSERT(!shape_.empty());

    const Tensor* parts[] = { this, &others... };

    std::vector<uint32_t> shape = shape_;
    shape[0] = 0;
    size_t bytes = 0;
    for (const Tensor* p : parts)
    {
        VAI_ASSERT(p->data_ && !p->data_->hostData_.empty());
        VAI_ASSERT(p->data_->dtype_ == data_->dtype_);
        VAI_ASSERT(p->shape_.size() == shape_.size());
        for (size_t d = 1; d < shape_.size(); ++d)   // only the leading axis adds up
            VAI_ASSERT(p->shape_[d] == shape_[d]);

        shape[0] += p->shape_[0];
        bytes += p->data_->hostData_.size();
    }

    HostBuffer dstData;
    dstData.reserve(bytes);
    for (const Tensor* p : parts)
        dstData.insert(dstData.end(), p->data_->hostData_.begin(), p->data_->hostData_.end());

    Tensor out(shape, data_->dtype_);
    out.data_->hostData_ = std::move(dstData);
    return out;
}

} // namespace vai

#endif // TENSOR_H
