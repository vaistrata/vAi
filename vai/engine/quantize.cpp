#include "quantize.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>


namespace vai {

// A quantized buffer holds every element's quant, row-major, then an fp16 scale
// per block. Returns the bytes the quants take, which is where the scales start.
static constexpr size_t quantBytes(Dtype dt, size_t count) noexcept
{
    return dt == Dtype::Qint8 ? count : 0;
}

static_assert(
    bytesForElements(Dtype::Qint8, quantizeBlock) == quantBytes(Dtype::Qint8, quantizeBlock) + sizeof(uint16_t),
    "bytesForElements() and the layout here disagree on a Qint8 block");


// Keeps eight running maxima so that the loop vectorizes. A NaN never wins a
// comparison, so it is tracked separately and returned.
static inline float blockAmax(const float* x)
{
    float m[8] = {};
    bool nan = false;
    for (size_t j = 0; j < quantizeBlock; j += 8)
        for (size_t l = 0; l < 8; ++l)
        {
            const float a = std::fabs(x[j + l]);
            m[l] = a > m[l] ? a : m[l];
            nan |= a != a;
        }
    if (nan)
        return std::numeric_limits<float>::quiet_NaN();
    float amax = m[0];
    for (size_t l = 1; l < 8; ++l)
        amax = std::max(amax, m[l]);
    return amax;
}


// Rounds to nearest even and keeps subnormals. Tensor's fp16 conversion flushes
// them to zero, which would lose a small scale.
static uint16_t scaleToFp16(float scale)
{
    const uint32_t b = std::bit_cast<uint32_t>(scale);
    const uint32_t a = b & 0x7FFFFFFFu;
    uint32_t h;
    if (a >= 0x47800000u)   // 65536 and up: inf, NaN stays NaN
        h = a > 0x7F800000u ? 0x7E00u : 0x7C00u;
    else if (a < 0x38800000u)   // under 2^-14: + 0.5 rounds to the subnormal ulp
        h = std::bit_cast<uint32_t>(std::bit_cast<float>(a) + 0.5f) - 0x3F000000u;
    else   // exponent rebiased, mantissa rounded
        h = (a + 0xC8000FFFu + ((a >> 13) & 1u)) >> 13;
    return uint16_t(((b >> 16) & 0x8000u) | h);
}


// Quantizes one block: fills its quants and returns its scale.
static float quantizeQint8(const float* x, int8_t* q)
{
    const float amax = blockAmax(x);
    if (!std::isfinite(amax))   // quantizeBlocks() reports it
        return amax;

    const float scale = amax / 127.0f;
    if (!scaleToFp16(scale))   // too small to store: the block is zeros
    {
        std::fill_n(q, quantizeBlock, int8_t(0));
        return scale;
    }

    const float invScale = 1.0f / scale;
    // round() written as trunc plus its remainder so that the loop vectorizes.
    for (size_t j = 0; j < quantizeBlock; ++j)
    {
        const float v = x[j] * invScale;
        const int t = int(v);
        const float f = v - float(t);
        q[j] = int8_t(t + int(f >= 0.5f) - int(f <= -0.5f));
    }
    return scale;
}


bool quantizeBlocks(Dtype dt, const float* src, size_t count, void* buffer, size_t total, size_t first)
{
    VAI_ASSERT(dt == Dtype::Qint8);
    VAI_ASSERT(count % quantizeBlock == 0 && first % quantizeBlock == 0 && first + count <= total);

    auto quants = static_cast<int8_t*>(buffer);
    auto scales = reinterpret_cast<uint16_t*>(quants + quantBytes(dt, total)) + first / quantizeBlock;
    int8_t* q = quants + quantBytes(dt, first);
    bool finite = true;
    for (size_t b = 0; b < count / quantizeBlock; ++b, src += quantizeBlock, q += quantizeBlock)
    {
        // The quants are rounded with the fp32 scale; the stored scale is fp16.
        const float scale = quantizeQint8(src, q);
        finite &= std::isfinite(scale);
        scales[b] = scaleToFp16(scale);
    }
    return finite;
}


}   // namespace vai
