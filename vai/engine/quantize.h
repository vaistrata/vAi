#ifndef VAI_ENGINE_QUANTIZE_H
#define VAI_ENGINE_QUANTIZE_H

#include "tensor.h"
#include <cstdint>

namespace vai {

// The shaders use these values as QUANTIZE_* (spec_constants.h).
enum class QuantizeType { None, Qint8 };

constexpr Dtype dtypeFromQuantizeType(QuantizeType type, Dtype wDefaultDtype) noexcept
{
    switch (type)
    {
    case QuantizeType::Qint8:
        return Dtype::Qint8;
    case QuantizeType::None:
        break;
    }
    return wDefaultDtype;
}

constexpr QuantizeType quantizeTypeFromDtype(Dtype dt) noexcept
{
    return dt == Dtype::Qint8 ? QuantizeType::Qint8 : QuantizeType::None;
}

// Number of elements along a row that share one scale. K is a multiple of it.
inline constexpr uint32_t quantizeBlock = 32;

// Fills elements [first, first + count) of a buffer laid out for `total` with the
// quantized `count` fp32 values. Returns false if a value is not finite.
bool quantizeBlocks(Dtype dt, const float* src, size_t count, void* buffer, size_t total, size_t first);

}   // namespace vai

#endif   // VAI_ENGINE_QUANTIZE_H
