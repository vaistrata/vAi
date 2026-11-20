// =============================================================================
// common/linear-store.h:
//     The epilogue of the Linear tile kernels: what linear-reduce.comp
//     does after a split, done by the kernel's own store when K is not split.
//
//   O[m, n] = s * activation(C[m, n] + B[m, n]) + R[m, n]     residualOrder 0
//   O[m, n] = s * activation(C[m, n] + B[m, n] + R[m, n])     residualOrder 1
//   GLU (C's N = 2H columns hold gate and up, O is [M, H]; no residual / scale):
//   O[m, c] = activation(gate + B[m, c]) * (up + B[m, H + c])
//
// B (typeB) is [N] or [M, N], or rows that repeat (biasElems); in the GLU it is
// in the weight's original order, gate H then up H. R is typeO, [M, N] or
// repeating rows (residualElems). Every constant defaults to the identity
// (no bias, no residual, s = 1, activation none), so a kernel built without them
// stores C as it is: the partials of a split, or a plain product.
//
// The including shader declares typeO (SPEC_ID_TYPE_O) and defines
// LINEAR_STORE_BIAS_BINDING and LINEAR_STORE_RESIDUAL_BINDING first. It also
// gets common/activation.h (activation at SPEC_ID_ACT_TYPE).
// =============================================================================
#ifndef VAI_COMMON_LINEAR_STORE_H
#define VAI_COMMON_LINEAR_STORE_H
#include "activation.h"
#include "spec_constants.h"

#if !defined(LINEAR_STORE_BIAS_BINDING) || !defined(LINEAR_STORE_RESIDUAL_BINDING)
#error "define LINEAR_STORE_BIAS_BINDING and LINEAR_STORE_RESIDUAL_BINDING before including linear-store.h"
#endif

// clang-format off
layout(constant_id = SPEC_ID_HAS_BIAS)       const uint storeHasBias       = 0;            // 1: B
layout(constant_id = SPEC_ID_BIAS_ELEMS)     const uint storeBiasElems     = 0;            // 0: B covers O; else it repeats
layout(constant_id = SPEC_ID_TYPE_B)         const uint storeTypeB         = 0;            // B's dtype
layout(constant_id = SPEC_ID_HAS_RESIDUAL)   const uint storeHasResidual   = 0;            // 1: R
layout(constant_id = SPEC_ID_RESIDUAL_ELEMS) const uint storeResidualElems = 0;            // 0: R covers O; else it repeats
layout(constant_id = SPEC_ID_RESIDUAL_ORDER) const uint storeResidualOrder = 0;            // 0: after the activation, 1: before
layout(constant_id = SPEC_ID_SCALE_BITS)     const uint storeScaleBits     = 0x3F800000u;  // s as float bits (1.0)

layout(set = 0, binding = LINEAR_STORE_BIAS_BINDING)     readonly buffer STORE_BIAS_F32 { float     storeBias_f32[]; };
layout(set = 0, binding = LINEAR_STORE_BIAS_BINDING)     readonly buffer STORE_BIAS_F16 { float16_t storeBias_f16[]; };
layout(set = 0, binding = LINEAR_STORE_RESIDUAL_BINDING) readonly buffer STORE_R_F32    { float     storeR_f32[]; };
layout(set = 0, binding = LINEAR_STORE_RESIDUAL_BINDING) readonly buffer STORE_R_F16    { float16_t storeR_f16[]; };
// clang-format on

// The store is C as it is: no bias, residual, scale or activation (a kernel may keep a faster path for it).
const bool linearStoreIdentity =
    storeHasBias == 0u && storeHasResidual == 0u && storeScaleBits == 0x3F800000u && actType == 0u;

// B and R at output index i = m * N + n (N: the row length of C).
float linearStoreBias(uint i)
{
    if (storeHasBias == 0u) return 0.0;
    const uint j = (storeBiasElems == 0u) ? i : i % storeBiasElems;
    return (storeTypeB == DTYPE_F32) ? storeBias_f32[j] : float(storeBias_f16[j]);
}
float linearStoreResidual(uint i)
{
    if (storeHasResidual == 0u) return 0.0;
    const uint j = (storeResidualElems == 0u) ? i : i % storeResidualElems;
    return (typeO == DTYPE_F32) ? storeR_f32[j] : float(storeR_f16[j]);
}

// O at output index i = m * N + n, from C there.
float linearStore(uint i, float c)
{
    const float s = uintBitsToFloat(storeScaleBits), v = c + linearStoreBias(i), r = linearStoreResidual(i);
    return (storeResidualOrder == 1u) ? s * applyActivation(v + r) : s * applyActivation(v) + r;
}

// The GLU's O[m, c] from its gate and up (C's N = 2H).
float linearStoreGlu(uint m, uint c, uint N, float gate, float up)
{
    return applyActivation(gate + linearStoreBias(m * N + c)) * (up + linearStoreBias(m * N + N / 2u + c));
}

#endif // VAI_COMMON_LINEAR_STORE_H
