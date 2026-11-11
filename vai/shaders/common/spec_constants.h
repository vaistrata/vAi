#ifndef VAI_COMMON_SPEC_CONSTANTS_H
#define VAI_COMMON_SPEC_CONSTANTS_H

// vai::Dtype as a dtype specialization constant carries it, so a kernel names
// the format instead of the enum position.
#define DTYPE_F32 0u
#define DTYPE_F16 1u
#define DTYPE_NONE 2u
#define DTYPE_Q4 3u     // weights only (layout at vai::Dtype::Q4)

// Shape specialization constants: 30-36.
#define SPEC_ID_TYPE_CI 30
#define SPEC_ID_TYPE_CO 31
#define SPEC_ID_TYPE_HI 32
#define SPEC_ID_TYPE_HO 33
#define SPEC_ID_TYPE_WI 34
#define SPEC_ID_TYPE_WO 35
#define SPEC_ID_TYPE_N 36

// Common data-path specialization constants: 100-105.
#define SPEC_ID_TYPE_O 100
#define SPEC_ID_TYPE_I 101
#define SPEC_ID_TYPE_W 102
#define SPEC_ID_ALIGNED 103
#define SPEC_ID_TYPE_P 104
#define SPEC_ID_TYPE_ACC 105

// Common operation specialization constants: 110+.
#define SPEC_ID_EPILOGUE 110
#define SPEC_ID_HAS_SCALE 111
#define SPEC_ID_HAS_BIAS 112

// Shared shape limits: sized arrays a push-constant or uniform block and its host
// mirror must agree on.
#define PERMUTE_MAX_RANK 8
#define MAX_CAUSAL_BLOCKS 32   // one causal bit per block in a uint

#endif // VAI_COMMON_SPEC_CONSTANTS_H
