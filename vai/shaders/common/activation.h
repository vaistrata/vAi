// =============================================================================
// common/activation.h:
//     Activation epilogue — the `epilogue` spec constant and helper
//     functions, shared across compute shaders.
//
// - Declares `epilogue` at SPEC_ID_EPILOGUE (reserved): an including shader
//   must not reuse that spec constant ID for any other purpose.
//   epilogue : 0 none, 1 relu, 2 gelu, 3 gelu (tanh), 4 silu, 5 sigmoid
// =============================================================================
#ifndef VAI_COMMON_ACTIVATION_H
#define VAI_COMMON_ACTIVATION_H
#include "spec_constants.h"

// clang-format off
layout(constant_id = SPEC_ID_EPILOGUE) const uint epilogue = 0;   // 0: none, 1: relu, 2: gelu, 3: gelu (tanh), 4: silu, 5: sigmoid
// clang-format on

// GLSL has no erf: Abramowitz-Stegun 7.1.26, within 1.5e-7 of it.
float erfApprox(float x)
{
    const float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    const float p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign(x) * (1.0 - p * exp(-x * x));
}
vec4 erfApprox(vec4 x)
{
    const vec4 t = vec4(1.0) / (vec4(1.0) + 0.3275911 * abs(x));
    const vec4 p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign(x) * (vec4(1.0) - p * exp(-x * x));
}
float gelu(float x) { return 0.5 * x * (1.0 + erfApprox(x * 0.707106781186547524400844)); }
vec4 gelu(vec4 x)   { return 0.5 * x * (vec4(1.0) + erfApprox(x * 0.707106781186547524400844)); }
float geluTanh(float x)
{
    const float c = 0.797884560802865355879892;   // sqrt(2/pi)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}
vec4 geluTanh(vec4 x)
{
    const float c = 0.797884560802865355879892;   // sqrt(2/pi)
    return 0.5 * x * (vec4(1.0) + tanh(c * (x + 0.044715 * x * x * x)));
}
float silu(float x)    { return x / (1.0 + exp(-x)); }
vec4 silu(vec4 x)      { return x / (vec4(1.0) + exp(-x)); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
vec4 sigmoid(vec4 x)   { return vec4(1.0) / (vec4(1.0) + exp(-x)); }

float applyEpilogue(float x)
{
    // clang-format off
    return (epilogue == 0) ? x :
           (epilogue == 1) ? max(x, 0.0) :
           (epilogue == 2) ? gelu(x) :
           (epilogue == 3) ? geluTanh(x) :
           (epilogue == 4) ? silu(x) :
           (epilogue == 5) ? sigmoid(x) : x;
    // clang-format on
}

vec4 applyEpilogue(vec4 x)
{
    // clang-format off
    return (epilogue == 0) ? x :
           (epilogue == 1) ? max(x, vec4(0.0)) :
           (epilogue == 2) ? gelu(x) :
           (epilogue == 3) ? geluTanh(x) :
           (epilogue == 4) ? silu(x) :
           (epilogue == 5) ? sigmoid(x) : x;
    // clang-format on
}

#endif // VAI_COMMON_ACTIVATION_H
