// =============================================================================
// zimage-vae-conv.glsl: stride-1 "same" convolution, with the VAE decoder's
//     neighbours fused in; zimage-vae-conv3x3.comp and zimage-vae-conv1x1.comp
//     include it with the kernel size (KS) defined
//
//     O[co, y, x] = b[co] + Σ_{ci,ky,kx} W[co, ci, ky, kx] · I'[ci, y+ky-r, x+kx-r]
//                   (+ R[co, y, x])
//
// I' is I, or I upsampled 2x nearest (upsample = 1: I'[c, y, x] =
// I[c, y/2, x/2], diffusers' Upsample2D folded into the conv that follows
// it), zero outside the map; r = (ks - 1) / 2. R is an optional residual
// (a ResNet block's shortcut, the attention's input).
//
// Direct, without shared memory (the frame of vai/shaders/linear-q4-direct):
// each invocation owns tM = 8 output channels x tN = 4 neighbouring
// positions of one row. Per input channel and kernel row it reads the
// tN + ks - 1 input values that row needs once and reuses them for every kx
// and all 8 channels; the 8 channels' weights of one tap are one 16-byte
// load from W packed as [Ci, ks*ks, Cop] (VaeConvNode::packWeight; Cop = Co
// rounded up to 8, zero-filled). A workgroup's invocations share the channels
// and cover consecutive positions, so the weight loads are uniform and the
// input loads coalesce. Tile and kernel size are compile-time: as
// specialization constants Adreno 830's compiler keeps the loops rolled.
//
// - I (typeI), O and R (typeO) fp32 or fp16; W, b fp16; accumulation fp32.
// - Wo % 4 == 0; any Co, Ci, H.
// - Dispatch (host): .dispatch(ceil(Ho*Wo / (tN * wgSize)), Cop / tM).
// =============================================================================
#include "common/spec_constants.h"

const uint wgSize = 64u;
const uint tM = 8u;             // output channels per invocation
const uint tN = 4u;             // positions per invocation (along x)
const uint ks = KS;             // kernel size, from the including .comp
const uint nV = tN + ks - 1u;   // input values a kernel row reads
const int r = int(ks / 2u);

layout(local_size_x = wgSize, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint upsample = 0;     // 1: input read 2x nearest-upsampled
layout(constant_id = 1) const uint hasResidual = 0;  // 1: + R
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;

layout(set = 0, binding = 0) writeonly buffer O_BUF_F32 { float bufferO_f32[]; };   // [Co, Ho, Wo]
layout(set = 0, binding = 0) writeonly buffer O_BUF_F16 { float16_t bufferO_f16[]; };
layout(set = 0, binding = 1) readonly buffer I_BUF_F32 { float bufferI_f32[]; };    // [Ci, Hi, Wi]
layout(set = 0, binding = 1) readonly buffer I_BUF_F16 { float16_t bufferI_f16[]; };
layout(set = 0, binding = 2) readonly buffer W_BUF { uvec4 bufferW[]; };            // [Ci, ks*ks, Cop] fp16, 8 per uvec4
layout(set = 0, binding = 3) readonly buffer B_BUF { float16_t bufferB[]; };        // [Cop]
layout(set = 0, binding = 4) readonly buffer R_BUF_F32 { float bufferR_f32[]; };    // [Co, Ho, Wo]
layout(set = 0, binding = 4) readonly buffer R_BUF_F16 { float16_t bufferR_f16[]; };

layout(push_constant) uniform PushConstants {
    uint Ci;
    uint Co;
    uint Hi;       // input map
    uint Wi;
    uint Wo;       // output map width (Wi, or 2 Wi when upsampling); Ho likewise
    uint Ho;
};

float readI(uint i) { return (typeI == DTYPE_F32) ? bufferI_f32[i] : float(bufferI_f16[i]); }
float readR(uint i) { return (typeO == DTYPE_F32) ? bufferR_f32[i] : float(bufferR_f16[i]); }
void writeO(uint i, float v) { if (typeO == DTYPE_F32) bufferO_f32[i] = v; else bufferO_f16[i] = float16_t(v); }

void main()
{
    const uint P = Ho * Wo;
    const uint p0 = (gl_WorkGroupID.x * wgSize + gl_LocalInvocationID.x) * tN;
    if (p0 >= P) return;
    const uint g = gl_WorkGroupID.y;          // channel group: co0 = g * tM
    const uint co0 = g * tM;
    const uint groups = gl_NumWorkGroups.y;   // Cop / tM
    const int y = int(p0 / Wo), x0 = int(p0 % Wo);

    float acc[tM][tN];
    [[unroll]] for (uint i = 0; i < tM; ++i)
        [[unroll]] for (uint j = 0; j < tN; ++j)
            acc[i][j] = 0.0;

    // Input columns this invocation reads (x0 - r .., in I' coordinates),
    // mapped to I's and zeroed outside the map.
    uint col[nV];
    bool colIn[nV];
    [[unroll]] for (uint j = 0; j < nV; ++j)
    {
        const int x = x0 + int(j) - r;
        colIn[j] = x >= 0 && x < int(Wo);
        col[j] = (upsample != 0u) ? uint(max(x, 0)) / 2u : uint(max(x, 0));
    }

    for (uint ci = 0; ci < Ci; ++ci)
    {
        [[unroll]] for (uint ky = 0; ky < ks; ++ky)
        {
            const int yy = y + int(ky) - r;
            if (yy < 0 || yy >= int(Ho)) continue;
            const uint rowBase = (ci * Hi + ((upsample != 0u) ? uint(yy) / 2u : uint(yy))) * Wi;

            float v[nV];
            [[unroll]] for (uint j = 0; j < nV; ++j)
                v[j] = colIn[j] ? readI(rowBase + col[j]) : 0.0;

            [[unroll]] for (uint kx = 0; kx < ks; ++kx)
            {
                const uvec4 wq = bufferW[(ci * ks * ks + ky * ks + kx) * groups + g];
                const vec2 w01 = unpackHalf2x16(wq.x), w23 = unpackHalf2x16(wq.y);
                const vec2 w45 = unpackHalf2x16(wq.z), w67 = unpackHalf2x16(wq.w);
                const float w[tM] = float[tM](w01.x, w01.y, w23.x, w23.y, w45.x, w45.y, w67.x, w67.y);
                [[unroll]] for (uint i = 0; i < tM; ++i)
                    [[unroll]] for (uint j = 0; j < tN; ++j)
                        acc[i][j] += w[i] * v[j + kx];
            }
        }
    }

    [[unroll]] for (uint i = 0; i < tM; ++i)
    {
        const uint co = co0 + i;
        if (co >= Co) break;
        const float bias = float(bufferB[co]);
        [[unroll]] for (uint j = 0; j < tN; ++j)
        {
            const uint p = p0 + j;
            float o = acc[i][j] + bias;
            if (hasResidual != 0u) o += readR(co * P + p);
            writeO(co * P + p, o);
        }
    }
}
