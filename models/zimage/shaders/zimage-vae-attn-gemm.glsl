// =============================================================================
// zimage-vae-attn-gemm.glsl: the two products of the VAE mid-block attention,
//     O = alpha · A · B over fp32 matrices; zimage-vae-attn-scores.comp and
//     zimage-vae-attn-values.comp include it with A's layout defined
//
//     O[m, n] = alpha · Σ_k A(m, k) · B[k, n]
//
//   scores (A_KMAJOR 1): A [K, M] (k-major): S[key, q] = K[:, key] · Q[:, q] / sqrt(D)
//   values (A_KMAJOR 0): A [M, K] (m-major): O[c, q]   = Σ_key V[c, key] · P[key, q]
//
// B, O and both A layouts are channel-major slices of the decoder's maps, so
// the attention needs no transposes. Direct, without shared memory (the
// frame of zimage-vae-conv.glsl): each invocation owns tM = 8 rows x tN = 4
// neighbouring columns as vec4 accumulators and steps k four at a time; a
// workgroup's invocations share the rows, so A's loads are uniform and B's
// coalesce. Compile-time tile and layout (Adreno 830 keeps spec-constant
// loops rolled).
//
// - Offsets and leading dimensions in floats, all multiples of 4, as are
//   K, N and (A_KMAJOR) M.
// - Dispatch (host): .dispatch(ceil(N / (tN * wgSize)), ceil(M / tM)).
// =============================================================================
const uint wgSize = 64u;
const uint tM = 8u;     // rows per invocation
const uint tN = 4u;     // columns per invocation (one vec4)

layout(local_size_x = wgSize, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) writeonly buffer O_BUF { vec4 bufferO[]; };
layout(set = 0, binding = 1) readonly buffer A_BUF { vec4 bufferA[]; };
layout(set = 0, binding = 2) readonly buffer B_BUF { vec4 bufferB[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint N;
    uint K;
    uint offA;     // A(m, k) = A[offA + k * M + m] (k-major) or A[offA + m * K + k]
    uint ldB;      // B[k, n] = B[offB + k * ldB + n]
    uint offB;
    uint ldO;      // O[m, n] = O[offO + m * ldO + n]
    uint offO;
    float alpha;
};

void main()
{
    const uint n0 = (gl_WorkGroupID.x * wgSize + gl_LocalInvocationID.x) * tN;
    if (n0 >= N) return;
    const uint m0 = gl_WorkGroupID.y * tM;

    vec4 acc[tM];
    [[unroll]] for (uint i = 0; i < tM; ++i) acc[i] = vec4(0.0);

#if !A_KMAJOR
    uint rowA[tM];   // tail rows read the last one; they are not stored
    [[unroll]] for (uint i = 0; i < tM; ++i) rowA[i] = (offA + min(m0 + i, M - 1u) * K) / 4u;
#endif

    for (uint k = 0; k < K; k += 4u)
    {
        vec4 b[4];
        [[unroll]] for (uint t = 0; t < 4u; ++t)
            b[t] = bufferB[(offB + (k + t) * ldB + n0) / 4u];
#if A_KMAJOR
        [[unroll]] for (uint t = 0; t < 4u; ++t)
        {
            const uint base = (offA + (k + t) * M + m0) / 4u;
            const vec4 a0 = bufferA[base];
            const vec4 a1 = (m0 + 4u < M) ? bufferA[base + 1u] : vec4(0.0);
            acc[0] += a0.x * b[t]; acc[1] += a0.y * b[t]; acc[2] += a0.z * b[t]; acc[3] += a0.w * b[t];
            acc[4] += a1.x * b[t]; acc[5] += a1.y * b[t]; acc[6] += a1.z * b[t]; acc[7] += a1.w * b[t];
        }
#else
        const uint kq = k / 4u;
        [[unroll]] for (uint i = 0; i < tM; ++i)
        {
            const vec4 a = bufferA[rowA[i] + kq];
            acc[i] += a.x * b[0] + a.y * b[1] + a.z * b[2] + a.w * b[3];
        }
#endif
    }

    [[unroll]] for (uint i = 0; i < tM; ++i)
        if (m0 + i < M)
            bufferO[(offO + (m0 + i) * ldO + n0) / 4u] = alpha * acc[i];
}
