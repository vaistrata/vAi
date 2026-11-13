// =============================================================================
// causal_mask.h - the attention blocks a stack sets on the host, read from a
// uniform buffer the host rewrites without re-recording.
//
// The block ends are prompt rows; a dispatch clips them to its own Tq, and its
// last block ends there. Bit (count - 1 - k) marks block k causal, so the first
// block holds the highest bit. causalSpan > 0 bounds every query from below to
// its last causalSpan keys, itself included (a sliding window).
//
// Include after the push constants, which carry Tq, with CAUSAL_MASK_BINDING
// defined and GL_EXT_scalar_block_layout required for the std430 block.
// =============================================================================
#ifndef VAI_COMMON_CAUSAL_MASK_H
#define VAI_COMMON_CAUSAL_MASK_H

layout(set = 0, binding = CAUSAL_MASK_BINDING, std430) uniform CAUSAL_MASK {
    uint causalBlockCount;
    uint causalBits;
    uint causalBlockEnd[MAX_CAUSAL_BLOCKS - 1];
    uint causalSpan;
};

// The new tokens a query at `t` may read: past itself only when its block is
// causal, otherwise to the end of its block.
uint newTokenEnd(uint t)
{
    // A null mask reads back 0 blocks: no mask, every new token.
    if (causalBlockCount == 0u)
        return Tq;

    uint k = 0u, end = Tq;
    for (; k + 1u < causalBlockCount; ++k)
    {
        uint e = min(causalBlockEnd[k], Tq);
        if (t < e) { end = e; break; }
    }
    return ((causalBits >> (causalBlockCount - 1u - k)) & 1u) != 0u ? t + 1u : end;
}

// The first key a query at key position `p` may read: 0 without a span (a null
// mask reads back 0), else the oldest of its last causalSpan keys.
uint keyStart(uint p)
{
    return (causalSpan == 0u || p < causalSpan) ? 0u : p + 1u - causalSpan;
}

#endif // VAI_COMMON_CAUSAL_MASK_H
