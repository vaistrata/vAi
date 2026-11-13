#ifndef VAI_NODES_ATTENTION_H
#define VAI_NODES_ATTENTION_H

// Attention (separate-Q/K/V KV-cache attention, GQA-capable;
// the FA kernels are ports of the ggml/llama.cpp Vulkan flash_attn kernels).
#include "node-common.h"
#include <algorithm>
#include <array>
#include <cstring>

namespace vai {


// The tokens the KV cache already holds, which attention and RoPE read. A model
// owns it and rewrites it after sync(), before each forward.
inline eva::Buffer createKvCacheCounter(eva::Device device)
{
    eva::Buffer buffer = device.createBuffer({
        .size = sizeof(uint32_t),
        .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT,
    });
    *(uint32_t*)buffer.map() = 0;
    return buffer;
}


using CausalBlockEnd = std::array<uint32_t, MAX_CAUSAL_BLOCKS - 1>;

// The host mirror of shaders/common/causal_mask.h. A model owns the buffer, as it
// owns the KV-cache counter, and hands it to every attention it masks: an
// attention without one binds it null, which reads as no mask at all.
struct CausalMask
{
    uint32_t blockCount;
    uint32_t bits;
    CausalBlockEnd blockEnd;
    uint32_t span;
};

// bits marks each block causal, the first block in the highest of blockCount
// bits; blockEnd lists every end but the last, which is the prompt's, and
// stops at its first 0. span > 0 limits every query to its last span keys, a
// sliding window. Rewriting a mask the GPU is reading is a race: write it
// after sync().
inline void writeCausalMask(eva::Buffer buffer, uint32_t bits, const CausalBlockEnd& blockEnd = {}, uint32_t span = 0)
{
    CausalMask mask{ .blockCount = 1, .bits = bits, .blockEnd = blockEnd, .span = span };
    while (mask.blockCount < MAX_CAUSAL_BLOCKS && blockEnd[mask.blockCount - 1] != 0)
    {
        VAI_ASSERT(mask.blockCount == 1 || blockEnd[mask.blockCount - 1] > blockEnd[mask.blockCount - 2]);
        ++mask.blockCount;
    }
    std::memcpy(buffer.map(), &mask, sizeof(mask));
}

inline eva::Buffer createCausalMask(eva::Device device, uint32_t bits, const CausalBlockEnd& blockEnd = {},
                                    uint32_t span = 0)
{
    eva::Buffer buffer = device.createBuffer(
        { .size = sizeof(CausalMask),
          .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
          .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT });
    writeCausalMask(buffer, bits, blockEnd, span);
    return buffer;
}


}   // namespace vai


namespace vai::node {


/*=============================================================================
  Attention - fused KV cache append + causal attention (GQA),
  split between three kernels
  Input : Q [..., Hq, Tq, headDim], K / V [..., Hkv, Tkv, headDim]
  Output: out0 attention [..., Tq, Hq, headDim]

  Leading axes are more sequences swept in one dispatch; they ride the grid,
  so every count below stays per-sequence.

  Q, K, V arrive as three separate buffers (so callers can transform Q/K but
  not V in between — e.g. RoPE). Supports GQA (numKVHeads < numQHeads).

  - coopmat FA (attention-fa-cm): subgroup-MMA flash attention,
    fixed 16x64 tile; prefill and GQA-packed decode on cm-capable devices.
  - scalar FA (attention-fa): flash attention with per-(shape,
    vendor) tuned tiles; everything the cm gates reject.
  - Intel FA (attention-fa-intel): Intel parts without a coopmat path;
    a 16-row tile packs one KV group's heads, subgroup-broadcast products.
  - naive (attention): the unconditional fallback — unfamiliar
    devices degrade instead of crashing. See ggmlFATuningParams().

  The path and tuning are selected per (shape, device) in prepare(); a tuning
  change re-requests the pipeline from the pool. Every path leaves the KV
  cache in the identical state (fused append), so the graph meaning does not
  depend on which one runs.

  The mask's span bounds a query to its last span keys, a sliding window
  (e.g. Gemma3's local layers). A ring cache (kvCacheRing) then keeps only the
  last kvCacheSize positions, position p at row p % kvCacheSize; the counter
  keeps counting positions. A dispatch overwrites the rows of its Tkv oldest
  positions, so kvCacheSize >= span + Tkv - 1 keeps every query's span whole.
  The node rounds a ring up to kvRingAlign, so no key tile wraps it.

  kvCacheSize = 0 is orthogonal to both: it drops the cache itself, so a
  dispatch attends its own tokens and nothing else (an encoder, e.g. SigLIP).
  Nothing is allocated and no counter is wired — the cache and counter
  descriptors go null, and the 0 a null counter reads back is the whole trick.

  Tkv may differ from Tq: queries and keys align at their ends, so query t
  reaches the keys up to its own mask end shifted by Tkv - Tq (e.g. a decoder
  over its encoder's output, Tq queries on Tkv keys).
  With a cache, K / V may stay unwired (Tkv = 0): the dispatch attends the
  cache alone and appends nothing, and the model keeps the counter at the
  rows held.

  No mask (setCausalMask never called) binds the mask null: every query
  attends every key it reaches.

  Split-K (decode): when the grid would underfill the GPU, the cache sweep is
  split into kCount segments accumulating partials (partialO/partialML) that
  a second pass (attention-fa-reduce) folds with the online-
  softmax merge. kCount is the dispatch grid, so it holds for the whole
  recording and is derived from occupancy alone; where the segments fall
  follows the cache, which grows under a recording, so the kernel derives that
  from kvCacheCount per forward.
=============================================================================*/
class Attention : public vai::Node
{
    eva::ComputePipeline pipeNaive;     // Fallback: naive attention, runs on any device
    eva::ComputePipeline pipeFA;        // Flash attention (scalar or coopmat kernel)
    eva::ComputePipeline pipeReduce;    // Split-K second pass (merges partial results)
    eva::DescriptorSet descSetNaive;
    eva::DescriptorSet descSetFA;
    eva::DescriptorSet descSetReduce;

    uint32_t Hkv;   // derived: numKVHeads > 0 ? numKVHeads : numQHeads
    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;

public:
    // FA tuning = ggml vk_fa_tuning_params (wgRows/wgKv/dhSplit/rowGroups =
    // ggml Br/Bc/d_split/row_split).
    struct PipeParamsFA {
        uint32_t   wgSize       = 0;
        uint32_t   sgSize       = 0;      // forced subgroup size (0 = don't force; Intel)
        uint32_t   wgRows       = 0;      // Q rows per tile
        uint32_t   wgKv         = 0;      // KV columns per tile step
        uint32_t   dhSplit      = 0;      // adjacent lanes splitting the head dim
        uint32_t   rowGroups    = 0;      // row-groups per workgroup (1 or 4)
        uint32_t   limitShmem   = 0;      // dead shared (vec4s) to cap occupancy (AMD)
        bool       coopmat      = false;  // cm kernel instead of the scalar one
        bool       shmemStaging = false;  // stage K/V tiles through shared memory
        bool       stagingF16   = false;  // scalar: f16 staging (= ggml FLOAT_TYPE)
        bool       accF16       = false;  // f16 O accumulator regs
        bool       appendBeforeSweep = false;  // cm packed Tq==1 decode: append first, then
                                               // sweep the cache as the single source (spec 12)
        bool intel = false;                    // attention-fa-intel instead of the ggml kernels
        uint32_t tileTokens = 0;               // intel: query tokens per tile (x the group's heads)
        bool qkAccF32 = false;                 // intel: Q K^T summed in fp32 (else fp16 within 32 dims)
        vai::Dtype iDtype       = vai::Dtype::None;

        bool valid() const { return wgSize != 0; }
        bool operator==(const PipeParamsFA&) const = default;
    };

    struct PipeParamsReduce {
        uint32_t blockSize = 0;

        bool operator==(const PipeParamsReduce&) const = default;
    };

    struct PipeParamsNaive {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParamsNaive&) const = default;
    };

    struct Config {
        uint32_t headDim;
        uint32_t numQHeads;
        uint32_t numKVHeads = 0; // for GQA (Hkv < numQHeads); 0 → Hkv = numQHeads
        uint32_t kvCacheSize = 0; // tokens the cache holds; 0 = no cache at all
        vai::Dtype kvCacheDtype = vai::Dtype::None;   // K / V cache; required when kvCacheSize > 0
        bool kvCacheRing = false;   // the cache keeps the last kvCacheSize positions, rounded up to kvRingAlign
    } config_;

    // Every path's key tile (fa / cm wgKv, Intel BK) divides it; prepare()
    // checks the tile it picks.
    static constexpr uint32_t kvRingAlign = 256;

    Attention(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
    // The mask is read at dispatch time, so rewriting it (writeCausalMask)
    // needs no re-recording.
    void setCausalMask(eva::Buffer buffer)
    {
        causalMask = buffer;
    }

    void setQkAccDtype(vai::Dtype dtype, vai::DtypeKey) { qkAccDtype_ = dtype; }

    void setOAccDtype(vai::Dtype dtype, vai::DtypeKey) { oAccDtype_ = dtype; }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        // Without a cache there is nothing to count from: the sweep would run
        // into a cache that does not exist.
        VAI_ASSERT(config_.kvCacheSize > 0);
        kvCacheCounter = buffer;
    }

    // Test / bench only: which path prepare() selected.
    bool usingFa() const { return useFa; }
    bool usingCm() const { return useFa && pipeParamsFA.coopmat; }
    bool usingIntel() const { return useFa && pipeParamsFA.intel; }
    const PipeParamsFA& getPipeParamsFA() const { return pipeParamsFA; }
    uint32_t getSplitKCount() const { return splitKCount; }
    uint32_t getGqaRatio() const { return gqaRatio; }

private:
    // Accumulators of the Intel kernel; the ggml kernels choose their own.
    vai::Dtype qkAccDtype_ = vai::Dtype::Fp16;   // Q K^T: Fp16 sums 32 head dims in fp16, the rest in fp32
    vai::Dtype oAccDtype_ = vai::Dtype::Fp32;    // O (P V)

    PipeParamsFA pipeParamsFA;
    PipeParamsReduce pipeParamsReduce;
    PipeParamsNaive pipeParamsNaive;
    bool useFa = false;
    uint32_t B = 0;             // per-forward: sequences the leading axes carry
    uint32_t Tkv = 0;           // per-forward: K / V input rows (0 = the cache alone)
    uint32_t gqaRatio = 0;      // per-forward: > 1 = GQA-packed decode (tile rows = one KV group's heads)
    uint32_t splitKCount = 1;   // KV splits = the grid multiplier
};


/////////////////////////////////////////////////////////////////////////////////////////
// Attention
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// FA tuning heuristics (adapted from ggml ggml-vulkan.cpp; upstream functions named)
/////////////////////////////////////////////////////////////////////////////////////////

// Shared-memory budget of the scalar FA kernel for one workgroup
// (vec4 data path; staging element size follows stagingF16). NOTE (vs
// llama.cpp): upstream adds mask / MMQ terms (ggml_vk_flash_attn_scalar_
// shmem_support); vai has neither.
inline bool ggmlFAScalarShmemSupport(const Attention::PipeParamsFA& t, uint32_t Dh, eva::Device device)
{
    const size_t v4 = 4 * sizeof(float);
    const size_t sv = t.stagingF16 ? v4 / 2 : v4;                           // staging element bytes
    size_t redEntries = t.sgSize == 0 ? t.wgSize                            // no-subgroup flavor: tree over the workgroup
                      : size_t(t.wgSize / t.sgSize) * t.dhSplit;            // = ggml tmpsh_size
    size_t sRed = redEntries * (sizeof(float) + v4);                        // s_red + s_redv4
    size_t sQ   = size_t(t.wgRows) * (Dh / 4 + 1) * sv;                     // Q tile (+1 bank pad)
    size_t sKv  = t.shmemStaging ? size_t(t.wgKv) * (Dh / 4 + 1) * sv : sv; // K/V staging
    size_t sOcc = size_t(t.limitShmem) * v4;                                // occupancy limiter (dead)
    return sRed + sQ + sKv + sOcc <= device.maxComputeSharedMemorySize();
}

// Shared-memory budget of the coopmat FA kernel: the shader's arrays for this
// configuration (f16 Q/K/V/P/PV, f32 S; KV_PARTS, SMALL_KV and SP_PAD as there).
inline bool ggmlFACmShmemSupport(uint32_t Dh, bool staging, bool append, eva::Device device)
{
    size_t dh4Pad  = ((Dh + 15) & ~15u) / 4;
    bool   smallKv = !staging && dh4Pad == Dh / 4 && !append;
    size_t parts   = (staging && Dh == 256) ? 2 : 1;                         // = the shader's KV_PARTS
    size_t spPad   = (append && staging) ? 2 : 0;
    size_t q  = 16 * (dh4Pad + 2) * 8;                                        // s_q (f16vec4)
    size_t kv = 64 * (smallKv ? 16 : dh4Pad / parts + 2) * 8;                 // s_kv: one head-dim part of the tile, or SMALL_KV's chunk union
    size_t sp = 64 * (16 / 4 + spPad) * 16 + 64 * (16 / 4 + spPad) * 8;       // s_s / s_p, transposed [Bc][Br/4 + SP_PAD]
    size_t pv = 16 * (dh4Pad + 2) * 8;                                        // s_pv (f16vec4)
    return q + kv + sp + pv <= device.maxComputeSharedMemorySize();
}

// Vendor screen for the attention cm kernel (ggml_vk_khr_cooperative_matrix_
// support port): drivers that report the coopmat extension but aren't trusted
// to run it well.
inline bool ggmlCoopmatAllowedAttention(eva::Device device)
{
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::INTEL:
        return device.architectureID() == eva::Architecture::INTEL_XE2 ||
               device.architectureID() == eva::Architecture::INTEL_XE3;
    case eva::VENDOR_ID::AMD:
        if (device.driverID() == eva::DRIVER_ID::AMD_PROPRIETARY
            || device.driverID() == eva::DRIVER_ID::AMD_OPEN_SOURCE)
            return device.architectureID() == eva::Architecture::AMD_POST_RDNA2;   // over-reporting drivers: RDNA3 only
        return true;   // RADV etc: trusted as-is
    default:
        return true;   // pre-Turing NVIDIA lacks the extension itself
    }
}

// Coopmat-path gate + fixed tuning; an invalid result falls back to the
// scalar kernel (= ggml get_fa_tuning_params_coopmat1). nRows = the PACKED
// row count when GQA-packing applies. NVIDIA Turing is demoted to scalar like
// upstream: its compiler miscompiles the cm1 FA shader (flash_attn_cm1.comp;
// reported upstream).
inline Attention::PipeParamsFA ggmlFACmTuningParams(uint32_t Dh, uint32_t nRows, eva::Device device)
{
    Attention::PipeParamsFA t{};    // invalid until fully validated

    if (nRows <= 1u) return {};                  // scalar wins single rows
    if (Dh % 8u != 0u || Dh < 16u || Dh > 256u) return {};
    if (!ggmlCoopmatAllowedAttention(device) || !device.supportsCooperativeMatrix()) return {};
    if (device.architectureID() == eva::Architecture::NVIDIA_TURING) return {};   // Turing compiler bug (see above)

    // The kernel needs both accumulator shapes: f32 for GEMM1 (S) and f16 for
    // GEMM2 (PV, fixed f16 = upstream O_TYPE). No f16-acc shape -> no cm path,
    // same as upstream (coopmat_support_16x16x16_f16acc).
    bool shapeOk = false;    // 16x16x16, f16 A/B, f32 acc, subgroup scope
    bool f16AccOk = false;   // + f16 acc shape
    for (const auto& p : device.cooperativeMatrixProperties())
        if (p.M == 16 && p.N == 16 && p.K == 16 &&
            p.aType == eva::COMPONENT_TYPE::FLOAT16 && p.bType == eva::COMPONENT_TYPE::FLOAT16 &&
            p.scope == eva::SCOPE::SUBGROUP)
        {
            if (p.cType == eva::COMPONENT_TYPE::FLOAT32 && p.resultType == eva::COMPONENT_TYPE::FLOAT32)
                shapeOk = true;
            if (p.cType == eva::COMPONENT_TYPE::FLOAT16 && p.resultType == eva::COMPONENT_TYPE::FLOAT16)
                f16AccOk = true;
        }
    if (!shapeOk || !f16AccOk) return {};

    uint32_t sg = device.subgroupSize();
    if (sg < 16u || (sg & (sg - 1u)) != 0u || 64u % sg != 0u) return {};

    // K/V staging on NVIDIA only (= ggml shmem_staging), Dh 256 included: the
    // shader takes that tile in two head-dim parts (KV_PARTS). Elsewhere
    // coopMatLoad reads global memory directly. Budgeted without
    // appendBeforeSweep; prepare() adds that where it fits.
    bool staging = device.vendorID() == eva::VENDOR_ID::NVIDIA;
    if (!ggmlFACmShmemSupport(Dh, staging, false, device)) return {};

    t = {
        .wgSize = 4u * sg,
        .sgSize = sg,
        .wgRows = 16u,
        .wgKv = 64u,
        .dhSplit = 1u,
        .rowGroups = 4u,
        .coopmat = true,
        .shmemStaging = staging,
    };
    t.accF16 = false;   // f32 O accumulation = llama.cpp's forced FA precision
    return t;
}

// Split-K / FlashDecoding heuristic (= the split_k block of ggml_vk_flash_attn):
// targets ~2 workgroups per shader core (pre-Xe2 Intel counts double — it
// prefers more workgroups). Splits cover every key a tile reaches, the cache
// and the new ones (= ggml); the count is unbounded, as the reduce holds no
// per-split array.
//
// NOTE (vs llama.cpp): upstream sizes a split first and derives the count from
// the KV length, which it can do because it rebuilds the command buffer every
// token. vai records once and lets the cache grow under that recording, so the
// derivation runs the other way: the count comes from occupancy, which a
// decode run holds constant, and the kernel sizes the splits from the live
// kvCacheCount. The two agree wherever the cache is comfortably larger
// than kCount * wgKv, and both are exact.
inline uint32_t ggmlFASplitKCount(uint32_t totalWgs, eva::Device device)
{
    uint32_t cores = device.computeBlockCount();
    if (cores != 0 && device.vendorID() == eva::VENDOR_ID::INTEL &&
        device.architectureID() != eva::Architecture::INTEL_XE2 &&
        device.architectureID() != eva::Architecture::INTEL_XE3)
        cores *= 2;               // ggml's Alchemist multiplier (not applied to the placeholder)
    if (cores == 0) cores = 16;   // ggml's placeholder when unreported
    if (totalWgs == 0 || totalWgs >= cores * 2) return 1;
    return std::max(1u, cores * 2 / totalWgs);
}

// Scalar-path tuning; an invalid result means FA does not apply — the caller
// falls back to the naive kernel, so unfamiliar devices degrade instead of
// crashing (= ggml get_fa_tuning_params_scalar).
inline Attention::PipeParamsFA ggmlFAScalarTuningParams(uint32_t Dh, uint32_t nRows, uint32_t nKv, eva::Device device)
{
    Attention::PipeParamsFA t{};    // invalid until fully validated

    if (Dh % 8u != 0u || Dh < 16u || Dh > 256u) return {};

    const uint32_t vendor = device.vendorID();
    const eva::Architecture arch = device.architectureID();
    // AMD RDNA (= upstream architecture != AMD_GCN; an undetected AMD part
    // counts as RDNA there too).
    const bool amdRdna = vendor == eva::VENDOR_ID::AMD && arch != eva::Architecture::AMD_GCN;

    // Intel: don't force a subgroup size (upstream disable_subgroups); sg=32
    // feeds the wgSize/dhSplit formulas, the pipeline gets sgSize = 0 and the
    // kernel's shared-tree epilogue flavor.
    bool noSubgroups = false;
    uint32_t sg = device.subgroupSize();
    if (vendor == eva::VENDOR_ID::INTEL) { sg = 32u; noSubgroups = true; }
    // AMD non-GCN prefers wave32 for small batches (only where the device can
    // pin it — an undetected GCN part must not be forced to 32).
    else if (amdRdna && nRows < 4u && device.minSubgroupSize() <= 32u) sg = 32u;
    if (sg < 8u || (sg & (sg - 1u)) != 0u) return {};   // sanity gate (vai addition)

    // rowGroups: only for large batch + large head size. Non-UMA RDNA raises
    // the head-size bound to 128 past 8 rows (= upstream row_split_max_hsk).
    uint32_t rowSplitMaxDh = 64u;
    if (amdRdna && device.deviceType() != eva::DEVICE_TYPE::INTEGRATED_GPU)
        rowSplitMaxDh = (nRows <= 8u) ? 64u : 128u;
    uint32_t rowGroups = (nRows < 4u || Dh <= rowSplitMaxDh) ? 1u : 4u;

    // 2 or 4 subgroups per workgroup.
    uint32_t wgSize = (sg > 32u && (nRows < 4u || Dh < (rowGroups == 1u ? 128u : 64u)))
                    ? sg * 2u : sg * 4u;

    // wgRows/wgKv; Dh&8 = head size not a multiple of 16.
    bool reduceRows = ((Dh & 8u) != 0u) || nKv < 1024u || noSubgroups;
    uint32_t wgRows, wgKv;
    if (nRows == 1u) { wgRows = 1u; wgKv = 64u; }
    else
    {
        if (rowGroups == 1u) wgRows = (nRows == 2u) ? 2u : ((nRows <= 4u || reduceRows) ? 4u : 8u);
        else                 wgRows = (nRows <= 4u) ? 4u : ((nRows <= 8u || reduceRows) ? 8u : 16u);
        wgKv = ((Dh & 8u) != 0u) ? 64u : 32u;
    }

    // dhSplit: capped at 8 and at the largest power of two dividing Dh/4.
    uint32_t lsb = Dh & (~Dh + 1u);
    uint32_t dhSplit = std::min(std::min(sg, 8u), lsb / 4u);

    // ggml stages every Dh < 256 on NVIDIA — tuned for its per-element-
    // guarded unstaged path. vai's full-tile unstaged loads beat staging at
    // Dh=64, so stage only where measured faster: 128 <= Dh < 256.
    bool staging = vendor == eva::VENDOR_ID::NVIDIA && Dh >= 128u && Dh < 256u;

    t = {
        .wgSize = wgSize,
        .sgSize = noSubgroups ? 0u : sg,
        .wgRows = wgRows,
        .wgKv = wgKv,
        .dhSplit = dhSplit,
        .rowGroups = rowGroups,
        .shmemStaging = staging,
    };

    // AMD occupancy limiter: dead shared to keep too many subgroups from
    // thrashing the cache (values = ggml's RDNA2/GCN targets). ggml applies
    // the RDNA branch to all of RDNA; vai keeps RDNA3+ out of it, as it
    // measured harmful there.
    if (vendor == eva::VENDOR_ID::AMD && device.maxComputeSharedMemorySize() == 65536u)
    {
        if (amdRdna && arch != eva::Architecture::AMD_POST_RDNA2 && nRows >= 64u && Dh <= 128u)
            t.limitShmem = (Dh <= 64u ? 26u : 30u) * 1024u / 4u / 4u;
        else if (!amdRdna && nRows <= 8u && Dh >= 256u)
            t.limitShmem = (nRows < 4u ? 14u : 26u) * 1024u / 4u / 4u;
    }
    // f16 staging needs the shaderFloat16 feature (= ggml device->fp16); eva
    // doesn't expose it, so a vendor whitelist — others keep fp32 staging.
    // TODO (eva): shaderFloat16 + storageBuffer16BitAccess feature queries
    //             (= ggml fp16_compute / fp16_storage)
    t.stagingF16 = vendor == eva::VENDOR_ID::NVIDIA || vendor == eva::VENDOR_ID::AMD
                || vendor == eva::VENDOR_ID::INTEL;

    // f16 O accumulation (spec 11). ggml's fp16 production ties O to the f16
    // staging type; vai decouples them and defaults O to fp32 (more accurate,
    // measured faster on NVIDIA/RDNA2), re-joining ggml's f16 only where it
    // measured faster: RDNA3+ small-batch big-head decode.
    // No device gate needed: the kernel's f16 exchange is uvec2-packed — no
    // shaderSubgroupExtendedTypes requirement, no OLD_AMD_WINDOWS caveat.
    // TODO (eva): shaderSubgroupExtendedTypes feature query — required before
    //             any kernel uses non-packable f16 group ops (subgroupAdd &c)
    t.accF16 = arch == eva::Architecture::AMD_POST_RDNA2 && nRows <= 8u && Dh >= 128u;

    // Structural requirements of the kernel's decomposition (see shader header).
    if (t.sgSize != 0u && t.rowGroups > 1u && t.wgSize != t.rowGroups * t.sgSize) return {};
    if (t.wgRows % t.rowGroups != 0u) return {};
    uint32_t colsPerStep = t.wgSize / t.dhSplit / t.rowGroups;
    if (colsPerStep == 0u || t.wgKv % colsPerStep != 0u) return {};
    if ((Dh / 4u) % t.dhSplit != 0u) return {};   // vec4 lane mapping (guaranteed by the lsb rule)

    // Shared-memory gate: halve wgRows once, then give up.
    if (!ggmlFAScalarShmemSupport(t, Dh, device))
    {
        if (t.wgRows < 2u || (t.wgRows / 2u) % t.rowGroups != 0u) return {};
        t.wgRows /= 2u;
        if (!ggmlFAScalarShmemSupport(t, Dh, device)) return {};
    }
    return t;
}

// Intel kernel (attention-fa-intel) for Intel parts the coopmat path leaves
// out: a 16-row tile of tileTokens tokens x the KV group's heads, subgroups of
// 16 taking 8 keys each, at most 22 KB of shared memory. The KV group has to
// fit the tile and its rows to fill it (Tq x groupSize >= 16). Invalid = the
// ggml tuners decide.
inline Attention::PipeParamsFA intelFATuningParams(uint32_t Dh, uint32_t groupSize, uint32_t Tq, Dtype qkAcc,
                                                   Dtype oAcc, eva::Device device)
{
    if (device.vendorID() != eva::VENDOR_ID::INTEL || device.architectureID() == eva::Architecture::INTEL_XE2 ||
        device.architectureID() == eva::Architecture::INTEL_XE3 || !device.supportsSubgroupExtendedTypes())
        return {};
    if (Dh % 8u != 0u || Dh < 16u || Dh > 256u || groupSize > 16u || Tq * groupSize < 16u)
        return {};

    const uint32_t wgSize = Dh <= 64u ? 128u : 256u;
    Attention::PipeParamsFA t = {
        .wgSize = wgSize,
        .sgSize = 16u,
        .wgRows = 16u,
        .wgKv = wgSize / 2u,
        .dhSplit = 1u,
        .rowGroups = 1u,
    };
    t.intel = true;
    t.tileTokens = 16u / groupSize;
    t.accF16 = oAcc == Dtype::Fp16;
    t.qkAccF32 = qkAcc == Dtype::Fp32;
    return t;
}

// Split-K for the Intel kernel: the keys split into enough spans to fill every
// Xe core, which holds 128 hardware threads (16 EUs x 8) = 128 / (wgSize / 16)
// workgroups, but no more spans than the keys a recording can reach have
// blocks: the cache's capacity, or with no cache this dispatch's own keys.
inline uint32_t intelFASplitKCount(uint32_t totalWgs, uint32_t nKv, uint32_t kvCacheSize,
                                   const Attention::PipeParamsFA& t, eva::Device device)
{
    const uint32_t target = device.computeBlockCount() * 128u / (t.wgSize / 16u);
    if (nKv == 0 || totalWgs == 0 || totalWgs >= target)
        return 1;
    const uint32_t keys = kvCacheSize > 0 ? kvCacheSize : nKv;
    return std::min(CEIL_DIV(target, totalWgs), CEIL_DIV(keys, t.wgKv));
}

// FA tuning entry: try the coopmat tile first, fall back to the scalar tuner
// (= ggml get_fa_tuning_params' cm1-first path selection; vai has no cm2
// path). Invalid result = FA not applicable -> naive.
inline Attention::PipeParamsFA ggmlFATuningParams(uint32_t Dh, uint32_t nRows, uint32_t nKv, Dtype iDtype, eva::Device device)
{
    Attention::PipeParamsFA t = ggmlFACmTuningParams(Dh, nRows, device);
    if (!t.valid()) t = ggmlFAScalarTuningParams(Dh, nRows, nKv, device);
    t.iDtype = iDtype;
    return t;
}


inline Attention::Attention(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("Q", NodeSlot::input);
    addSlot("K", NodeSlot::input);
    addSlot("V", NodeSlot::input);
    addSlot("partialO", NodeSlot::temporary);    // split-K partials, assigned when kCount > 1
    addSlot("partialML", NodeSlot::temporary);

    Hkv = config_.numKVHeads > 0 ? config_.numKVHeads : config_.numQHeads;
    VAI_ASSERT(config_.numQHeads % Hkv == 0);
    VAI_ASSERT(config_.kvCacheSize == 0 || config_.kvCacheDtype != Dtype::None);
    VAI_ASSERT(!config_.kvCacheRing || config_.kvCacheSize > 0);
    if (config_.kvCacheRing)
        config_.kvCacheSize = ROUNDUP(config_.kvCacheSize, kvRingAlign);

    // Any one input carries the node: Q has to arrive, and K / V may stay out
    // once their rows sit in the cache. prepare() holds both.
    flags_ |= Node::partialInputs;

    // Cacheless leaves the slots out: an unassigned persistent tensor fails the
    // upload sweep, and run() binds those three descriptors null instead.
    if (config_.kvCacheSize > 0)
    {
        addSlot("K_cache", NodeSlot::persistent);
        addSlot("V_cache", NodeSlot::persistent);
        (*this)["K_cache"] = Tensor({Hkv, config_.kvCacheSize, config_.headDim}, config_.kvCacheDtype);
        (*this)["V_cache"] = Tensor({Hkv, config_.kvCacheSize, config_.headDim}, config_.kvCacheDtype);
    }
}

inline void Attention::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& Q = (*this)["Q"];
    const Tensor& K = (*this)["K"];
    const Tensor& V = (*this)["V"];

    // The last three axes are (heads, tokens, headDim); anything ahead of them
    // is another sequence this dispatch runs alongside, the way a Linear takes
    // its leading axes as more rows.
    auto batchSize = [&](const Tensor& t, uint32_t heads) {
        const auto& shape = t.shape();
        VAI_ASSERT(shape.size() >= 3);
        VAI_ASSERT(shape[shape.size() - 3] == heads);
        VAI_ASSERT(shape.back() == config_.headDim);
        return outerSize(shape, 3);
    };

    // partialInputs runs the node on any one input; a recording without Q's
    // producer has nothing to attend from.
    VAI_ASSERT(Q.validShape());

    // Without K / V (Tkv = 0) the keys are the cache's alone.
    const bool hasKv = K.validShape();
    VAI_ASSERT(hasKv == V.validShape());
    VAI_ASSERT(hasKv || config_.kvCacheSize > 0);

    const Dtype iDtype = Q.dtype();
    VAI_ASSERT(!hasKv || (K.dtype() == iDtype && V.dtype() == iDtype));

    B = batchSize(Q, config_.numQHeads);
    VAI_ASSERT(!hasKv || (batchSize(K, Hkv) == B && batchSize(V, Hkv) == B));
    // One counter serves the whole dispatch, so cached sequences would all have
    // to sit at the same position. Nothing needs that yet.
    VAI_ASSERT(B == 1 || config_.kvCacheSize == 0);

    uint32_t Tq = Q.shape()[Q.shape().size() - 2];
    Tkv = hasKv ? K.shape()[K.shape().size() - 2] : 0u;
    VAI_ASSERT(!hasKv || V.shape()[V.shape().size() - 2] == Tkv);
    // Cacheless binds the counter null, which reads back as 0 on the device.
    uint32_t cached = config_.kvCacheSize > 0 ? *(uint32_t*)kvCacheCounter.map() : 0u;
    VAI_ASSERT(cached + Tkv <= config_.kvCacheSize || config_.kvCacheSize == 0 || config_.kvCacheRing);

    // FA path selection; tuning depends on (Tq, KV) so re-request on change.
    const uint32_t Dh = config_.headDim, nKv = cached + Tkv;

    // GQA-packed decode (= ggml ggml_vk_flash_attn): pack one KV group's groupSize
    // query heads into one tile — K/V streams once per group. The large-row
    // probe learns the device's max tile height before the real tuning.
    uint32_t groupSize = config_.numQHeads / Hkv;   // unchanged by folding: both sides scale
    gqaRatio = 0;
    PipeParamsFA t = intelFATuningParams(Dh, groupSize, Tq, qkAccDtype_, oAccDtype_, device());
    if (t.valid())
        t.iDtype = iDtype;   // packs every tile itself: no gqaRatio
    else if (Tq <= 8u && groupSize > 1u)
    {
        PipeParamsFA probe = ggmlFATuningParams(Dh, 512u, nKv, iDtype, device());
        uint32_t maxGqa = std::min(probe.valid() ? probe.wgRows : 0u, 32u);
        if (groupSize <= maxGqa) gqaRatio = groupSize;
    }

    // Tune with the packed row count (= ggml: N is promoted before tuning),
    // so the cm "scalar when rows==1" demotion sees groupSize, not Tq.
    if (!t.intel)
        t = ggmlFATuningParams(Dh, gqaRatio ? groupSize : Tq, nKv, iDtype, device());
    if (gqaRatio && (!t.valid() || t.wgRows < groupSize))   // safety: tile must hold the group
    {
        gqaRatio = 0;
        t = ggmlFATuningParams(Dh, Tq, nKv, iDtype, device());
    }

    // appendBeforeSweep (cm packed Tq==1 decode): the last split appends, then
    // sweeps the cache as its single source (no cache: off; Tq > 1 would race).
    // Its binary is the larger one (no SMALL_KV, padded S/P), so it has to fit
    // on its own; otherwise the packed decode keeps the cm kernel without it.
    if (t.valid())
        t.appendBeforeSweep = config_.kvCacheSize > 0 && Tkv == Tq && t.coopmat && gqaRatio > 1u && Tq == 1u &&
                              ggmlFACmShmemSupport(Dh, t.shmemStaging, true, device());
    useFa = t.valid();
    // A key tile that does not divide the ring would wrap it: kvRingAlign is stale.
    VAI_ASSERT(!useFa || !config_.kvCacheRing || config_.kvCacheSize % t.wgKv == 0);

    splitKCount = 1;

    if (useFa)
    {
        // FA pipeline: the tuning is baked into spec constants — re-request
        // only when it changes (a prefill/decode flip is a pool lookup, not
        // a rebuild).
        if (!pipeFA || t != pipeParamsFA)
        {
            pipeParamsFA = t;

            // clang-format off
            if (t.intel)
                pipeFA = pipelinePool().request("attention-fa-intel", {
                    eva::constant_id<0>(t.wgSize),                       // wgSize (= local_size_x)
                    eva::constant_id<1>(t.tileTokens),                   // tileTokens
                    eva::constant_id<3>(Dh),                             // Dh
                    eva::constant_id<4>(uint32_t(t.qkAccF32)),           // qkAccF32
                    eva::constant_id<10>(uint32_t(t.accF16)),            // accF16 (O)
                    eva::constant_id<12>(uint32_t(config_.kvCacheRing)), // ring cache
                    eva::constant_id<13>(iDtype),                       // typeI (Q / K / V)
                    eva::constant_id<14>(config_.kvCacheDtype),         // typeCache
                    eva::constant_id<15>(uint32_t(config_.kvCacheSize > 0)),  // kvCaching
                    eva::constant_id<SPEC_ID_TYPE_O>(oDtype)    // typeO
                }, { .requiredSubgroupSize = t.sgSize });
            else
            pipeFA = pipelinePool().request(
                t.coopmat ? "attention-fa-cm" : "attention-fa", {
                eva::constant_id<0>(t.wgSize),                       // wgSize (= local_size_x)
                eva::constant_id<1>(t.wgRows),                       // wgRows
                eva::constant_id<2>(t.wgKv),                         // wgKv
                eva::constant_id<3>(Dh),                             // Dh
                eva::constant_id<4>(t.dhSplit),                      // dhSplit
                eva::constant_id<5>(t.rowGroups),                    // rowGroups
                eva::constant_id<6>(t.sgSize),                       // sgSize
                eva::constant_id<7>(uint32_t(t.shmemStaging)),       // shmemStaging
                eva::constant_id<8>(uint32_t(t.stagingF16)),         // stagingF16 (scalar; cm has no spec 8)
                eva::constant_id<9>(t.limitShmem),                   // occupancy limiter (scalar; cm ignores)
                eva::constant_id<10>(uint32_t(t.accF16)),
                eva::constant_id<11>(uint32_t(t.appendBeforeSweep)), // cm packed Tq==1 decode; scalar ignores
                eva::constant_id<12>(uint32_t(config_.kvCacheRing)), // ring cache
                eva::constant_id<13>(iDtype),                       // typeI (Q / K / V)
                eva::constant_id<14>(config_.kvCacheDtype),         // typeCache
                eva::constant_id<15>(uint32_t(config_.kvCacheSize > 0)),  // kvCaching
                eva::constant_id<SPEC_ID_TYPE_O>(oDtype)    // typeO
            }, { .requiredSubgroupSize = t.sgSize });   // 0 = not forced (Intel flavor)
            // clang-format on

            if (!descSetFA)
                descSetFA = gVaiDestSetPool(pipeFA.descSetLayout(0));
        }

        // Split-K second pass. The count follows the grid this shape dispatches,
        // never the cache — the kernel sizes the segments from that itself.
        uint32_t totalWgs = B * (t.intel    ? CEIL_DIV(Tq, t.tileTokens) * Hkv
                                 : gqaRatio ? Tq * Hkv
                                            : CEIL_DIV(Tq, t.wgRows) * config_.numQHeads);
        splitKCount = t.intel ? intelFASplitKCount(totalWgs, nKv, config_.kvCacheSize, t, device())
                              : ggmlFASplitKCount(totalWgs, device());
        // The partials and the reduce pipeline materialize only when a split
        // actually happens; both FA kernels write the same partial layout.
        if (splitKCount > 1)
        {
            (*this)["partialO"]  = Tensor({B * Tq * config_.numQHeads, splitKCount, Dh}, Dtype::Fp32);
            (*this)["partialML"] = Tensor({B * Tq * config_.numQHeads, splitKCount, 2}, Dtype::Fp32);
            // blockSize = subgroup size (= ggml BLOCK_SIZE); sgSize can change
            // between steps (AMD wave32 fires only for nRows < 4), and a stale
            // smaller block would leave Dh columns un-reduced — re-request.
            const PipeParamsReduce r = { .blockSize = t.sgSize ? t.sgSize : 32u };   // no-subgroup flavor: plain 32-wide blocks
            if (!pipeReduce || r != pipeParamsReduce)
            {
                pipeParamsReduce = r;
                // clang-format off
                pipeReduce = pipelinePool().request("attention-fa-reduce", {
                    eva::constant_id<0>(r.blockSize),                    // blockSize
                    eva::constant_id<SPEC_ID_TYPE_O>(oDtype)     // typeO
                });
                // clang-format on
                if (!descSetReduce)
                    descSetReduce = gVaiDestSetPool(pipeReduce.descSetLayout(0));
            }
        }
    }

    // Naive fallback pipeline — lazy, built on the first forward that needs it.
    const PipeParamsNaive n = { .iDtype = iDtype };
    if (!useFa && (!pipeNaive || n != pipeParamsNaive))
    {
        pipeParamsNaive = n;
        // clang-format off
        pipeNaive = pipelinePool().request("attention", {
            eva::constant_id<0>(uint32_t(config_.kvCacheRing)),  // ring cache
            eva::constant_id<1>(iDtype),                        // typeI (Q / K / V)
            eva::constant_id<2>(config_.kvCacheDtype),          // typeCache
            eva::constant_id<3>(uint32_t(config_.kvCacheSize > 0)),   // kvCaching
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype)    // typeO
        });
        // clang-format on
        if (!descSetNaive)
            descSetNaive = gVaiDestSetPool(pipeNaive.descSetLayout(0));
    }

    // Token-major output [..., Tq, Hq, Dh]: the merge to [..., Tq, Hq*Dh] is then
    // a plain reshape (no permute pass), matching the FA kernels' store layout.
    // Leading axes come back where they went in, so a caller that had none sees
    // the [Tq, Hq, Dh] it always did.
    std::vector<uint32_t> outShape = Q.shape();
    outShape[outShape.size() - 3] = Tq;
    outShape[outShape.size() - 2] = config_.numQHeads;
    (*this)["out0"] = Tensor(outShape, oDtype);
}

inline void Attention::run(eva::CommandBuffer cmdBuff)
{
    auto O = (*this)["out0"].buffer();
    auto Q = (*this)["Q"].buffer();
    // Without K / V (the cache alone) both bind null: no key reads them.
    auto K = Tkv > 0 ? (*this)["K"].buffer() : eva::Buffer();
    auto V = Tkv > 0 ? (*this)["V"].buffer() : eva::Buffer();
    // [Hkv, Tmax, Dh], typeKV. Cacheless binds all three null: the kernels never
    // touch the cache, and the counter reads back as the 0 they need.
    const bool caching = config_.kvCacheSize > 0;
    auto K_cache = caching ? (*this)["K_cache"].buffer() : eva::Buffer();
    auto V_cache = caching ? (*this)["V_cache"].buffer() : eva::Buffer();
    auto counter = caching ? kvCacheCounter : eva::Buffer();

    // out0 is token-major [..., Tq, Hq, Dh]: the tokens sit third from the end,
    // wherever the leading axes put them.
    const auto& outShape = (*this)["out0"].shape();
    uint32_t Tq = outShape[outShape.size() - 3];

    // Both kernels share the binding order:
    // 0=O, 1=Q, 2=K, 3=V, 4=K_cache, 5=V_cache, 6=kvCacheCount; the causal
    // mask follows at 7 (naive) or 9 (FA, after its split-K partials).
    if (useFa)
    {
        struct {
            uint32_t Dh, Hq, groupSize, Tq, Tkv, Tmax, gqaRatio, kCount;
            float scale;
        } faConsts = {
            config_.headDim,
            config_.numQHeads,
            config_.numQHeads / Hkv,
            Tq,
            Tkv,
            config_.kvCacheSize,
            gqaRatio,
            splitKCount,
            1.0f / std::sqrt(float(config_.headDim)),
        };

        // Packed decode dispatches one workgroup per (token, KV head), else
        // one per (Q row tile, head); split-K multiplies x by kCount. Unused
        // partial bindings take O as a stand-in (= ggml's fully-bound trick).
        bool packed = gqaRatio > 1;
        bool split  = splitKCount > 1;
        auto PO  = split ? (*this)["partialO"].buffer()  : O;
        auto PML = split ? (*this)["partialML"].buffer() : O;

        descSetFA.write({ O, Q, K, V, K_cache, V_cache, counter, PO, PML, causalMask });
        cmdBuff.bindPipeline(pipeFA)
            .bindDescSets({ descSetFA })
            .setPushConstants(0, sizeof(faConsts), &faConsts)
            .dispatch((pipeParamsFA.intel ? CEIL_DIV(Tq, pipeParamsFA.tileTokens)
                       : packed           ? Tq
                                          : CEIL_DIV(Tq, pipeParamsFA.wgRows)) *
                          splitKCount,
                      (pipeParamsFA.intel || packed) ? Hkv : config_.numQHeads,
                      B);

        if (split)
        {
            struct { uint32_t Dh, kCount; } reduceConsts = { config_.headDim, splitKCount };

            descSetReduce.write({ O, PO, PML });
            cmdBuff
                .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / PO / eva::SYNC_SCOPE::COMPUTE_READ)
                .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / PML / eva::SYNC_SCOPE::COMPUTE_READ)
                .bindPipeline(pipeReduce)
                .bindDescSets({descSetReduce})
                .setPushConstants(0, sizeof(reduceConsts), &reduceConsts)
                // The head dim is split across workgroups in y.
                .dispatch(B * Tq * config_.numQHeads,
                          CEIL_DIV(config_.headDim, pipeParamsReduce.blockSize), 1);
        }
        cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / O / eva::SYNC_SCOPE::COMPUTE_READ);
    }
    else
    {
        struct {
            uint32_t Dh, Hq, groupSize, Tq, Tkv, Tmax;
            float scale;
        } constants = {
            config_.headDim,
            config_.numQHeads,
            config_.numQHeads / Hkv,
            Tq,
            Tkv,
            config_.kvCacheSize,
            1.0f / std::sqrt(float(config_.headDim)),
        };

        descSetNaive.write({ O, Q, K, V, K_cache, V_cache, counter, causalMask });
        cmdBuff
            .bindPipeline(pipeNaive)
            .bindDescSets({descSetNaive})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch2(config_.headDim, Tq, B * config_.numQHeads)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / O / eva::SYNC_SCOPE::COMPUTE_READ);
    }
}


}   // namespace vai::node

#endif   // VAI_NODES_ATTENTION_H
