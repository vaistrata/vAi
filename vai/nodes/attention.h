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
    devices degrade instead of crashing. See supportsFlashAttention().

  Intel's coopmat tiles (Xe2 / Xe3) run attention-fa-cm-intel, the kernel
  Intel was verified with; attention-fa-cm moved on for the other vendors.

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
    // ggml Br/Bc/d_split/row_split), plus the kernel, what a workgroup covers,
    // and the dtypes the pipeline is specialized on.
    struct PipeParamsFA {
        // Tile geometry
        uint32_t   wgSize       = 0;
        uint32_t sgSize = 0;       // forced subgroup size (0 = don't force: the Intel scalar flavor)
        uint32_t wgRows = 0;       // Q rows per tile
        uint32_t wgKv = 0;         // KV columns per tile step
        uint32_t dhSplit = 0;      // scalar: adjacent lanes splitting the head dim
        uint32_t rowGroups = 0;    // scalar: row-groups per workgroup (1 or 4)
        uint32_t tileTokens = 0;   // tokens one workgroup covers ...
        uint32_t tileHeads = 0;    // ... x query heads it covers (> 1: one KV group's, packed)
        // Kernel
        bool coopmat = false;   // the coopmat kernel instead of the scalar one (on Intel, attention-fa-cm-intel)
        bool intel = false;     // attention-fa-intel (packs its KV group itself)
        // Precision
        bool stagingF16 = false;   // scalar: f16 K / V staging (= ggml FLOAT_TYPE)
        bool accF16 = false;       // f16 O accumulator (intel: setOAccDtype)
        bool qkAccF32 = false;     // intel: Q K^T summed in fp32, else fp16 within 32 dims (setQkAccDtype)
        // Per-kernel flags
        bool shmemStaging = false;        // stage K / V tiles through shared memory
        uint32_t limitShmem = 0;          // scalar (AMD): dead shared (vec4s) to cap occupancy
        bool appendBeforeSweep = false;   // cm packed Tq==1 decode: append, then sweep the cache alone (spec 11)
        // Inputs
        vai::Dtype iDtype = vai::Dtype::None;   // Q / K / V inputs

        bool valid() const { return wgSize != 0; }
        bool operator==(const PipeParamsFA&) const = default;
    };

    struct PipeParamsReduce {
        uint32_t blockSize = 0;

        bool operator==(const PipeParamsReduce&) const = default;
    };

    struct PipeParamsNaive {
        vai::Dtype iDtype = vai::Dtype::None;   // Q / K / V inputs

        bool operator==(const PipeParamsNaive&) const = default;
    };

    struct Config {
        uint32_t headDim;
        uint32_t numQHeads;
        uint32_t numKVHeads = 0;                      // for GQA (Hkv < numQHeads); 0 → Hkv = numQHeads
        uint32_t kvCacheSize = 0;                     // tokens the cache holds; 0 = no cache at all
        vai::Dtype kvCacheDtype = vai::Dtype::None;   // K / V cache; required when kvCacheSize > 0
        bool kvCacheRing = false;   // the cache keeps the last kvCacheSize positions, rounded up to kvRingAlign
    } config_;

    // Every path's key tile (fa / cm wgKv, Intel BK) divides it; prepare()
    // checks the tile it picks.
    static constexpr uint32_t kvRingAlign = 256;

    Attention(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
    // Read at dispatch time: rewriting the mask needs no re-recording.
    void setCausalMask(eva::Buffer buffer)
    {
        causalMask = buffer;
    }

    void setQkAccDtype(vai::Dtype dtype, vai::DtypeKey) { qkAccDtype_ = dtype; }

    void setOAccDtype(vai::Dtype dtype, vai::DtypeKey) { oAccDtype_ = dtype; }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        VAI_ASSERT(config_.kvCacheSize > 0);
        kvCacheCounter = buffer;
    }

    // Test / bench only: which path prepare() selected.
    bool usingFa() const { return useFa; }
    bool usingCm() const { return useFa && pipeParamsFA.coopmat; }
    bool usingIntel() const { return useFa && pipeParamsFA.intel; }
    const PipeParamsFA& getPipeParamsFA() const { return pipeParamsFA; }
    uint32_t getSplitKCount() const { return splitKCount; }
    uint32_t getGqaRatio() const { return gqaRatioOf(pipeParamsFA); }

    // The ggml kernels' packed-decode push constant: the heads a tile packs, 0
    // when unpacked (the Intel kernel packs on its own and ignores it).
    static uint32_t gqaRatioOf(const PipeParamsFA& t) { return !t.intel && t.tileHeads > 1u ? t.tileHeads : 0u; }

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
    uint32_t splitKCount = 1;   // KV splits = the grid multiplier
};

/////////////////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////////////////

// The tile's shared memory in bytes: one line per shader shared array, in shader order.
inline size_t tileSharedMemoryBytes(const Attention::PipeParamsFA& t, uint32_t Dh)
{
    constexpr size_t vec4 = 16, f16vec4 = 8;   // bytes
    size_t bytes = 0;

    if (t.coopmat)
    {
        const size_t dh4Pad = ((Dh + 15) & ~15u) / 4;                                       // DH4_PAD
        const size_t kvParts = (t.shmemStaging && Dh == 256) ? 2 : 1;                       // KV_PARTS
        const size_t dh4Part = dh4Pad / kvParts;                                            // DH4_PART
        const size_t spPad = (t.appendBeforeSweep && t.shmemStaging) ? 2 : 0;               // SP_PAD
        const bool smallKv = !t.shmemStaging && dh4Pad == Dh / 4 && !t.appendBeforeSweep;   // SMALL_KV
        const size_t numSg = t.wgSize / t.sgSize;                                           // NUM_SG
        bytes += t.wgRows * (dh4Pad + 2) * f16vec4;                                         // s_q  [wgRows][Q_STRIDE]
        bytes += t.wgKv * (smallKv ? numSg * 4 : dh4Part + 2) * f16vec4;   // s_kv [wgKv][VS_STRIDE | KV_STRIDE]
        bytes += std::max(t.wgKv * (t.wgRows / 4 + spPad), t.wgRows * (dh4Part + 2)) *
                 vec4;                                        // s_s  [SS_SIZE]: S^T, then the f32 P.V drain
        bytes += t.wgKv * (t.wgRows / 4 + spPad) * f16vec4;   // s_p  [wgKv][P_STRIDE]
    }
    else
    {
        const size_t stride = Dh / 4 + 1;                          // SHMEM_STRIDE
        const size_t numSg = t.sgSize ? t.wgSize / t.sgSize : 1;   // NUM_SG
        const size_t redEntries =
            t.sgSize == 0 ? t.wgSize : (t.rowGroups == 1 ? numSg * t.dhSplit : numSg);   // RED_ENTRIES
        const size_t element = t.stagingF16 ? f16vec4 : vec4;
        bytes += t.wgRows * stride * element;   // s_q16 | s_q32
        if (t.shmemStaging)
            bytes += t.wgKv * stride * element;                                // s_kv16 | s_kv32
        bytes += redEntries * (sizeof(float) + (t.accF16 ? f16vec4 : vec4));   // s_red + s_redv4h | s_redv4
        bytes += t.limitShmem * vec4;                                          // s_occupancy
    }
    return bytes;
}

// The same for Intel's coopmat kernel (attention-fa-cm-intel), the fixed 16x64 tile: the budget it was
// verified with.
inline size_t intelCoopmatTileSharedMemoryBytes(const Attention::PipeParamsFA& t, uint32_t Dh)
{
    constexpr size_t vec4 = 16, f16vec4 = 8;   // bytes
    const size_t dh4Pad = ((Dh + 15) & ~15u) / 4;
    const bool smallKv = !t.shmemStaging && dh4Pad == Dh / 4 && !t.appendBeforeSweep;
    const size_t parts = (t.shmemStaging && Dh == 256) ? 2 : 1;   // KV_PARTS
    const size_t spPad = (t.appendBeforeSweep && t.shmemStaging) ? 2 : 0;
    size_t bytes = 16 * (dh4Pad + 2) * f16vec4;                                // s_q
    bytes += 64 * (smallKv ? 16 : dh4Pad / parts + 2) * f16vec4;               // s_kv
    bytes += 64 * (16 / 4 + spPad) * vec4 + 64 * (16 / 4 + spPad) * f16vec4;   // s_s / s_p
    bytes += 16 * (dh4Pad + 2) * f16vec4;                                      // s_pv
    return bytes;
}

// Whether the tile fits the device's shared memory, laid out as the kernel the device runs.
inline bool tileFitsSharedMemory(const Attention::PipeParamsFA& t, uint32_t Dh, eva::Device device)
{
    const bool intelCoopmat = device.vendorID() == eva::VENDOR_ID::INTEL && t.coopmat;
    const size_t bytes = intelCoopmat ? intelCoopmatTileSharedMemoryBytes(t, Dh) : tileSharedMemoryBytes(t, Dh);
    return bytes <= device.maxComputeSharedMemorySize();
}

// 16x16x16 f16 x f16 at subgroup scope with both accumulators: f32 for S,
// f16 for P.V (= upstream coopmat_support_16x16x16_f16acc).
inline bool supportsCoopmat16Shapes(eva::Device device)
{
    if (!device.supportsCooperativeMatrix())
        return false;
    bool f32Acc = false, f16Acc = false;
    for (const auto& p : device.cooperativeMatrixProperties())
        if (p.M == 16 && p.N == 16 && p.K == 16 &&
            p.aType == eva::COMPONENT_TYPE::FLOAT16 && p.bType == eva::COMPONENT_TYPE::FLOAT16 &&
            p.scope == eva::SCOPE::SUBGROUP)
        {
            if (p.cType == eva::COMPONENT_TYPE::FLOAT32 && p.resultType == eva::COMPONENT_TYPE::FLOAT32)
                f32Acc = true;
            if (p.cType == eva::COMPONENT_TYPE::FLOAT16 && p.resultType == eva::COMPONENT_TYPE::FLOAT16)
                f16Acc = true;
        }
    return f32Acc && f16Acc;
}

// = ggml d_split: at most the subgroup size, 8, and the largest power of two dividing Dh/4.
inline uint32_t scalarDhSplit(uint32_t Dh, uint32_t sg)
{
    uint32_t lsb = Dh & (~Dh + 1u);
    return std::min(std::min(sg, 8u), lsb / 4u);
}


/////////////////////////////////////////////////////////////////////////////////////////
// FA tile selection shared by the vendors: the coopmat and scalar tiles, the split count
/////////////////////////////////////////////////////////////////////////////////////////

// The coopmat kernel's fixed 16x64 tile (= ggml get_fa_tuning_params_coopmat1
// minus the vendor lines).
inline Attention::PipeParamsFA coopmatFlashAttentionTile(uint32_t Dh, uint32_t nRows, bool staging, eva::Device device)
{
    if (nRows <= 1u)
        return {};   // scalar wins single rows

    if (!supportsCoopmat16Shapes(device))
        return {};

    uint32_t sg = device.subgroupSize();
    if (sg < 16u || (sg & (sg - 1u)) != 0u || 64u % sg != 0u)
        return {};

    Attention::PipeParamsFA t = { .wgSize = 4u * sg,
                                  .sgSize = sg,
                                  .wgRows = 16u,
                                  .wgKv = 64u,
                                  .dhSplit = 1u,
                                  .rowGroups = 4u,
                                  .coopmat = true,
                                  .shmemStaging = staging };
    t.accF16 = false;   // O in f32 (llama.cpp cm1: f16)

    // Budgeted without appendBeforeSweep; prepare() adds that where it fits.
    if (!tileFitsSharedMemory(t, Dh, device))
        return {};

    return t;
}

// = ggml get_fa_tuning_params_scalar minus the vendor lines: the scalar tile's
// geometry from the vendor's subgroup size, row-split bound and no-subgroup flag.
inline Attention::PipeParamsFA scalarFlashAttentionDefaultTile(uint32_t Dh, uint32_t nRows, uint32_t nKv, uint32_t sg,
                                                               uint32_t rowSplitMaxDh, bool noSubgroups)
{
    // rowGroups: only for large batch + large head size.
    uint32_t rowGroups = (nRows < 4u || Dh <= rowSplitMaxDh) ? 1u : 4u;

    // 2 or 4 subgroups per workgroup.
    uint32_t wgSize = (sg > 32u && (nRows < 4u || Dh < (rowGroups == 1u ? 128u : 64u)))
                    ? sg * 2u : sg * 4u;

    // wgRows/wgKv; Dh&8 = head size not a multiple of 16.
    bool reduceRows = ((Dh & 8u) != 0u) || nKv < 1024u || noSubgroups;
    uint32_t wgRows, wgKv;
    if (nRows == 1u)
    {
        wgRows = 1u;
        wgKv = 64u;
    }
    else
    {
        if (rowGroups == 1u) wgRows = (nRows == 2u) ? 2u : ((nRows <= 4u || reduceRows) ? 4u : 8u);
        else                 wgRows = (nRows <= 4u) ? 4u : ((nRows <= 8u || reduceRows) ? 8u : 16u);
        wgKv = ((Dh & 8u) != 0u) ? 64u : 32u;
    }

    Attention::PipeParamsFA t = { .wgSize = wgSize,
                                  .sgSize = noSubgroups ? 0u : sg,
                                  .wgRows = wgRows,
                                  .wgKv = wgKv,
                                  .dhSplit = scalarDhSplit(Dh, sg),
                                  .rowGroups = rowGroups };
    return t;
}

// The scalar kernel's tile from the vendor's default tile and flags: the
// kernel's structural checks, then the shared-memory gate (halve wgRows once,
// then give up).
inline Attention::PipeParamsFA scalarFlashAttentionTile(Attention::PipeParamsFA t, uint32_t Dh, eva::Device device)
{
    if (t.sgSize != 0u && t.rowGroups > 1u && t.wgSize != t.rowGroups * t.sgSize) return {};
    if (t.wgRows % t.rowGroups != 0u) return {};
    uint32_t colsPerStep = t.wgSize / t.dhSplit / t.rowGroups;
    if (colsPerStep == 0u || t.wgKv % colsPerStep != 0u) return {};
    if ((Dh / 4u) % t.dhSplit != 0u)
        return {};

    if (!tileFitsSharedMemory(t, Dh, device))
    {
        if (t.wgRows < 2u || (t.wgRows / 2u) % t.rowGroups != 0u) return {};
        t.wgRows /= 2u;
        if (!tileFitsSharedMemory(t, Dh, device))
            return {};
    }
    return t;
}

// = the split_k block of ggml_vk_flash_attn: ~2 workgroups per shader core.
// ggml sizes the splits from the KV length; vai records once, so the count
// comes from occupancy and the kernel cuts the keys it reaches by it.
inline uint32_t occupancyFlashAttentionSplitK(uint32_t totalWgs, uint32_t cores)
{
    if (cores == 0)
        cores = 16;   // ggml's placeholder when unreported
    if (totalWgs == 0 || totalWgs >= cores * 2)
        return 1;
    return std::max(1u, cores * 2 / totalWgs);
}


/////////////////////////////////////////////////////////////////////////////////////////
// NVIDIA
/////////////////////////////////////////////////////////////////////////////////////////

inline Attention::PipeParamsFA nvFlashAttentionTile(uint32_t Dh, uint32_t nRows, uint32_t nKv, eva::Device device)
{
    // Turing: scalar, as upstream (the cm1 FA shader miscompiles there).
    if (device.architectureID() != eva::Architecture::NVIDIA_TURING)
    {
        Attention::PipeParamsFA t = coopmatFlashAttentionTile(Dh, nRows, /*staging*/ true, device);
        if (t.valid())
            return t;
    }

    const uint32_t sg = device.subgroupSize();
    Attention::PipeParamsFA t = scalarFlashAttentionDefaultTile(Dh, nRows, nKv, sg, 64u, false);
    // = ggml (every Dh < 256) minus Dh < 128: unstaged full-tile loads measured faster at Dh = 64.
    t.shmemStaging = Dh >= 128u && Dh < 256u;
    // f16 staging needs shaderFloat16 (= ggml device->fp16), which eva does not
    // expose: assumed on NVIDIA. TODO (eva): the feature query.
    t.stagingF16 = true;
    t.accF16 = false;   // O in f32: more accurate, and measured faster than ggml's f16 here

    return scalarFlashAttentionTile(t, Dh, device);
}


/////////////////////////////////////////////////////////////////////////////////////////
// AMD
/////////////////////////////////////////////////////////////////////////////////////////

inline Attention::PipeParamsFA amdFlashAttentionTile(uint32_t Dh, uint32_t nRows, uint32_t nKv, eva::Device device)
{
    const eva::Architecture arch = device.architectureID();
    const bool rdna = arch != eva::Architecture::AMD_GCN;   // = upstream: undetected parts count as RDNA

    // cm (= ggml_vk_khr_cooperative_matrix_support): the proprietary and AMDVLK
    // drivers over-report the extension, so only RDNA3+ there; RADV as reported.
    const bool allowCm =
        (device.driverID() == eva::DRIVER_ID::AMD_PROPRIETARY || device.driverID() == eva::DRIVER_ID::AMD_OPEN_SOURCE)
            ? arch == eva::Architecture::AMD_POST_RDNA2
            : true;
    if (allowCm)
    {
        Attention::PipeParamsFA t = coopmatFlashAttentionTile(Dh, nRows, /*staging*/ false, device);
        if (t.valid())
            return t;
    }

    // scalar (= ggml get_fa_tuning_params_scalar, AMD branches): wave32 for
    // small batches on RDNA, only where the device can pin it.
    uint32_t sg = device.subgroupSize();
    if (rdna && nRows < 4u && device.minSubgroupSize() <= 32u)
        sg = 32u;

    // = upstream row_split_max_hsk: non-UMA RDNA raises the bound to 128 past 8 rows.
    uint32_t rowSplitMaxDh = 64u;
    if (rdna && device.deviceType() != eva::DEVICE_TYPE::INTEGRATED_GPU)
        rowSplitMaxDh = (nRows <= 8u) ? 64u : 128u;

    Attention::PipeParamsFA t = scalarFlashAttentionDefaultTile(Dh, nRows, nKv, sg, rowSplitMaxDh, false);
    t.shmemStaging = false;   // = ggml: staging is NVIDIA-only

    // Occupancy limiter (= ggml's RDNA2 / GCN values). ggml applies the RDNA
    // branch to all of RDNA; vai keeps RDNA3+ out: measured harmful there.
    if (device.maxComputeSharedMemorySize() == 65536u)
    {
        if (rdna && arch != eva::Architecture::AMD_POST_RDNA2 && nRows >= 64u && Dh <= 128u)
            t.limitShmem = (Dh <= 64u ? 26u : 30u) * 1024u / 4u / 4u;
        else if (!rdna && nRows <= 8u && Dh >= 256u)
            t.limitShmem = (nRows < 4u ? 14u : 26u) * 1024u / 4u / 4u;
    }
    t.stagingF16 = true;   // shaderFloat16 assumed, as on NVIDIA
    // O in f32 except where ggml's f16 measured faster (RDNA3+ small-batch,
    // big-head decode); the kernel's f16 exchange is uvec2-packed, so no
    // subgroup-extended-types gate.
    t.accF16 = arch == eva::Architecture::AMD_POST_RDNA2 && nRows <= 8u && Dh >= 128u;

    return scalarFlashAttentionTile(t, Dh, device);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Intel
/////////////////////////////////////////////////////////////////////////////////////////

// Intel kernel (attention-fa-intel) for Intel parts the coopmat path leaves
// out: a 16-row tile of tileTokens tokens x the KV group's heads, subgroups of
// 16 taking 8 keys each, at most 22 KB of shared memory. The KV group has to
// fit the tile and its rows to fill it (Tq x groupSize >= 16). Otherwise the
// ggml kernels: cm on Xe2 / Xe3, else the scalar tile.
inline Attention::PipeParamsFA intelFlashAttentionTile(uint32_t Dh, uint32_t nRows, uint32_t nKv, uint32_t groupSize,
                                                       uint32_t Tq, eva::Device device)
{
    const bool xe2Plus = device.architectureID() == eva::Architecture::INTEL_XE2 ||
                         device.architectureID() == eva::Architecture::INTEL_XE3;

    if (!xe2Plus && device.supportsSubgroupExtendedTypes() && groupSize <= 16u && Tq * groupSize >= 16u)
    {
        const uint32_t wgSize = Dh <= 64u ? 128u : 256u;
        Attention::PipeParamsFA t = {
            .wgSize = wgSize, .sgSize = 16u, .wgRows = 16u, .wgKv = wgSize / 2u, .dhSplit = 1u, .rowGroups = 1u
        };
        t.intel = true;
        t.tileTokens = 16u / groupSize;   // accumulators: setQkAccDtype / setOAccDtype, applied by prepare()
        t.tileHeads = groupSize;
        return t;
    }

    // Vendor screen for the cm kernel (ggml_vk_khr_cooperative_matrix_support
    // port): Intel runs it on Xe2 / Xe3 only. No K/V staging.
    if (xe2Plus)
    {
        Attention::PipeParamsFA t = coopmatFlashAttentionTile(Dh, nRows, /*staging*/ false, device);
        if (t.valid())
            return t;
    }

    // Intel: don't force a subgroup size (upstream disable_subgroups); sg=32
    // feeds the wgSize/dhSplit formulas, the pipeline gets sgSize = 0 and the
    // kernel's shared-tree epilogue flavor.
    Attention::PipeParamsFA t = scalarFlashAttentionDefaultTile(Dh, nRows, nKv, /*sg*/ 32u, 64u, /*noSubgroups*/ true);
    t.shmemStaging = false;   // = ggml: staging is NVIDIA-only
    // f16 staging needs the shaderFloat16 feature (= ggml device->fp16); eva
    // doesn't expose it, so Intel is taken as having it.
    // TODO (eva): shaderFloat16 + storageBuffer16BitAccess feature queries
    //             (= ggml fp16_compute / fp16_storage)
    t.stagingF16 = true;
    t.accF16 = false;   // O accumulates in fp32

    return scalarFlashAttentionTile(t, Dh, device);
}

// Split-K for the Intel kernel: the keys split into enough spans to fill every
// Xe core, which holds 128 hardware threads (16 EUs x 8) = 128 / (wgSize / 16)
// workgroups, but no more spans than the keys a recording can reach have
// blocks: the cache's capacity, or with no cache this dispatch's own keys. The
// ggml kernels take the occupancy count, with pre-Xe2 cores counted double
// (ggml's Alchemist multiplier; not applied to the placeholder).
inline uint32_t intelFlashAttentionSplitK(uint32_t totalWgs, uint32_t nKv, uint32_t kvCacheSize,
                                          const Attention::PipeParamsFA& t, eva::Device device)
{
    if (t.intel)
    {
        const uint32_t target = device.computeBlockCount() * 128u / (t.wgSize / 16u);
        if (nKv == 0 || totalWgs == 0 || totalWgs >= target)
            return 1;
        const uint32_t keys = kvCacheSize > 0 ? kvCacheSize : nKv;
        return std::min(CEIL_DIV(target, totalWgs), CEIL_DIV(keys, t.wgKv));
    }
    uint32_t cores = device.computeBlockCount();
    if (cores != 0 && device.architectureID() != eva::Architecture::INTEL_XE2 &&
        device.architectureID() != eva::Architecture::INTEL_XE3)
        cores *= 2;
    return occupancyFlashAttentionSplitK(totalWgs, cores);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Other vendors (Qualcomm, software, unknown): ggml's default branches
/////////////////////////////////////////////////////////////////////////////////////////

// cm as reported, then ggml's default scalar branches with fp32 staging.
inline Attention::PipeParamsFA otherFlashAttentionTile(uint32_t Dh, uint32_t nRows, uint32_t nKv, eva::Device device)
{
    Attention::PipeParamsFA t = coopmatFlashAttentionTile(Dh, nRows, /*staging*/ false, device);
    if (t.valid())
        return t;

    const uint32_t sg = device.subgroupSize();
    t = scalarFlashAttentionDefaultTile(Dh, nRows, nKv, sg, 64u, false);
    t.shmemStaging = false;
    t.stagingF16 = false;   // shaderFloat16 unknown: fp32 staging
    t.accF16 = false;

    return scalarFlashAttentionTile(t, Dh, device);
}


/////////////////////////////////////////////////////////////////////////////////////////
// FA kernel, tile and split: the vendor decides
/////////////////////////////////////////////////////////////////////////////////////////

inline Attention::PipeParamsFA selectFlashAttentionTile(uint32_t Dh, uint32_t nRows, uint32_t nKv, uint32_t groupSize,
                                                        uint32_t Tq, eva::Device device)
{
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::NVIDIA:
        return nvFlashAttentionTile(Dh, nRows, nKv, device);
    case eva::VENDOR_ID::AMD:
        return amdFlashAttentionTile(Dh, nRows, nKv, device);
    case eva::VENDOR_ID::INTEL:
        return intelFlashAttentionTile(Dh, nRows, nKv, groupSize, Tq, device);
    default:
        return otherFlashAttentionTile(Dh, nRows, nKv, device);
    }
}

inline uint32_t selectFlashAttentionSplitK(uint32_t totalWgs, uint32_t nKv, uint32_t kvCacheSize,
                                           const Attention::PipeParamsFA& t, eva::Device device)
{
    // One rule (ggml's occupancy count) for every vendor but Intel.
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::INTEL:
        return intelFlashAttentionSplitK(totalWgs, nKv, kvCacheSize, t, device);
    default:
        return occupancyFlashAttentionSplitK(totalWgs, device.computeBlockCount());
    }
}


/////////////////////////////////////////////////////////////////////////////////////////
// FA or naive: the first decision
/////////////////////////////////////////////////////////////////////////////////////////

inline bool supportsFlashAttention(uint32_t Dh, eva::Device device)
{
    // Dh the FA kernels address: multiples of 8 (= ggml), 16..256.
    if (Dh % 8u != 0u || Dh < 16u || Dh > 256u)
        return false;

    // The subgroup-forced flavors need a sane subgroup size.
    const uint32_t sg = device.subgroupSize();
    if (sg < 8u || (sg & (sg - 1u)) != 0u)
        return false;

    // The smallest scalar tile (one row, no staging) must fit shared memory.
    const Attention::PipeParamsFA minTile = {
        .wgSize = 64u, .sgSize = 0u, .wgRows = 1u, .wgKv = 64u, .dhSplit = 1u, .rowGroups = 1u
    };

    return tileFitsSharedMemory(minTile, Dh, device);
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

    // Cacheless leaves the slots out (an unassigned persistent tensor fails the
    // upload sweep); run() binds them null.
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

    // Axes ahead of (heads, tokens, headDim) are more sequences, as a Linear's
    // leading axes are more rows.
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
    // Intel keeps the tuning it was verified with, and its coopmat tiles the kernel too (attention-fa-cm-intel).
    const bool intelKernels = device().vendorID() == eva::VENDOR_ID::INTEL;
    VAI_ASSERT(!hasKv || (K.dtype() == iDtype && V.dtype() == iDtype));

    B = batchSize(Q, config_.numQHeads);
    VAI_ASSERT(!hasKv || (batchSize(K, Hkv) == B && batchSize(V, Hkv) == B));
    // One counter per dispatch: cached sequences cannot sit at different positions.
    VAI_ASSERT(B == 1 || config_.kvCacheSize == 0);

    uint32_t Tq = Q.shape()[Q.shape().size() - 2];
    Tkv = hasKv ? K.shape()[K.shape().size() - 2] : 0u;
    VAI_ASSERT(!hasKv || V.shape()[V.shape().size() - 2] == Tkv);
    uint32_t cached = config_.kvCacheSize > 0 ? *(uint32_t*)kvCacheCounter.map() : 0u;
    VAI_ASSERT(cached + Tkv <= config_.kvCacheSize || config_.kvCacheSize == 0 || config_.kvCacheRing);

    // FA or naive first; on the FA side the GQA packing, then the vendor's tile.
    // Tuning depends on (Tq, KV): the pipeline is re-requested on change.
    const uint32_t Dh = config_.headDim, nKv = cached + Tkv;
    const uint32_t groupSize = config_.numQHeads / Hkv;   // unchanged by folding: both sides scale
    useFa = supportsFlashAttention(Dh, device());

    if (useFa)
    {
        auto tune = [&](uint32_t nRows) { return selectFlashAttentionTile(Dh, nRows, nKv, groupSize, Tq, device()); };

        // GQA-packed decode (= ggml ggml_vk_flash_attn): one KV group's heads share a
        // tile when the vendor's largest tile (the 512-row probe) holds them.
        uint32_t packed = (Tq <= 8u && groupSize > 1u && groupSize <= tune(512u).wgRows) ? groupSize : 0u;

        PipeParamsFA t = tune(packed ? groupSize : Tq);    // = ggml: N is promoted before tuning
        if (packed && (t.intel || t.wgRows < groupSize))   // Intel packs itself; a halved tile cannot hold the group
        {
            packed = 0;
            t = tune(Tq);
        }
        VAI_ASSERT(t.valid() && "Attention: supportsFlashAttention() passed but the vendor tuner returned no tile");
        // A key tile that does not divide the ring would wrap it: kvRingAlign is stale.
        VAI_ASSERT(!config_.kvCacheRing || config_.kvCacheSize % t.wgKv == 0);

        // What one workgroup covers: the ggml kernels take wgRows tokens of one
        // head, or one token of a KV group's heads when packed; the Intel kernel
        // set its own, and its accumulators are the node's setting.
        if (!t.intel)
        {
            t.tileTokens = packed ? 1u : t.wgRows;
            t.tileHeads = packed ? groupSize : 1u;
        }
        else
        {
            t.qkAccF32 = qkAccDtype_ == Dtype::Fp32;
            t.accF16 = oAccDtype_ == Dtype::Fp16;
        }
        t.iDtype = iDtype;

        // cm packed Tq==1 decode: the last split appends, then sweeps the cache alone
        // (Tq > 1 would race). Its binary is larger (no SMALL_KV, padded S/P): it must fit by itself.
        t.appendBeforeSweep = config_.kvCacheSize > 0 && Tkv == Tq && t.coopmat && t.tileHeads > 1u && Tq == 1u;
        if (t.appendBeforeSweep)
            t.appendBeforeSweep = tileFitsSharedMemory(t, Dh, device());

        // Spec constants carry the tuning: re-request only when it changes.
        if (!pipeFA || t != pipeParamsFA)
        {
            pipeParamsFA = t;

            if (t.intel)
                pipeFA =
                    pipelinePool().request("attention-fa-intel",
                                           { eva::constant_id<0>(t.wgSize),
                                             eva::constant_id<1>(t.tileTokens),
                                             eva::constant_id<3>(Dh),
                                             eva::constant_id<4>(uint32_t(t.qkAccF32)),
                                             eva::constant_id<10>(uint32_t(t.accF16)),
                                             eva::constant_id<12>(uint32_t(config_.kvCacheRing)),   // ring cache
                                             eva::constant_id<13>(iDtype),                          // typeI (Q / K / V)
                                             eva::constant_id<14>(config_.kvCacheDtype),            // typeCache
                                             eva::constant_id<15>(uint32_t(config_.kvCacheSize > 0)),   // kvCaching
                                             eva::constant_id<SPEC_ID_TYPE_O>(oDtype) },
                                           { .requiredSubgroupSize = t.sgSize });
            else if (intelKernels && t.coopmat)   // Intel's coopmat kernel, as verified there
                pipeFA =
                    pipelinePool().request("attention-fa-cm-intel",
                                           { eva::constant_id<0>(t.wgSize),
                                             eva::constant_id<1>(t.wgRows),
                                             eva::constant_id<2>(t.wgKv),
                                             eva::constant_id<3>(Dh),
                                             eva::constant_id<6>(t.sgSize),
                                             eva::constant_id<7>(uint32_t(t.shmemStaging)),
                                             eva::constant_id<10>(uint32_t(t.accF16)),
                                             eva::constant_id<11>(uint32_t(t.appendBeforeSweep)),
                                             eva::constant_id<12>(uint32_t(config_.kvCacheRing)),   // ring cache
                                             eva::constant_id<13>(iDtype),                          // typeI (Q / K / V)
                                             eva::constant_id<14>(config_.kvCacheDtype),            // typeCache
                                             eva::constant_id<15>(uint32_t(config_.kvCacheSize > 0)),   // kvCaching
                                             eva::constant_id<SPEC_ID_TYPE_O>(oDtype) },
                                           { .requiredSubgroupSize = t.sgSize });
            else
                pipeFA =
                    pipelinePool().request(t.coopmat ? "attention-fa-cm" : "attention-fa",
                                           { eva::constant_id<0>(t.wgSize),
                                             eva::constant_id<1>(t.wgRows),
                                             eva::constant_id<2>(t.wgKv),
                                             eva::constant_id<3>(Dh),
                                             eva::constant_id<4>(t.dhSplit),
                                             eva::constant_id<5>(t.rowGroups),
                                             eva::constant_id<6>(t.sgSize),
                                             eva::constant_id<7>(uint32_t(t.shmemStaging)),
                                             eva::constant_id<8>(uint32_t(t.stagingF16)),   // scalar only
                                             eva::constant_id<9>(t.limitShmem),             // scalar only
                                             eva::constant_id<10>(uint32_t(t.accF16)),
                                             eva::constant_id<11>(uint32_t(t.appendBeforeSweep)),   // cm only
                                             eva::constant_id<12>(uint32_t(config_.kvCacheRing)),   // ring cache
                                             eva::constant_id<13>(iDtype),                          // typeI (Q / K / V)
                                             eva::constant_id<14>(config_.kvCacheDtype),            // typeCache
                                             eva::constant_id<15>(uint32_t(config_.kvCacheSize > 0)),   // kvCaching
                                             eva::constant_id<SPEC_ID_TYPE_O>(oDtype) },
                                           { .requiredSubgroupSize = t.sgSize });   // 0 = not forced

            if (!descSetFA)
                descSetFA = gVaiDestSetPool(pipeFA.descSetLayout(0));
        }

        // Split-K: the count follows the grid, never the cache (the kernel sizes the segments).
        uint32_t totalWgs = B * CEIL_DIV(Tq, t.tileTokens) * (config_.numQHeads / t.tileHeads);
        splitKCount = selectFlashAttentionSplitK(totalWgs, nKv, config_.kvCacheSize, t, device());
        if (splitKCount > 1)
        {
            (*this)["partialO"]  = Tensor({B * Tq * config_.numQHeads, splitKCount, Dh}, Dtype::Fp32);
            (*this)["partialML"] = Tensor({B * Tq * config_.numQHeads, splitKCount, 2}, Dtype::Fp32);
            // blockSize = subgroup size (= ggml BLOCK_SIZE), re-requested when it changes
            // (AMD wave32 for nRows < 4): a stale smaller block leaves Dh columns unreduced.
            const PipeParamsReduce r = { .blockSize = t.sgSize ? t.sgSize : 32u };   // no-subgroup flavor: 32
            if (!pipeReduce || r != pipeParamsReduce)
            {
                pipeParamsReduce = r;
                pipeReduce = pipelinePool().request(
                    "attention-fa-reduce",
                    { eva::constant_id<0>(r.blockSize), eva::constant_id<SPEC_ID_TYPE_O>(oDtype) });
                if (!descSetReduce)
                    descSetReduce = gVaiDestSetPool(pipeReduce.descSetLayout(0));
            }
        }
    }

    const PipeParamsNaive n = { .iDtype = iDtype };
    if (!useFa && (!pipeNaive || n != pipeParamsNaive))
    {
        pipeParamsNaive = n;
        pipeNaive = pipelinePool().request("attention",
                                           { eva::constant_id<0>(uint32_t(config_.kvCacheRing)),   // ring cache
                                             eva::constant_id<1>(iDtype),                          // typeI (Q / K / V)
                                             eva::constant_id<2>(config_.kvCacheDtype),            // typeCache
                                             eva::constant_id<3>(uint32_t(config_.kvCacheSize > 0)),   // kvCaching
                                             eva::constant_id<SPEC_ID_TYPE_O>(oDtype) });

        if (!descSetNaive)
            descSetNaive = gVaiDestSetPool(pipeNaive.descSetLayout(0));
    }

    // Token-major [..., Tq, Hq, Dh] (the FA kernels' store layout): the merge to
    // [..., Tq, Hq*Dh] is a reshape. Leading axes come back as they went in.
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
    // Cacheless binds the cache and the counter null: the kernels read 0 from the counter.
    const bool caching = config_.kvCacheSize > 0;
    auto K_cache = caching ? (*this)["K_cache"].buffer() : eva::Buffer();
    auto V_cache = caching ? (*this)["V_cache"].buffer() : eva::Buffer();
    auto counter = caching ? kvCacheCounter : eva::Buffer();

    const auto& outShape = (*this)["out0"].shape();
    uint32_t Tq = outShape[outShape.size() - 3];

    // Bindings: 0 O, 1 Q, 2 K, 3 V, 4 K_cache, 5 V_cache, 6 kvCacheCount, then the
    // causal mask at 7 (naive) or 9 (FA, after the split-K partials).
    if (useFa)
    {
        struct {
            uint32_t Dh, Hq, groupSize, Tq, Tkv, Tmax, gqaRatio, kCount;
            float scale;
        } faConsts = { config_.headDim,
                       config_.numQHeads,
                       config_.numQHeads / Hkv,
                       Tq,
                       Tkv,
                       config_.kvCacheSize,
                       gqaRatioOf(pipeParamsFA),
                       splitKCount,
                       1.0f / std::sqrt(float(config_.headDim)) };

        // One workgroup per (tileTokens tokens x tileHeads heads); split-K multiplies x.
        // Unused partial bindings take O (= ggml's fully-bound trick).
        const PipeParamsFA& p = pipeParamsFA;
        bool split = splitKCount > 1;
        auto PO  = split ? (*this)["partialO"].buffer()  : O;
        auto PML = split ? (*this)["partialML"].buffer() : O;

        descSetFA.write({ O, Q, K, V, K_cache, V_cache, counter, PO, PML, causalMask });
        cmdBuff.bindPipeline(pipeFA)
            .bindDescSets({ descSetFA })
            .setPushConstants(0, sizeof(faConsts), &faConsts)
            .dispatch(CEIL_DIV(Tq, p.tileTokens) * splitKCount, config_.numQHeads / p.tileHeads, B);

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
        } constants = { config_.headDim,
                        config_.numQHeads,
                        config_.numQHeads / Hkv,
                        Tq,
                        Tkv,
                        config_.kvCacheSize,
                        1.0f / std::sqrt(float(config_.headDim)) };

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
