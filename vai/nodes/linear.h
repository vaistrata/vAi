#ifndef VAI_NODES_LINEAR_H
#define VAI_NODES_LINEAR_H

// Linear / fully-connected layer, tile (scalar / coopmat) and slab kernels
// (ports of the ggml/llama.cpp Vulkan mul_mm / mul_mat_vec kernels).
#include "vai/nodes/node-common.h"

namespace vai {

// Where Linear folds its "residual" in, relative to the epilogue.
enum class ResidualOrder : uint32_t {
    AfterEpilogue  = 0,   // a following Add (e.g. YOLO's bottleneck)
    BeforeEpilogue = 1,   // a bias that changes per run (e.g. pi0's time MLP)
};

}   // namespace vai

namespace vai::node {


/*=============================================================================
  Linear - O = I · Wᵀ, split between two output decompositions
  Input:  [..., inDim]
  Output: [..., outDim]

  vai naming: O [M, N], I [M, K] (M = flattened batch), W [N, K]
  (N = outDim, K = inDim).

  - slab (M <= 8, linear-slab): a workgroup owns wgN whole columns of O,
    so K is split across its threads and reduced; bias / epilogue are fused
    in-kernel. See useSlabPath().
  - tile (M > 8): a workgroup owns a wgM x wgN tile. Two kernels share the
    frame, chosen per device in ggmlMulmmTileParams(): scalar
    (linear-tile, each thread accumulates its own outputs) and
    cooperative-matrix (linear-cm, subgroup MMA tiles). On Adreno, Q4
    weights take a third, linear-q4-direct, which skips
    shared memory and never splits K (useQ4Direct()); where the device has
    packed int8 dot products, its int8 variant linear-q4-q8 runs over an
    input quantized by linear-q8-input (useQ4Q8()). Everything below
    applies to all of them:

  The tile params are selected per output shape in prepare(); when a shape
  change selects a different tile, the pipeline is re-requested from the pool.

  Split-K (splitSize option; 0 selects the ggml heuristic per shape)
  accumulates partials P[splitCount, M, N] (oDtype) in a temporary buffer
  and folds them in a second pass (linear-reduce); bias / epilogue / the
  residual live in that reduce pass, so requesting them without splitting still
  runs the reduce as a plain epilogue pass (splitCount = 1).

  O = s · epilogue(I · Wᵀ + bias) + R      residualOrder AfterEpilogue
  O = s · epilogue(I · Wᵀ + bias + R)      residualOrder BeforeEpilogue

  s is Config::scalarScale (e.g. dt of an Euler step x + dt · v), 1 by default.

  "bias" (wDtype) is a parameter and "residual" R (oDtype) an input. Each is
  [M, N] or rows that repeat over M: [N], or one image's position table under
  a batch.

  Q4 weights (vai::Dtype::Q4, K a multiple of 32) are dequantized as the tile
  kernels load them, so a Q4 Linear always takes the tile path; its bias is
  fp16.
=============================================================================*/
class Linear : public vai::Node
{
    eva::ComputePipeline pipeTile;
    eva::ComputePipeline pipeSlab;
    eva::ComputePipeline pipeReduce;
    eva::DescriptorSet descSetTile;
    eva::DescriptorSet descSetSlab;
    eva::DescriptorSet descSetReduce;
    eva::ComputePipeline pipeQ8;      // linear-q8-input, before linear-q4-q8
    eva::DescriptorSet descSetQ8;

public:
    // The device's coopmat GEMM capability. The reported (M, N, K) maps to
    // vai tile axes as (tN, tM, tK) — ggml feeds coopmat A from the weights,
    // so the coopmat M axis runs along vai's N. valid() = the device allows
    // the cm kernel at all (per-tile gates are coopmatTileUsable's).
    struct CmShape
    {
        uint32_t m = 0, n = 0, k = 0;   // device-reported coopmat shape
        bool accF16 = false;            // the same shape also accumulates in fp16

        bool valid() const { return m != 0; }
    };

    // Tile: wgSize threads = (wgM/sgM)*(wgN/sgN) subgroups of sgSize
    // lanes. Scalar path: each subgroup owns a sgM x sgN tile as
    // sgpMIter x sgpNIter pieces of sgpM x sgpN, each lane accumulating a
    // tM x tN register tile per piece. Coopmat path (tK != 0): the piece
    // level vanishes (sgp* = 0) and tN x tM is a subgroup-collective
    // coopmat accumulator tile.
    struct PipeParamsTile {
        uint32_t wgSize;
        uint32_t sgSize;    // scalar: may differ from the real subgroup size; coopmat: must equal it
        uint32_t wgM;       // workgroup tile
        uint32_t wgN;
        uint32_t wgK;       // k elements per workgroup tile
        uint32_t sgM;       // subgroup tile
        uint32_t sgN;
        uint32_t sgpM;      // subgroup tile piece
        uint32_t sgpN;
        uint32_t sgpMIter;  // sgM / sgpM
        uint32_t sgpNIter;  // sgN / sgpN
        uint32_t tM;        // thread tile
        uint32_t tN;
        uint32_t tK = 0;    // coopmat k tile: k depth per coopMatMulAdd (= device coopmat K);
                            // 0 on the scalar path — doubles as the coopmat-tile marker
        vai::Dtype iDtype = vai::Dtype::None;
        bool direct = false;   // linear-q4-direct: only wgSize, wgM, wgN, tM, tN apply (q4DirectTile)
        bool q8 = false;       // with direct: linear-q4-q8 over the int8-quantized input

        PipeParamsTile() = default;

        PipeParamsTile(
            uint32_t wgSize,
            uint32_t sgSize,
            uint32_t tN,
            uint32_t tM,
            uint32_t sgpNIter,
            uint32_t sgpMIter,
            uint32_t wgN,
            uint32_t wgK,
            uint32_t sgN
        ) : wgSize(wgSize), sgSize(sgSize), wgN(wgN), wgK(wgK), sgN(sgN)
          , sgpMIter(sgpMIter), sgpNIter(sgpNIter), tM(tM), tN(tN)
        {
            VAI_ASSERT((wgSize * tM * tN * sgpMIter * sgpNIter) % wgN == 0);
            wgM = wgSize * (tM * tN * sgpMIter * sgpNIter) / wgN;

            VAI_ASSERT(sgN % sgpNIter == 0);
            sgpN = sgN / sgpNIter;
            VAI_ASSERT((sgSize * tM * tN) % sgpN == 0);
            sgpM = (sgSize * tM * tN) / sgpN;
            sgM = sgpMIter * sgpM;

            // The subgroups must tile the workgroup tile exactly.
            VAI_ASSERT(wgM % sgM == 0 && wgN % sgN == 0);
            // The thread tiles must tile the subgroup tile piece exactly
            // (with the exact divisions above, this also guarantees
            //  sgSize = (sgpM/tM)*(sgpN/tN) and wgSize = (wgM/sgM)*(wgN/sgN)*sgSize).
            VAI_ASSERT(sgpM % tM == 0 && sgpN % tN == 0);
        }

        bool operator==(const PipeParamsTile&) const = default;
    };

    struct PipeParamsSlab {
        uint32_t M = 0;
        uint32_t biasRepeat = 0;
        uint32_t residualRepeat = 0;
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParamsSlab&) const = default;
    };

    struct PipeParamsReduce {
        uint32_t biasRepeat = 0;
        uint32_t residualRepeat = 0;

        bool operator==(const PipeParamsReduce&) const = default;
    };

    // A warptile's workgroup grid over the (M, N) output: workgroup counts
    // per axis, and whether the grid's padding reaches M/2 rows.
    struct TileGrid {
        uint32_t tilesM, tilesN;
        bool smallM;

        TileGrid(uint32_t M, uint32_t N, uint32_t wgM, uint32_t wgN)
        : tilesM(CEIL_DIV(M, wgM)), tilesN(CEIL_DIV(N, wgN))
        , smallM(tilesM * wgM >= M + M / 2u)
        {}

        uint32_t tiles() const { return tilesM * tilesN; }
    };

    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        bool hasBias = false;
        bool acceptResidual = false;  // input "residual"
        vai::ResidualOrder residualOrder = vai::ResidualOrder::AfterEpilogue;
        vai::Epilogue epilogue = vai::Epilogue::None;
        float scalarScale = 1.0f;     // s, right after the epilogue
        vai::Dtype wDtype = vai::Dtype::None;   // None: wDefaultDtype; Q4 allowed here
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t splitSize = 0;   // split-K: K length per split (0: ggml auto heuristic)
        vai::Dtype accDtype = vai::Dtype::None;
        // Q4 weights on Adreno: quantize the input to int8 per 32 elements and
        // multiply in integers (linear-q4-q8), where the device allows. About
        // 7-9x faster there and adds ~4e-3 relative error; false keeps the
        // input as is (linear-q4-direct).
        bool int8Input = true;
    } config_;

    Linear(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    // The reduce pass runs when K is split and/or bias / epilogue / the residual
    // is requested (the tile kernel itself carries no epilogue).
    bool needReducePass() const
    {
        return splitCount > 1 || config_.hasBias || config_.acceptResidual
            || config_.epilogue != Epilogue::None || config_.scalarScale != 1.0f;
    }

    // Small batches run the slab kernel instead of the tile one;
    // 8 = ggml mul_mat_vec_max_cols. The slab kernel reads no Q4.
    bool useSlabPath(uint32_t M) const
    {
        return M <= 8u && config_.wDtype != Dtype::Q4;
    }

    Dtype biasDtype() const
    {
        return (config_.wDtype == Dtype::Q4) ? Dtype::Fp16 : config_.wDtype;
    }

    PipeParamsTile pipeParamsTile;
    PipeParamsSlab pipeParamsSlab;
    PipeParamsReduce pipeParamsReduce;
    uint32_t splitSize  = 0;  // K length per split for the current shape (= K when not splitting)
    uint32_t splitCount = 1;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Linear
/////////////////////////////////////////////////////////////////////////////////////////

// Vendor screen for the linear cm kernel (ggml_vk_khr_cooperative_matrix_support
// port): drivers that report the coopmat extension but aren't trusted to run
// it well.
inline bool ggmlCoopmatAllowedLinear(eva::Device device)
{
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::INTEL:
        return device.architectureID() == eva::Architecture::INTEL_XE2
            || (device.architectureID() == eva::Architecture::INTEL_XE1
                && device.deviceType() == eva::DEVICE_TYPE::INTEGRATED_GPU
                && device.driverID() == eva::DRIVER_ID::INTEL_PROPRIETARY_WINDOWS);
    case eva::VENDOR_ID::AMD:
        if (device.driverID() == eva::DRIVER_ID::AMD_PROPRIETARY
            || device.driverID() == eva::DRIVER_ID::AMD_OPEN_SOURCE)
            return device.architectureID() == eva::Architecture::AMD_POST_RDNA2;   // over-reporting drivers: RDNA3 only
        return true;   // RADV etc: trusted as-is
    default:
        return true;   // pre-Turing NVIDIA lacks the extension itself
    }
}

// Every device-side coopmat condition in one answer (ggml's coopmat_support
// flag): extension + vendor screen + shape lock — the first fp16 A/B
// subgroup-scope shape wins, only acc types reported for that exact shape
// count, and no fp32 acc disables the path entirely.
//
// NOTE (vs llama.cpp): ggml also gates on computeFullSubgroups and sets
// REQUIRE_FULL_SUBGROUPS_BIT (warps are placed by gl_SubgroupID); eva
// exposes neither yet. Full subgroups still hold structurally — eva never
// uses ALLOW_VARYING_SUBGROUP_SIZE, and coopmatTileUsable() requires
// wgSize % sgSize == 0 (= ggml's fallback without the feature).
inline Linear::CmShape ggmlCoopmatShape(eva::Device device)
{
    Linear::CmShape s;
    if (!device.supportsCooperativeMatrix() || !ggmlCoopmatAllowedLinear(device))
        return s;

    bool accF32 = false;
    for (const auto& p : device.cooperativeMatrixProperties())
    {
        if (p.aType != eva::COMPONENT_TYPE::FLOAT16
            || p.bType != eva::COMPONENT_TYPE::FLOAT16
            || p.scope != eva::SCOPE::SUBGROUP)
            continue;

        const bool f32 = p.cType == eva::COMPONENT_TYPE::FLOAT32
                      && p.resultType == eva::COMPONENT_TYPE::FLOAT32;
        const bool f16 = p.cType == eva::COMPONENT_TYPE::FLOAT16
                      && p.resultType == eva::COMPONENT_TYPE::FLOAT16;
        if (!f32 && !f16)
            continue;

        if (!s.valid())
        {
            s.m = p.M; s.n = p.N; s.k = p.K;
        }
        else if (s.m != p.M || s.n != p.N || s.k != p.K)
        {
            continue;   // acc types count only for the locked shape
        }

        accF32   = accF32 || f32;
        s.accF16 = s.accF16 || f16;
    }

    if (!accF32)
        s = {};   // ggml: no fp32 accumulation -> no matrix cores
    return s;
}


// Whether a coopmat tile candidate works on this device. A failure means
// "not on this device" (degrade the tier / fall back to scalar), not a bad
// table.
inline bool coopmatTileUsable(const Linear::PipeParamsTile& t, eva::Device device)
{
    // Subgroups exactly cover the wg tile (warps are placed by gl_SubgroupID).
    if (t.wgM % t.sgM != 0 || t.wgN % t.sgN != 0 || t.wgSize % t.sgSize != 0)
        return false;
    if ((t.wgM / t.sgM) * (t.wgN / t.sgN) != t.wgSize / t.sgSize)
        return false;

    // The coopmat grid exactly covers the subgroup tile.
    if (t.sgN % t.tN != 0 || t.sgM % t.tM != 0)
        return false;

    // The k loop steps by tK; f16vec2 slots need even tK (i / 2 offsets).
    if (t.wgK % t.tK != 0 || t.tK % 2 != 0)
        return false;

    // Staged-store lane split. NOTE (vs llama.cpp): ggml skips this — every
    // reported coopmat shape satisfies it (power-of-two dims, sgSize/tN <= tM).
    if (t.sgSize % t.tN != 0 || t.tM % (t.sgSize / t.tN) != 0)
        return false;

    // Shared budget: tileA/tileB + the fp32 store stage. NOTE (vs llama.cpp):
    // ggml divides the stage term by warps — the shader declares
    // cmStage[warps * tN * tM], so vai multiplies.
    const uint32_t shmem = (t.wgN + t.wgM) * (t.wgK / 2 + 4) * 4
                         + (t.wgSize / t.sgSize) * t.tN * t.tM * 4;
    return shmem <= device.maxComputeSharedMemorySize();
}


// Facts shared by the tile / split selectors below.

// Does the ggml L (128x128) grid cover the cores? Grids that are not small-M
// count from ~1x cores (an exact fit needs no more); small-M ones only from 1.5x.
inline bool cmGridFilled(uint32_t M, uint32_t N, uint32_t cores)
{
    const Linear::TileGrid g(M, N, 128u, 128u);
    return g.tiles() >= cores * 3u / 2u || (!g.smallM && g.tiles() >= cores);
}

// A refined pick is geometry-recognizable: ggml's coopmat tiles are all
// square k32, refined ones never are.
inline bool cmRefinedTile(const Linear::PipeParamsTile& t)
{
    return t.tK != 0u && (t.wgK != 32u || t.wgM != t.wgN);
}

// The K span each split covers, rounded so whole 256-blocks stream.
inline uint32_t ggmlSplitSize(uint32_t K, uint32_t splitCount)
{
    return ROUNDUP_POW2(CEIL_DIV(K, splitCount), 256u);
}


// The ggml mul_mm l/m/s warptiles as vai tile params (ggml's M/N are vai's
// N/M; ggml's WNITER maps to sgpMIter and is precomputed here as
// (WM x WN) / (WARP x TM x TN x WMITER)). The coopmat candidates take the
// same geometry with their thread-tile level replaced.
inline Linear::PipeParamsTile ggmlScalarWarptile(int tier, eva::Device device)
{
    const uint32_t sgSize = device.subgroupSize();
    const uint32_t sg8  = std::max(sgSize, 8u);
    const uint32_t sg32 = std::max(sgSize, 32u);
    const uint32_t sSgN = (sgSize == 8) ? 8 : (sgSize == 16) ? 16 : 32;

    // ( wgSize, sgSize, tN, tM, sgpNIter, sgpMIter, wgN, wgK, sgN )
    if (tier == 0)          // S
        return {
            sg32, sg8,
            2, 2, 2, 4 * sSgN / sg8,
            32, 32, sSgN
        };

    if (tier == 1)          // M
        return {
            128, sg8,
            4, 2, 2, 2,
            64, 32, sg8
        };

    return {                // L
        128, sg8,
        4, 4, 2, 4,
        128, 32, sg8 * 2
    };
}


// ggml_vk_guess_matmul_pipeline port: the shape tier, narrowed to the tiers
// this device may run (ggml's device->mul_mat_l/m/s — a vendor switch sets
// it, a shared-memory check clears what does not fit). `coopmat` picks which
// kernel the tiers stand for: upstream makes L conditional on coopmat for
// Intel and AMD, and the coopmat tiles carry their own shared-memory check.
inline int ggmlGuessTier(uint32_t M, uint32_t N, eva::Device device, bool coopmat, bool stageF32 = false)
{
    // tileA/tileB slots (f16vec2, or vec2 with fp32 staging) over stride
    // wgK / 2 + 1. S fits everywhere. ggml sizes both with sizeof(float).
    const uint32_t slotBytes = stageF32 ? 8u : 4u;
    auto fits = [&](int tier) {
        const Linear::PipeParamsTile t = ggmlScalarWarptile(tier, device);
        return (t.wgN + t.wgM) * (t.wgK / 2 + 1) * slotBytes <= device.maxComputeSharedMemorySize();
    };

    const uint32_t vendor = device.vendorID();
    const bool appleLike  = vendor == eva::VENDOR_ID::APPLE          // medium only
                         || device.driverID() == eva::DRIVER_ID::MESA_HONEYKRISP;

    const bool mmS = !appleLike;
    const bool mmM = coopmat || fits(1);
    const bool mmL = !appleLike
                  && vendor != eva::VENDOR_ID::QUALCOMM
                  && (coopmat || (fits(2)                              // scalar: shared memory,
                               && vendor != eva::VENDOR_ID::INTEL      // L wants coopmat, and
                               && vendor != eva::VENDOR_ID::AMD));     // a trusted driver on AMD

    if ((mmS && (M <= 32 || N <= 32)) || (!mmM && !mmL)) return 0;
    if ((mmM && (M <= 64 || N <= 64)) || !mmL)           return 1;
    return 2;
}


// A tier's coopmat warptile: the scalar tier with its thread-tile level
// replaced by the device coopmat shape (the derived wgM / sgM are ggml's
// path-independent warptile slots).
inline Linear::PipeParamsTile ggmlCoopmatWarptile(int tier, const Linear::CmShape& cm, eva::Device device)
{
    const uint32_t sgSize = device.subgroupSize();
    const uint32_t sg8    = std::max(sgSize, 8u);

    Linear::PipeParamsTile t = ggmlScalarWarptile(tier, device);
    if (tier == 2)   // coopmat-only L vendor overrides (ggml l_warptile tuning)
    {
        if (device.vendorID() == eva::VENDOR_ID::AMD)        { t.wgSize = 256; t.sgN = sg8; }
        else if (device.vendorID() == eva::VENDOR_ID::INTEL) { t.wgSize = 512; t.sgN = sg8; t.sgM = 32; }
    }

    t.sgSize = sgSize;    // real subgroup size: warps place by gl_SubgroupID
    t.tN     = cm.m;      // device coopmat M runs along vai's N
    t.tM     = cm.n;
    t.tK     = cm.k;
    t.sgpM = t.sgpN = t.sgpMIter = t.sgpNIter = 0;   // no thread-tile level
    return t;
}

// Shape-aware tile-param selection (vai naming: M = flattened batch,
// N = outDim): the device coopmat tile where one is usable, refined per vendor
// and shape in the second half below — the gated scalar warptile otherwise.
inline Linear::PipeParamsTile ggmlMulmmWarptile(uint32_t M, uint32_t N, uint32_t K, eva::Device device, bool stageF32 = false)
{
    // Coopmat candidate first: a gated-out tier degrades L -> M -> S; when
    // every tier fails, the scalar kernel runs instead.
    Linear::PipeParamsTile t{};
    bool cmUsable = false;
    const Linear::CmShape cm = ggmlCoopmatShape(device);
    if (cm.valid())
    {
        for (int tier = ggmlGuessTier(M, N, device, true); tier >= 0; tier--)
        {
            t = ggmlCoopmatWarptile(tier, cm, device);
            if (coopmatTileUsable(t, device)) { cmUsable = true; break; }
        }
    }
    if (!cmUsable)
        return ggmlScalarWarptile(ggmlGuessTier(M, N, device, false, stageF32), device);

    // Per-shape refinement of the ggml pick (= upstream's vendor l_warptile
    // overrides, made shape-aware). Wherever no rule applies, ggml's pick
    // stands; the paired split rules live in ggmlGuessSplitCount.
    const uint32_t cores = device.computeBlockCount();
    // Every rule below wants a core count and was measured on whole k-blocks
    // (the picks step k32 / k64; the lone k128 pick checks K % 128 itself).
    if (cores == 0 || K % 64 != 0) return t;

    uint32_t wgM, wgN, wgK, sgM, sgN;
    uint32_t wgSize = 128;

    if (device.vendorID() == eva::VENDOR_ID::AMD && t.sgSize == 64)
    {
        // UMA iGPUs: the weight stream dominates, so tiles go M-fat with
        // k64 blocks and never split. Discrete parts price weight re-reads
        // with VRAM + caches instead and stay on ggml's pick, as do low CU
        // counts.
        if (device.deviceType() != eva::DEVICE_TYPE::INTEGRATED_GPU || cores < 16u)
            return t;

        if (N <= 512u && M <= 256u)   // narrow layers, small M: at any CU count
        {
            wgM = 32; wgN = 32; wgK = 64; sgM = 32; sgN = 32; wgSize = 64;
        }
        else if (cores >= 32u)   // >= 32 CUs: one tile per layer width
        {
            if (N >= 8192u)      // widest layers: one fat M band of 32-wide
            {                    // columns, so the weight streams once per band
                wgK = (M >= 1024u) ? 32u : 64u;
                if (M <= 192u) { wgM = 192; wgN = 32; sgM = 96;  sgN = 16; wgSize = 256; }
                else           { wgM = 256; wgN = 32; sgM = 256; sgN = 16; }
            }
            else if (N <= 512u)  // narrow layers, bigger M
            {
                wgM = 128; wgN = 128; wgK = 64; sgM = 128; sgN = 16; wgSize = 512;
            }
            else                 // mid widths (q/o, down-proj class)
            {
                // One M band pays only while the grid stays wide and deep;
                // narrow or shallow layers keep the small square.
                if (M <= 256u)
                {
                    if (K >= 2048u && N >= 2048u) { wgM = 192; wgN = 64; wgK = 64;
                                                    sgM = 96;  sgN = 16; wgSize = 512; }
                    else                          { wgM = 64;  wgN = 64; wgK = 64;
                                                    sgM = 64;  sgN = 16; wgSize = 256; }
                }
                else if (M < 1024u) { wgM = 256; wgN = 128; wgK = 64;
                                      sgM = 256; sgN = 16;  wgSize = 512; }
                else                { wgM = 96;  wgN = 128; wgK = 64;
                                      sgM = 96;  sgN = 32;  wgSize = 256; }
            }
        }
        else                     // < 32 CUs: one general tile, plus a
        {                        // narrow-N and a mid-M case
            if (N <= 512u)       // narrow layers, bigger M
            {
                wgM = 64; wgN = 128; wgK = 64; sgM = 32; sgN = 128;
            }
            else if (M > 256u && M < 2048u && (M < 1024u || K >= 8192u))
            {                    // mid M rides a taller-N tile
                wgM = 128; wgN = 256; wgK = 64; sgM = 32; sgN = 256; wgSize = 256;
            }
            else                 // the general tile
            {
                wgM = 192; wgN = 128; wgK = 64; sgM = 48; sgN = 128; wgSize = 256;
            }
        }
    }
    else if (device.vendorID() == eva::VENDOR_ID::NVIDIA && t.sgSize == 32)
    {
        // Measured per shape and SM count; the split is 40-95 SMs vs. the rest.
        const Linear::TileGrid gridL(M, N, 128u, 128u);   // ggml's L tile
        const uint32_t tilesL = gridL.tiles();
        const bool smallM = gridL.smallM;
        // Big-L2 proxy: weight re-reads are cheap on the high-core parts.
        const bool coresRich = cores >= 96u;
        const bool midCores = cores >= 40u && !coresRich;
        const bool filled = cmGridFilled(M, N, cores);

        if (midCores)
        {
            if (smallM)
            {
                if (K >= 4096u || N <= 1024u) return t;   // ggml (with its split) wins these
                if (tilesL >= cores * 3u)
                {
                    // Many waves at small M (gate/up, lm_head): one M tile.
                    wgM = (M <= 96u) ? 96u : 192u;
                    wgN = 96; wgK = 64; sgM = wgM / 2; sgN = 48;
                }
                else
                {
                    wgM = 32; wgN = 128; wgK = 64; sgM = 16; sgN = 32; wgSize = 256;
                }
            }
            else
            {
                if (N < 2048u) return t;
                const uint32_t waves = CEIL_DIV(tilesL, cores);
                if (tilesL >= cores * 2u)
                {
                    // Deep grids: 1.5x-area tiles; grow M against a wide N,
                    // otherwise grow N.
                    if (N >= 4u * M) { wgM = 256; wgN = 96;  sgM = 64; sgN = 96; }
                    else             { wgM = 128; wgN = 192; sgM = 64; sgN = 96; }
                    wgK = 32;
                }
                else if (tilesL >= cores && (waves * cores - tilesL) * 4u >= waves * cores)
                {
                    // 1-2 waves with >= 25% of the round-up idle: halve the M tile.
                    wgM = 64; wgN = 128; wgK = 32; sgM = 32; sgN = 64;
                }
                else return t;   // near-exact fit: ggml's L is already right
            }
        }
        else if (filled)
        {
            // Trim wgM to the smallest step keeping the tile count (pure waste cut).
            wgM = 128;
            for (uint32_t w : { 32u, 64u, 96u })
                if (CEIL_DIV(M, w) == CEIL_DIV(M, 128u)) { wgM = w; break; }
            wgN = 128;
            wgK = (K >= 4096u) ? 64u : 32u;
            sgM = wgM / 2; sgN = 64;                    // 4 subgroups
            if (wgM == 128u && wgK == 32u) return t;    // = ggml's L exactly
        }
        else if (K >= 4096u && !smallM)
        {
            wgM = 128; wgN = 128; wgK = 64; sgM = 64; sgN = 64;   // L, deeper k-block
        }
        else if (K >= 4096u && !coresRich)
        {
            wgM = 96; wgN = 96; wgK = 64; sgM = 48; sgN = 48;     // tall tile: weights stream once per 96 rows
        }
        else
        {
            // Underfilled: 32-row tiles, the widest wgN that still fills.
            wgM = 32;
            wgN = 64;
            for (uint32_t w : { 128u, 96u })
                if (Linear::TileGrid(M, N, wgM, w).tiles() >= cores * 6u / 5u) { wgN = w; break; }
            const uint32_t preTiles = Linear::TileGrid(M, N, wgM, wgN).tiles();
            wgK = (preTiles < cores / 2u && K % 128u == 0u) ? 128u : 64u;
            sgM = (wgN == 128u) ? 32u : 16u;
            sgN = (wgN == 128u) ? 32u : wgN / 2;        // 4 subgroups either way
        }
    }
    else return t;   // every other vendor / subgroup size: ggml's pick stands

    Linear::PipeParamsTile tuned = t;
    tuned.wgSize = wgSize;
    tuned.wgM = wgM; tuned.wgN = wgN; tuned.wgK = wgK;
    tuned.sgM = sgM; tuned.sgN = sgN;
    // ggmlGuessSplitCount recognizes these picks by geometry: no branch above
    // may land on a square k32 tile, which is what ggml's own picks all are.
    VAI_ASSERT(cmRefinedTile(tuned));
    return coopmatTileUsable(tuned, device) ? tuned : t;
}


// Tile pipeline params for one shape: the warptile above, with the input dtype.
inline Linear::PipeParamsTile ggmlMulmmTileParams(uint32_t M, uint32_t N, uint32_t K, Dtype iDtype, Dtype wDtype, eva::Device device)
{
    Linear::PipeParamsTile t = ggmlMulmmWarptile(M, N, K, device, iDtype == Dtype::Fp32 && wDtype == Dtype::Fp32);
    t.iDtype = iDtype;
    return t;
}


// Q4 weights on Adreno run linear-q4-direct, which reads W
// and I straight from memory instead of staging them in shared memory. On an
// Adreno 830 (Galaxy S25 Ultra) it runs 3840- and 10240-wide Linears at
// M = 1056 about 1.8x faster than the scalar tile kernel, and no tile shape
// of that kernel measured faster than its default. It always accumulates in
// fp32. Other GPUs keep the tile kernels: it is unmeasured there.
inline bool useQ4Direct(Dtype iDtype, Dtype wDtype, eva::Device device)
{
    return wDtype == Dtype::Q4 && (iDtype == Dtype::Fp16 || iDtype == Dtype::Fp32)
        && device.vendorID() == eva::VENDOR_ID::QUALCOMM;
}

// Its int8 variant, linear-q4-q8, quantizes I to int8 per 32 elements
// (linear-q8-input) and computes each group's sum with packed 4x8 integer dot
// products. Where the device runs those natively, it replaces the direct
// kernel's dequantize-and-fma work: on the Adreno 830 at M = 1056, 3840-wide
// Linears run 9.0x faster than linear-q4-direct and the FFN shapes 6.7x
// (quantization included), at ~4e-3 relative error against the fp product,
// small next to Q4's own.
inline bool useQ4Q8(eva::Device device)
{
    return device.isDot4x8PackedMixedAccelerated();
}

// Their tiles (wgM = wgSize / (wgN / tN) * tM):
// - linear-q4-direct: 128 threads, each owning a 4 x 4 block of O; 32 threads
//   along N cover 128 columns and 4 along M cover 16 rows. On the Adreno 830,
//   the other blocks tried (1-8 rows by 1-8 columns, 64-256 threads, 256
//   columns) ran 1-140% slower.
// - linear-q4-q8: 64 threads (one Adreno subgroup), each owning a 2 x 4 block;
//   8 threads along N cover 32 columns and 8 along M cover 16 rows. Its inner
//   loop is cheap enough that the W loads dominate once W outgrows the cache,
//   and a subgroup's W load touches one cache line per thread along N: 32 with
//   the direct tile, 8 here. On the Adreno 830 that runs a 10240 x 10240
//   Linear 3.3x faster than the direct tile and the FFN 2.0x; the other
//   blocks tried (1-8 rows by 1-4 columns, 64-256 threads, 8-128 columns)
//   were slower on 3840- or 10240-wide Linears or on the FFN. The shader
//   fixes this tile at compile time.
inline Linear::PipeParamsTile q4DirectTile(Dtype iDtype, bool q8)
{
    Linear::PipeParamsTile t{};
    t.direct = true;
    t.q8 = q8;
    t.wgSize = q8 ? 64 : 128;
    t.tM = q8 ? 2 : 4;
    t.tN = 4;
    t.wgN = q8 ? 32 : 128;
    t.wgM = t.wgSize / (t.wgN / t.tN) * t.tM;
    t.iDtype = iDtype;
    return t;
}


// Split-count selection; a refined pick carries its own split rule.
inline uint32_t ggmlGuessSplitCount(uint32_t M, uint32_t N, uint32_t K,
                                    const Linear::PipeParamsTile& t, eva::Device device)
{
    const uint32_t cores = device.computeBlockCount();
    const uint32_t tiles = Linear::TileGrid(M, N, t.wgM, t.wgN).tiles();

    if (cmRefinedTile(t))
    {
        if (device.vendorID() == eva::VENDOR_ID::AMD)
            return 1;   // AMD picks exist only on the UMA iGPUs, where splits lose

        if (K < 2048u || (cores >= 40u && cores < 96u))
            return 1;   // shallow K, and every mid-core NVIDIA pick, run unsplit
                        // (also floors the K/1024, K/256 targets below away from 0)

        if (t.wgM == 96u && t.wgN == 96u)   // tall tile: fewer, fatter workgroups
            return std::min(ROUNDUP_POW2(CEIL_DIV(cores * 4u, tiles), 2u),   // ~4 wgs/core
                            std::min(8u, K / 1024u));

        if (t.wgM == 32u && !cmGridFilled(M, N, cores))   // 32-row underfill
            return std::min(std::max((2u * cores + tiles / 2u) / tiles, 1u),   // ~2 wgs/core
                            std::min(8u, K / 256u));

        // The filled trims and the deep-K L take ggml's guess below.
    }

    // ggml_vk_guess_split_k port (ggml m/n are vai N/M): split only an
    // underfilled deep-K grid; shrink while the 256-rounded split size would
    // leave the last split empty.
    if (cores == 0 || N < t.wgN || M < t.wgM || K < 2048)
        return 1;

    uint32_t splitCount = 1;
    if (tiles <= cores / 2)          splitCount = cores / tiles;
    else if (tiles <= cores * 2 / 3) splitCount = 3;
    splitCount = std::min(splitCount, 8u);

    while (splitCount > 1)
    {
        const uint32_t splitSize = ggmlSplitSize(K, splitCount);
        if (splitSize * (splitCount - 1) < K)
            break;
        splitCount--;
    }
    return splitCount;
}

// ggml mul_mat_vec subgroup-size rule: Intel runs the slab kernel at subgroup size
// 16 when the device can pin it (Xe is SIMD8/16 native); everyone else uses
// the device default. The pinned size feeds the workgroup sizing below and
// the pipeline's requiredSubgroupSize.
inline uint32_t ggmlSlabSgSize(eva::Device device)
{
    uint32_t sg = device.subgroupSize();
    if (device.vendorID() == eva::VENDOR_ID::INTEL
        && device.minSubgroupSize() <= 16u && 16u <= device.maxSubgroupSize())
        sg = 16;
    return sg;
}

// ggml mul_mat_vec workgroup-size heuristic port (ggml m/k are vai N/K):
// NVIDIA/Intel spread deep-K, few-row layers (N <= 8192, K >= 1024) over a
// 4-subgroup workgroup (LARGE, hybrid reduction); everything else stays at a
// single subgroup (SUBGROUP) — enough rows to fill the GPU, so the extra
// reduction stage would be pure overhead. Pre-Turing NVIDIA is excluded like
// upstream: its fingerprint there is the absence of VK_KHR_cooperative_matrix,
// which supportsCooperativeMatrix() reports (eva::architectureID() carries
// no PRE_TURING class).
//
// NOTE (vs llama.cpp): the AMD branch is a deliberate deviation — RDNA3+
// picks LARGE for K >= 1024, no N cap.
inline uint32_t ggmlSlabWgSize(uint32_t K, uint32_t N, uint32_t sgSize, eva::Device device)
{
    const uint32_t vendor = device.vendorID();
    const bool nvPostTuring = vendor == eva::VENDOR_ID::NVIDIA && device.supportsCooperativeMatrix();
    if ((nvPostTuring || vendor == eva::VENDOR_ID::INTEL)
        && N <= 8192u && 1024u <= K )
        return sgSize * 4u;   // LARGE
    if (device.architectureID() == eva::Architecture::AMD_POST_RDNA2 && 1024u <= K)
        return sgSize * 4u;   // LARGE (RDNA3+)
    return sgSize;            // SUBGROUP
}

inline Linear::Linear(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    addSlot("partial", NodeSlot::temporary);   // assigned only when the reduce pass runs
    addSlot("inQ8", NodeSlot::temporary);      // assigned only on the linear-q4-q8 path

    if (config_.hasBias)        addSlot("bias", NodeSlot::persistent);
    if (config_.acceptResidual) addSlot("residual", NodeSlot::input);

    config_.wDtype = (config_.wDtype == Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
    config_.accDtype = (config_.accDtype == Dtype::None) ? wDefaultDtype() : config_.accDtype;
}

inline void Linear::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    const Tensor& wTen  = (*this)["weight"];

    uint32_t N = config_.outDim;
    uint32_t K = config_.inDim;
    VAI_ASSERT(inTen.shape().back() == K);

    const Dtype iDtype = inTen.dtype();
    VAI_ASSERT(wTen.dtype() == config_.wDtype);
    VAI_ASSERT(wTen.isShapeOf(N, K));
    VAI_ASSERT(config_.wDtype != Dtype::Q4 || K % 32 == 0);

    const auto& inShape = inTen.shape();
    uint32_t M = outerSize(inShape);

    // An operand block that covers O is 0, so the shader indexes it directly;
    // a shorter one ([N], or one image's rows) is its length, read modulo.
    auto repeatOf = [&](const Tensor& t)
    {
        const uint32_t outElems = M * N, elems = t.numElements();
        VAI_ASSERT(t.shape().back() == N && elems > 0 && outElems % elems == 0);
        return (elems == outElems) ? 0u : elems;
    };

    uint32_t biasElems = 0;
    if (config_.hasBias)
    {
        const Tensor& bTen = (*this)["bias"];
        VAI_ASSERT(bTen.dtype() == biasDtype());
        biasElems = repeatOf(bTen);
    }

    uint32_t residualElems = 0;
    if (config_.acceptResidual)
    {
        const Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.dtype() == config_.oDtype);
        residualElems = repeatOf(rTen);
    }

    if (useSlabPath(M))
    {
        // Slab policy (device / option-derived) — recomputed per prepare like
        // the tile params below; run() only re-derives wgN.
        const uint32_t sgSize  = ggmlSlabSgSize(device());
        const uint32_t wgSize  = ggmlSlabWgSize(K, N, sgSize, device());
        const uint32_t wgN = (config_.wDtype == Dtype::Fp16) ? 2u : 1u;   // fp16 W pairs N positions
        const bool     aligned = (K % 4 == 0);   // K_PER_ITER = 4 wide loads

        const PipeParamsSlab p = { .M = M, .biasRepeat = biasElems, .residualRepeat = residualElems, .iDtype = iDtype };
        if (!pipeSlab || p != pipeParamsSlab)
        {
            pipeParamsSlab = p;

            pipeSlab = pipelinePool().request("linear-slab", {
                eva::constant_id<0>(wgSize),                          // wgSize (= local_size_x)
                eva::constant_id<1>(sgSize),                          // sgSize (pinned subgroup size)
                eva::constant_id<2>(M),                               // M (flattened batch)
                eva::constant_id<3>(wgN),                             // wgN
                eva::constant_id<4>(config_.hasBias ? 1u : 0u),       // hasBias
                eva::constant_id<5>(biasElems),                       // bias repeat (0: covers O)
                eva::constant_id<6>(config_.acceptResidual ? 1u : 0u),   // hasResidual
                eva::constant_id<7>(residualElems),                   // residual repeat (0: covers O)
                eva::constant_id<8>((uint32_t) config_.residualOrder),   // residualOrder
                eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),     // typeO
                eva::constant_id<SPEC_ID_TYPE_I>(iDtype),             // typeI
                eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),     // typeW (W & bias)
                eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u), // aligned (K % 4 == 0)
                eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)   // epilogue
            }, {
                .requiredSubgroupSize = sgSize,
            });

            if (!descSetSlab)
                descSetSlab = gVaiDestSetPool(pipeSlab.descSetLayout(0));
        }
    }
    else
    {
        // fp32 x fp32 on the scalar tile kernel stages and accumulates fp32;
        // coopmat (like llama.cpp) and any fp16 operand keep fp16 staging.
        const bool stageF32 = iDtype == Dtype::Fp32 && config_.wDtype == Dtype::Fp32;
        // An explicit splitSize keeps the tile kernels, which split K.
        const bool direct = config_.splitSize == 0 && useQ4Direct(iDtype, config_.wDtype, device());
        const bool q8 = direct && config_.int8Input && useQ4Q8(device());
        const PipeParamsTile t = direct ? q4DirectTile(iDtype, q8)
                                        : ggmlMulmmTileParams(M, N, K, iDtype, config_.wDtype, device());

        // Resolve split-K for this shape: none for the direct kernel, an
        // explicit splitSize, or the ggml heuristic when 0.
        if (direct)
        {
            splitSize  = K;
            splitCount = 1;
        }
        else if (config_.splitSize > 0)
        {
            VAI_ASSERT(config_.splitSize >= 32);
            splitSize  = config_.splitSize;
            splitCount = CEIL_DIV(K, splitSize);
        }
        else
        {
            splitCount = ggmlGuessSplitCount(M, N, K, t, device());
            splitSize  = (splitCount > 1) ? ggmlSplitSize(K, splitCount) : K;
        }

        if (direct && (!pipeTile || t != pipeParamsTile))
        {
            pipeParamsTile = t;

            if (t.q8)
            {
                pipeQ8 = pipelinePool().request("linear-q8-input", {
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype)             // typeI
                });
                if (!descSetQ8)
                    descSetQ8 = gVaiDestSetPool(pipeQ8.descSetLayout(0));
            }

            // Same constants for both kernels; linear-q4-q8 reads no I dtype.
            pipeTile = pipelinePool().request(t.q8 ? "linear-q4-q8" : "linear-q4-direct", {
                eva::constant_id<0>(t.wgSize),                       // wgSize (= local_size_x)
                eva::constant_id<1>(t.wgN),                          // wgN
                eva::constant_id<2>(t.tM),                           // tM
                eva::constant_id<3>(t.tN),                           // tN
                eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
                eva::constant_id<SPEC_ID_TYPE_I>(iDtype)             // typeI
            });

            if (!descSetTile)
                descSetTile = gVaiDestSetPool(pipeTile.descSetLayout(0));
        }
        else if (!direct && (!pipeTile || t != pipeParamsTile))
        {
            pipeParamsTile = t;

            // Whole k-blocks need K % wgK == 0 (and each split starting on a
            // k-block boundary); otherwise the kernel's scalar masked-load path
            // is selected. Heuristic splits are multiples of 256, so only wgK
            // among the key moves this.
            const bool aligned = (K % t.wgK == 0) && (splitSize % t.wgK == 0);

            // Robustness folds the loaders' row-tail guards away (spec 12):
            // garbage from a robust OOB tail read only feeds outputs the
            // stores mask off (the K-tail masks stay — that axis is summed).
            // Guarded loads stall staging: 17-33% measured with device-local
            // input (pinned inputs invert that — see tests/linear-cm).
            // NOTE (vs llama.cpp): ggml relies on core robustBufferAccess,
            // always on; vai needs VK_EXT_pipeline_robustness, else guards stay.
            const bool robust = device().supportsPipelineRobustness();

            // Loader constraints, common to both tile kernels (they share
            // the cooperative loaders): wgK 8-wide-load aligned, and the
            // loaders sweep whole wgM/wgN tiles per pass (elements per load
            // call: aligned 8, scalar 2).
            const uint32_t el = aligned ? 8u : 2u;
            VAI_ASSERT(t.wgK % 8 == 0);
            VAI_ASSERT((t.wgSize * el) % t.wgK == 0);
            VAI_ASSERT(t.wgM % (t.wgSize * el / t.wgK) == 0 && t.wgN % (t.wgSize * el / t.wgK) == 0);

            if (t.tK != 0)   // coopmat kernel
            {
                // fp16 accumulation only when the locked shape reports it;
                // otherwise promote to fp32 (ggml falls back to its f32acc
                // pipeline the same way).
                const Dtype acc = (config_.accDtype == Dtype::Fp16
                                   && !ggmlCoopmatShape(device()).accF16)
                                ? Dtype::Fp32 : config_.accDtype;

                // NOTE (vs llama.cpp): ggml passes REQUIRE_FULL_SUBGROUPS_BIT
                // here; eva doesn't expose it (see ggmlCoopmatShape).
                pipeTile = pipelinePool().request("linear-cm", {
                    eva::constant_id<0>(t.wgSize),                       // wgSize (= local_size_x)
                    eva::constant_id<1>(t.sgSize),                       // sgSize (= device subgroup size)
                    eva::constant_id<2>(t.wgM),                          // wgM
                    eva::constant_id<3>(t.wgN),                          // wgN
                    eva::constant_id<4>(t.wgK),                          // wgK
                    eva::constant_id<5>(t.sgM),                          // sgM
                    eva::constant_id<6>(t.sgN),                          // sgN
                    eva::constant_id<7>(t.tM),                           // tM
                    eva::constant_id<8>(t.tN),                           // tN
                    eva::constant_id<9>(t.tK),                           // tK
                    eva::constant_id<11>(acc),                           // accType
                    eva::constant_id<12>(robust ? 1u : 0u),              // robustBufferAccess
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),            // typeI
                    eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),    // typeW
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u) // aligned
                }, {
                    .robustBufferAccess = robust,
                });
            }
            else   // scalar tile kernel
            {
                // Force the tile's assumed subgroup size onto the pipeline when the
                // device supports that size; otherwise leave the driver's choice.
                // (measured on Intel Arc iGPU: forcing 32 doubles the kernel time —
                //  the driver prefers a narrower SIMD width for this kernel)
                const bool forceSg = t.sgSize >= device().minSubgroupSize()
                                  && t.sgSize <= device().maxSubgroupSize()
                                  && false;

                // tN even: the kernel accumulates W-row pairs.
                VAI_ASSERT(t.tN % 2 == 0);

                pipeTile = pipelinePool().request("linear-tile", {
                    eva::constant_id<0>(t.wgSize),                                     // wgSize (= local_size_x)
                    eva::constant_id<1>(t.sgSize),                                     // sgSize
                    eva::constant_id<2>(t.wgM),                                        // wgM
                    eva::constant_id<3>(t.wgN),                                        // wgN
                    eva::constant_id<4>(t.wgK),                                        // wgK
                    eva::constant_id<5>(t.sgM),                                        // sgM
                    eva::constant_id<6>(t.sgN),                                        // sgN
                    eva::constant_id<7>(t.tM),                                         // tM
                    eva::constant_id<8>(t.tN),                                         // tN
                    eva::constant_id<9>(t.sgpMIter),                                   // sgpMIter
                    eva::constant_id<10>(t.sgpNIter),                                  // sgpNIter
                    eva::constant_id<11>(stageF32 ? Dtype::Fp32 : config_.accDtype),   // accType (fp32 over fp32 staging)
                    eva::constant_id<12>(robust ? 1u : 0u),                            // robustBufferAccess
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),                  // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                          // typeI
                    eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),                  // typeW
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)               // aligned
                }, {
                    .requiredSubgroupSize = forceSg ? t.sgSize : 0u,
                    .robustBufferAccess = robust,
                });
            }

            if (!descSetTile)
                descSetTile = gVaiDestSetPool(pipeTile.descSetLayout(0));
        }

        const PipeParamsReduce r = { .biasRepeat = biasElems, .residualRepeat = residualElems };
        if (needReducePass() && (!pipeReduce || r != pipeParamsReduce))
        {
            pipeParamsReduce = r;
            pipeReduce = pipelinePool().request("linear-reduce", {
                eva::constant_id<0>(config_.hasBias ? 1u : 0u),          // hasBias
                eva::constant_id<1>(biasElems),                          // bias repeat (0: covers O)
                eva::constant_id<2>(config_.acceptResidual ? 1u : 0u),   // hasResidual
                eva::constant_id<3>(residualElems),                      // residual repeat (0: covers O)
                eva::constant_id<4>((uint32_t) config_.residualOrder),   // residualOrder
                eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),        // typeO (O, P & residual)
                eva::constant_id<SPEC_ID_TYPE_W>(biasDtype()),           // typeW (bias)
                eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)   // epilogue
            });

            if (!descSetReduce)
                descSetReduce = gVaiDestSetPool(pipeReduce.descSetLayout(0));
        }

        // The partial buffer stays inside the tile branch: on the slab path
        // splitCount keeps the previous shape's value and must not be read.
        if (needReducePass())
            (*this)["partial"] = Tensor({splitCount, M, N}, config_.oDtype);

        // The int8 input: M * K bytes, then 8 bytes of (d, sum q) per 32
        // elements, 5/4 * M * K bytes in all (held as fp32 words).
        if (t.q8)
            (*this)["inQ8"] = Tensor({M * K / 16u * 5u}, Dtype::Fp32);
    }

    auto outShape = inShape;
    outShape.back() = N;
    (*this)["out0"] = Tensor(outShape, config_.oDtype);

}

inline void Linear::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto wBuf   = (*this)["weight"].buffer();

    uint32_t M = outerSize((*this)["in0"].shape());

    uint32_t N = config_.outDim;
    uint32_t K = config_.inDim;

    auto rBuf = config_.acceptResidual ? (*this)["residual"].buffer() : eva::Buffer();

    // Slab path: bias / epilogue / the residual are fused in the kernel, so
    // neither the split-K partial buffer nor the reduce pass runs. M is baked
    // into the pipeline as a spec constant, so the push constants are just
    // {K, N} (the tile kernel below uses {M, N, K, splitSize}).
    if (useSlabPath(M))
    {
        const uint32_t wgN = (config_.wDtype == Dtype::Fp16) ? 2u : 1u;   // fp16 W pairs N positions
        auto bBuf = findSlot("bias") ? (*this)["bias"].buffer() : eva::Buffer();
        struct { uint32_t K, N; float scalarScale; } slabConstants = { K, N, config_.scalarScale };

        // 1 workgroup = wgN N positions. An N past the device's X-grid limit folds
        // into z (ggml's groups_z split — the kernel re-linearizes with the z
        // term); only lmhead-sized N reaches it in practice.
        uint32_t gx = CEIL_DIV(N, wgN);
        uint32_t gz = 1;
        if (gx > device().maxComputeWorkGroupCount()[0])
        {
            gz = 64;   // = ggml groups_z
            gx = CEIL_DIV(gx, gz);
        }

        descSetSlab.write({ outBuf, inBuf, wBuf, bBuf, rBuf });   // 0 = O, 1 = I, 2 = W, 3 = bias, 4 = residual
        cmdBuff
            .bindPipeline(pipeSlab)
            .bindDescSets({descSetSlab})
            .setPushConstants(0, sizeof(slabConstants), &slabConstants)
            .dispatch(gx, 1, gz)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        return;
    }

    uint32_t constants[] = { M, N, K, splitSize };
    const TileGrid grid(M, N, pipeParamsTile.wgM, pipeParamsTile.wgN);

    // linear-q4-q8: quantize I first, then the tile kernel reads that instead.
    if (pipeParamsTile.q8)
    {
        auto qBuf = (*this)["inQ8"].buffer();
        struct { uint32_t M, K; } q8Constants = { M, K };
        constexpr uint32_t q8WgSize = 64;   // = local_size_x of linear-q8-input

        descSetQ8.write({ qBuf, inBuf });   // 0 = Q, 1 = I
        cmdBuff
            .bindPipeline(pipeQ8)
            .bindDescSets({descSetQ8})
            .setPushConstants(0, sizeof(q8Constants), &q8Constants)
            .dispatch(CEIL_DIV(M * (K / 32u), q8WgSize), 1, 1)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / qBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        inBuf = qBuf;
    }

    if (!needReducePass())
    {
        descSetTile.write({ outBuf, inBuf, wBuf });
        cmdBuff
            .bindPipeline(pipeTile)
            .bindDescSets({descSetTile})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(grid.tilesN, grid.tilesM, 1)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        return;
    }

    // Split path: the tile kernel writes partials P[splitCount, M, N], then the
    // reduce pass folds them and applies bias / epilogue / the residual.
    auto pBuf = (*this)["partial"].buffer();
    auto bBuf = findSlot("bias") ? (*this)["bias"].buffer() : eva::Buffer();
    descSetTile.write({ pBuf, inBuf, wBuf });
    descSetReduce.write({ outBuf, pBuf, bBuf, rBuf });   // 0 = O, 1 = P, 2 = bias, 3 = residual

    constexpr uint32_t reduceWgSize = 256;   // = local_size_x of linear-reduce
    constexpr uint32_t reduceBatch  = 4;     // elements folded per invocation
    struct { uint32_t MN, splitCount, N; float scalarScale; } reduceConstants = { M * N, splitCount, N, config_.scalarScale };

    cmdBuff
        .bindPipeline(pipeTile)
        .bindDescSets({descSetTile})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(grid.tilesN, grid.tilesM, splitCount)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / pBuf / eva::SYNC_SCOPE::COMPUTE_READ)
        .bindPipeline(pipeReduce)
        .bindDescSets({descSetReduce})
        .setPushConstants(0, sizeof(reduceConstants), &reduceConstants)
        .dispatch(CEIL_DIV(M * N, reduceWgSize * reduceBatch), 1, 1)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}


}   // namespace vai::node

#endif // VAI_NODES_LINEAR_H
