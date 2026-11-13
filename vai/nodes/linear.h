#ifndef VAI_NODES_LINEAR_H
#define VAI_NODES_LINEAR_H

// Linear / fully-connected layer, tile (scalar / coopmat) and slab kernels
// (ports of the ggml/llama.cpp Vulkan mul_mm / mul_mat_vec kernels).
#include "node-common.h"

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
    frame, chosen per device in selectLinearTile(): scalar
    (linear-tile, each thread accumulates its own outputs) and
    cooperative-matrix (linear-cm, subgroup MMA tiles; NVIDIA's Fill tiles
    run linear-cm-nv, which masks the K tail per k-block).
  - Intel (useIntelTile; every M, slab included): linear-tile-intel, a port of
    OpenVINO's fully_connected_gpu_bf_tiled, reading W packed once on the GPU
    (packWeightIntel); its tile choice is intelTileParams(), its split
    intelTileSplitCount().
  Everything below applies to every tile kernel:

  The tile params are selected per output shape in prepare(); when a shape
  change selects a different tile, the pipeline is re-requested from the pool.

  Split-K (splitSize option; 0 selects the vendor's rule per shape)
  accumulates partials P[splitCount, M, N] (oDtype) in a temporary buffer
  and folds them in a second pass (linear-reduce); bias / epilogue / the
  residual live in that reduce pass, so requesting them without splitting still
  runs the reduce as a plain epilogue pass (splitCount = 1).

  O = s · epilogue(I · Wᵀ + bias) + R      residualOrder AfterEpilogue
  O = s · epilogue(I · Wᵀ + bias + R)      residualOrder BeforeEpilogue

  s is Config::scalarScale (e.g. dt of an Euler step x + dt · v), 1 by default.

  "bias" (its own dtype) is a parameter and "residual" R (oDtype) an input. Each is
  [M, N] or rows that repeat over M: [N], or one image's position table under
  a batch. The "bias" slot always exists; the bias applies only when the
  weight loader filled it.
=============================================================================*/
class Linear : public vai::Node
{
    eva::ComputePipeline pipeTile;
    eva::ComputePipeline pipeSlab;
    eva::ComputePipeline pipeReduce;
    eva::DescriptorSet descSetTile;
    eva::DescriptorSet descSetSlab;
    eva::DescriptorSet descSetReduce;

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
        bool kTailMasked = false;   // coopmat: linear-cm-nv (K tail masked per k-block)
        vai::Dtype iDtype = vai::Dtype::None;

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

    // Intel scalar path (linear-tile-intel): tileB rows x 32 columns per subgroup.
    struct PipeParamsIntel
    {
        uint32_t tileB = 0;
        uint32_t bandTiles = 0;
        vai::Dtype accDtype = vai::Dtype::None;
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParamsIntel&) const = default;
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

        bool acceptResidual = false;  // input "residual"
        vai::ResidualOrder residualOrder = vai::ResidualOrder::AfterEpilogue;
        vai::Epilogue epilogue = vai::Epilogue::None;
        float scalarScale = 1.0f;   // s, right after the epilogue

        uint32_t splitSize = 0;   // split-K: K length per split (0: the vendor's count per shape)
    } config_;

    Linear(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The weight loader filled "bias"; fixed once the weights are up.
    bool hasBias() const { return (*this)["bias"].validShape(); }

    void setAccDtype(vai::Dtype dtype, vai::DtypeKey) { accDtype_ = dtype; }

private:
    vai::Dtype accDtype_ = vai::Dtype::None;   // the matmul accumulator; None: the weight's dtype

    // Intel path: the "weight" slot rewritten into linear-tile-intel's layout,
    // [ceil(N / 32), ceil(K / 2), 64], once (the first prepare).
    void packWeightIntel();
    bool weightPackedIntel() const;

    // The reduce pass runs when K is split and/or bias / epilogue / the residual
    // is requested (the tile kernel itself carries no epilogue).
    bool needReducePass() const
    {
        return splitCount > 1 || hasBias() || config_.acceptResidual || config_.epilogue != Epilogue::None ||
               config_.scalarScale != 1.0f;
    }

    // Small batches run the slab kernel instead of the tile one;
    // 8 = ggml mul_mat_vec_max_cols.
    bool useSlabPath(uint32_t M) const { return M <= 8u && !intel_; }

    bool intel_ = false;   // the Intel tile path (useIntelTile), set by prepare()
    PipeParamsIntel pipeParamsIntel;
    PipeParamsTile pipeParamsTile;
    PipeParamsSlab pipeParamsSlab;
    PipeParamsReduce pipeParamsReduce;
    uint32_t splitSize  = 0;  // K length per split for the current shape (= K when not splitting)
    uint32_t splitCount = 1;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Helpers: shared across the sections below and prepare()
/////////////////////////////////////////////////////////////////////////////////////////

// Shared memory of a coopmat tile: tileA/tileB + the fp32 store stage. NOTE (vs llama.cpp): ggml
// divides the stage term by warps — the shader declares cmStage[warps * tN * tM], so vai multiplies.
inline uint32_t coopmatTileShmem(const Linear::PipeParamsTile& t)
{
    return (t.wgN + t.wgM) * (t.wgK / 2 + 4) * 4 + (t.wgSize / t.sgSize) * t.tN * t.tM * 4;
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

    // Shared budget.
    return coopmatTileShmem(t) <= device.maxComputeSharedMemorySize();
}

// Whether the cooperative loaders (shared by both tile kernels) cover a tile:
// wgK 8-wide-load aligned, and whole wgM / wgN tiles per pass at the load
// code's elements per load call (1: 8-wide 8, 2: 4-wide 4, 3: 2-wide and
// 0: scalar 2).
inline bool tileLoadersFit(const Linear::PipeParamsTile& t, uint32_t loadCode)
{
    const uint32_t el = (loadCode == 1u) ? 8u : (loadCode == 2u) ? 4u : 2u;
    if (t.wgK % 8 != 0 || (t.wgSize * el) % t.wgK != 0)
        return false;
    const uint32_t rows = t.wgSize * el / t.wgK;
    return t.wgM % rows == 0 && t.wgN % rows == 0;
}


/////////////////////////////////////////////////////////////////////////////////////////
// ggml ports: llama.cpp's Vulkan mul_mm / mul_mat_vec tile, split and slab rules
/////////////////////////////////////////////////////////////////////////////////////////

// Vendor screen for the linear cm kernel (ggml_vk_khr_cooperative_matrix_support
// port): drivers that report the coopmat extension but aren't trusted to run
// it well.
inline bool ggmlCoopmatAllowedLinear(eva::Device device)
{
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::INTEL:
        return device.architectureID() == eva::Architecture::INTEL_XE2 ||
               device.architectureID() == eva::Architecture::INTEL_XE3 ||
               ((device.architectureID() == eva::Architecture::INTEL_XE ||
                 device.architectureID() == eva::Architecture::INTEL_XE_LPG) &&
                device.deviceType() == eva::DEVICE_TYPE::INTEGRATED_GPU &&
                device.driverID() == eva::DRIVER_ID::INTEL_PROPRIETARY_WINDOWS);
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

// ggml_vk_guess_matmul_pipeline port, the tile (vai naming: M = flattened
// batch, N = outDim): the device coopmat tile of the shape's tier where one is
// usable, degrading L -> M -> S, the gated scalar warptile otherwise. ggml's
// pick as such: AMD refines it in its section below, and NVIDIA falls back to
// it where no Fill tile fits (nvLinearTile).
inline Linear::PipeParamsTile ggmlMulmmWarptile(uint32_t M, uint32_t N, eva::Device device, bool stageF32 = false)
{
    const Linear::CmShape cm = ggmlCoopmatShape(device);
    if (cm.valid())
    {
        for (int tier = ggmlGuessTier(M, N, device, true); tier >= 0; tier--)
        {
            const Linear::PipeParamsTile t = ggmlCoopmatWarptile(tier, cm, device);
            if (coopmatTileUsable(t, device))
                return t;
        }
    }
    return ggmlScalarWarptile(ggmlGuessTier(M, N, device, false, stageF32), device);
}

// ggml's tile pipeline params for one shape: the warptile above, with the input dtype.
inline Linear::PipeParamsTile ggmlMulmmTileParams(uint32_t M, uint32_t N, Dtype iDtype, Dtype wDtype,
                                                  eva::Device device)
{
    Linear::PipeParamsTile t = ggmlMulmmWarptile(M, N, device, iDtype == Dtype::Fp32 && wDtype == Dtype::Fp32);
    t.iDtype = iDtype;
    return t;
}

// The K span each split covers, rounded so whole 256-blocks stream.
inline uint32_t ggmlSplitSize(uint32_t K, uint32_t splitCount)
{
    return ROUNDUP_POW2(CEIL_DIV(K, splitCount), 256u);
}

// ggml_vk_guess_split_k port (ggml m/n are vai N/M): split only an
// underfilled deep-K grid; shrink while the 256-rounded split size would
// leave the last split empty. ggml's guess as such: AMD's refined picks take
// AMD's own rule first.
inline uint32_t ggmlGuessSplitCount(uint32_t M, uint32_t N, uint32_t K,
                                    const Linear::PipeParamsTile& t, eva::Device device)
{
    const uint32_t cores = device.computeBlockCount();
    const uint32_t tiles = Linear::TileGrid(M, N, t.wgM, t.wgN).tiles();

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
// no PRE_TURING class). ggml's rule as such: AMD deviates in its section below.
inline uint32_t ggmlSlabWgSize(uint32_t K, uint32_t N, uint32_t sgSize, eva::Device device)
{
    const uint32_t vendor = device.vendorID();
    const bool nvPostTuring = vendor == eva::VENDOR_ID::NVIDIA && device.supportsCooperativeMatrix();
    if ((nvPostTuring || vendor == eva::VENDOR_ID::INTEL)
        && N <= 8192u && 1024u <= K )
        return sgSize * 4u;   // LARGE
    return sgSize;            // SUBGROUP
}


/////////////////////////////////////////////////////////////////////////////////////////
// NVIDIA
/////////////////////////////////////////////////////////////////////////////////////////

// The Fill selector: a coopmat tile and split count from M, N, Kp, the core count and the
// shared-memory budget — no vendor tables, no per-GPU constants. Its tiles run linear-cm-nv.comp
// (kTailMasked).

// Candidate at wgM x wgN: 4 subgroups as 2x2, 1x4 or 4x1, whichever minimizes sgM + sgN
// (16-multiples).
inline Linear::PipeParamsTile fillCandidate(const Linear::PipeParamsTile& base, uint32_t m, uint32_t n)
{
    Linear::PipeParamsTile c = base;
    c.wgM = m;
    c.wgN = n;
    uint32_t bestSum = ~0u;
    for (auto [rows, cols] : { std::pair{ 2u, 2u }, std::pair{ 1u, 4u }, std::pair{ 4u, 1u } })
    {
        const uint32_t sm = m / rows, sn = n / cols;
        if (m % rows || n % cols || sm % 16u || sn % 16u)
            continue;
        if (sm + sn < bestSum)
        {
            bestSum = sm + sn;
            c.sgM = sm;
            c.sgN = sn;
        }
    }
    return c;
}

// Workgroups of a tile, and the fill target (1.5 waves).
inline uint32_t fillTiles(uint32_t M, uint32_t N, const Linear::PipeParamsTile& c)
{
    return CEIL_DIV(M, c.wgM) * CEIL_DIV(N, c.wgN);
}

inline bool fillFilled(uint32_t tiles, uint32_t cores)
{
    return tiles * 2u >= cores * 3u;
}

// The M-tile ladder and its cost (tilesM x rows covered).
inline uint32_t fillNextM(uint32_t m)
{
    return m == 128u ? 96u : m == 96u ? 64u : 32u;
}

inline uint64_t fillMCost(uint32_t M, uint32_t m)
{
    const uint64_t tm = CEIL_DIV(M, m);
    return tm * tm * m;
}

// Split count: the smallest S (2..8) that fills the grid, each split >= minSpan deep and the last
// one non-empty. A grid below half a wave may split from K 1024, others from 2048.
// `settle` -> the largest allowed S.
inline uint32_t fillSplitFor(uint32_t Kp, uint32_t tiles, uint32_t cores, uint32_t minSpan, bool settle)
{
    if (Kp < (tiles * 2u < cores ? 1024u : 2048u))
        return 1u;
    uint32_t last = 1u;
    for (uint32_t s = 2; s <= 8u; ++s)
    {
        if (Kp / s < minSpan)
            break;
        if (ggmlSplitSize(Kp, s) * (s - 1u) >= Kp)
            continue;
        if (fillFilled(tiles * s, cores))
            return s;
        last = s;
    }
    return settle ? last : 1u;
}

// Half-height tile for a 128x128 grid below half a wave: S = cores / tiles (<= 8, >= 256 deep),
// then 64x128 or 64x64 if that grid x S reaches one WG per core.
inline bool fillHalfTile(uint32_t M, uint32_t N, uint32_t Kp, uint32_t cores, const Linear::PipeParamsTile& base,
                         eva::Device device, Linear::PipeParamsTile& t, uint32_t& splitCount)
{
    uint32_t s = std::min(8u, std::max(1u, cores / fillTiles(M, N, t)));
    while (s > 1u && (Kp / s < 256u || ggmlSplitSize(Kp, s) * (s - 1u) >= Kp))
        --s;
    if (s < 2u)
        return false;
    for (auto [hm, hn] : { std::pair{ 64u, 128u }, std::pair{ 64u, 64u } })
    {
        const Linear::PipeParamsTile d = fillCandidate(base, hm, hn);
        if (coopmatTileUsable(d, device) && fillTiles(M, N, d) * s >= cores)
        {
            t = d;
            splitCount = s;
            return true;
        }
    }
    return false;
}

// Resident-round split: the S (1..8, splits >= 512 deep) that minimizes rounds of resident workgroups
// x k-blocks per round plus the partials' traffic; only raises Fill's S, compute-bound GEMMs only.
inline uint32_t fillSplitRounds(uint32_t M, uint32_t N, uint32_t tiles, uint32_t Kp, uint64_t wBytes, uint32_t cores,
                                const Linear::PipeParamsTile& t, eva::Device device)
{
    if (Kp < 2048u)
        return 1u;
    if (2.0 * double(M) * double(N) * double(Kp) < double(wBytes) * 100.0)
        return 1u;   // W-bandwidth-bound regime
    // Resident workgroups per core: the per-workgroup shared limit is taken as half of a core's.
    const uint32_t r = std::max(1u, 2u * device.maxComputeSharedMemorySize() / coopmatTileShmem(t));
    const uint64_t slots = uint64_t(cores) * r;
    const double roundWork = double(slots) * 2.0 * t.wgM * t.wgN * t.wgK;
    const double partialFlopPerSplit = 8.0 * double(M) * double(N) * 100.0;
    const double weightStream = double(wBytes) * 100.0;
    uint32_t best = 1u;
    double bestCost = 0.0;
    for (uint32_t s = 1; s <= 8u; ++s)
    {
        if (Kp / s < 512u)
            break;
        if (s > 1u && ggmlSplitSize(Kp, s) * (s - 1u) >= Kp)
            continue;   // empty last split
        const uint64_t rounds = CEIL_DIV(uint64_t(tiles) * s, slots);
        const uint64_t blocks = CEIL_DIV(uint64_t(s > 1u ? ggmlSplitSize(Kp, s) : Kp), uint64_t(t.wgK));
        const double cost = std::max(double(rounds) * double(blocks) * roundWork, weightStream) +
                            (s > 1u ? s * partialFlopPerSplit : 0.0);
        if (s == 1u || cost < bestCost)
        {
            bestCost = cost;
            best = s;
        }
    }
    return best;
}

// The Fill selector on Kp (K rounded up to 64 when K % 32 != 0; the kernel masks the tail).
// splitCount 0: no fill tile fits the device (nvLinearTile then takes ggml's pick and split).
inline Linear::PipeParamsTile fillTileParams(uint32_t M, uint32_t N, uint32_t Kp, Dtype iDtype, Dtype wDtype,
                                             eva::Device device, uint32_t& splitCount)
{
    splitCount = 1;
    const Linear::CmShape cm = ggmlCoopmatShape(device);
    if (!cm.valid())   // no coopmat GEMM: ggml's pick
    {
        splitCount = 0;
        return {};
    }

    const uint32_t cores = std::max(device.computeBlockCount(), 1u);
    Linear::PipeParamsTile base = ggmlCoopmatWarptile(2, cm, device);   // ggml L: 128x128, 4 subgroups
    base.wgK = (Kp % 64u == 0u) ? 64u : 32u;

    uint32_t wgM = 128, wgN = 128;
    if (N <= 32u)
    {
        wgM = 32;
        wgN = 32;
    }
    else if (M <= 32u)
    {
        wgM = 32;
        wgN = 64;
    }
    else if (M <= 64u || N <= 64u)
    {
        wgM = 64;
        wgN = 64;
    }
    for (uint32_t m = wgM; m > 32u; m = fillNextM(m))
        if (fillMCost(M, fillNextM(m)) < fillMCost(M, wgM))
            wgM = fillNextM(m);
    Linear::PipeParamsTile t = fillCandidate(base, wgM, wgN);

    if (!fillFilled(fillTiles(M, N, t), cores))
    {
        splitCount = fillSplitFor(Kp, fillTiles(M, N, t), cores, 1024u, false);
        if (splitCount == 1u)
        {
            const uint32_t sMax = fillSplitFor(Kp, fillTiles(M, N, t), cores, 1024u, true);
            if (sMax >= 4u && fillTiles(M, N, t) * sMax * 4u >= cores * 3u)
                splitCount = sMax;
        }
        if (splitCount == 1u && t.wgM == 128u && t.wgN == 128u && Kp >= 2048u && fillTiles(M, N, t) * 2u < cores)
            fillHalfTile(M, N, Kp, cores, base, device, t, splitCount);
        if (splitCount == 1u)
        {
            while (!fillFilled(fillTiles(M, N, t), cores) && t.wgM > 32u)
            {
                const Linear::PipeParamsTile d = fillCandidate(base, fillNextM(t.wgM), t.wgN);
                if (!coopmatTileUsable(d, device))
                    break;
                t = d;
            }
            while (!fillFilled(fillTiles(M, N, t), cores) && t.wgN > 64u)
            {
                const Linear::PipeParamsTile d = fillCandidate(base, t.wgM, t.wgN == 128u ? 96u : 64u);
                if (!coopmatTileUsable(d, device))
                    break;
                t = d;
            }
            if (!fillFilled(fillTiles(M, N, t), cores))
            {
                splitCount = fillSplitFor(Kp, fillTiles(M, N, t), cores, 1024u, false);
                if (splitCount == 1u)
                    splitCount = fillSplitFor(Kp, fillTiles(M, N, t), cores, 256u, true);
            }
        }
    }

    if (!coopmatTileUsable(t, device))
    {
        t = fillCandidate(base, 128, 128);
        splitCount = 1;
    }

    if (t.wgM + t.wgN <= 96u && Kp % 128u == 0u            // k-block 128: a small tile
        && fillTiles(M, N, t) * splitCount * 2u < cores)   // below half a wave
    {
        Linear::PipeParamsTile d = t;
        d.wgK = 128u;
        if (coopmatTileUsable(d, device))
            t = d;
    }

    if (!coopmatTileUsable(t, device))
    {
        splitCount = 0;   // no fill tile on this device: ggml's pick and its split guess
        return {};
    }

    // Only ever raises Fill's S.
    const uint64_t wBytes = uint64_t(Kp) * N * (wDtype == Dtype::Fp16 ? 2u : 4u);
    splitCount = std::max(splitCount, fillSplitRounds(M, N, fillTiles(M, N, t), Kp, wBytes, cores, t, device));

    t.iDtype = iDtype;
    t.kTailMasked = true;   // linear-cm-nv
    return t;
}

// The tile and split count: the Fill selector on Kp, ggml's pick and split guess where no fill tile
// fits the device (no coopmat GEMM, or none within the shared-memory budget).
inline Linear::PipeParamsTile nvLinearTile(uint32_t M, uint32_t N, uint32_t K, Dtype iDtype, Dtype wDtype,
                                           eva::Device device, uint32_t& splitCount)
{
    const uint32_t Kp = (K % 32u != 0u) ? ROUNDUP_POW2(K, 64u) : K;
    Linear::PipeParamsTile t = fillTileParams(M, N, Kp, iDtype, wDtype, device, splitCount);
    if (splitCount == 0)
    {
        t = ggmlMulmmTileParams(M, N, iDtype, wDtype, device);
        splitCount = ggmlGuessSplitCount(M, N, K, t, device);
    }
    return t;
}

// linear-cm-nv's K tail: the selector works on Kp and the kernel masks the tail per k-block, so K
// need not be a multiple of wgK. Load-width code = the widest vector dividing both K and the split
// span (1: 8-wide, 2: 4-wide, 3: 2-wide), else 0 = the scalar masked path (odd K).
inline uint32_t nvCmLoadCode(uint32_t K, uint32_t splitSize)
{
    if (K % 8u == 0u && splitSize % 8u == 0u)
        return 1u;
    if (K % 4u == 0u && splitSize % 4u == 0u)
        return 2u;
    if (K % 2u == 0u && splitSize % 2u == 0u)
        return 3u;
    return 0u;
}


/////////////////////////////////////////////////////////////////////////////////////////
// AMD
/////////////////////////////////////////////////////////////////////////////////////////

// A vendor's refinement of ggml's coopmat pick is geometry-recognizable:
// ggml's coopmat tiles are all square k32, refined ones never are. The split
// rule below tells the two apart by it.
inline bool cmRefinedTile(const Linear::PipeParamsTile& t)
{
    return t.tK != 0u && (t.wgK != 32u || t.wgM != t.wgN);
}

// A vendor rule's geometry over ggml's coopmat pick `t`: it stands where the
// device runs it, ggml's pick otherwise.
inline Linear::PipeParamsTile refineCoopmatTile(const Linear::PipeParamsTile& t, uint32_t wgSize, uint32_t wgM,
                                                uint32_t wgN, uint32_t wgK, uint32_t sgM, uint32_t sgN,
                                                eva::Device device)
{
    Linear::PipeParamsTile tuned = t;
    tuned.wgSize = wgSize;
    tuned.wgM = wgM;
    tuned.wgN = wgN;
    tuned.wgK = wgK;
    tuned.sgM = sgM;
    tuned.sgN = sgN;
    // The split rule recognizes these picks by geometry: no rule may land on
    // a square k32 tile, which is what ggml's own picks all are.
    VAI_ASSERT(cmRefinedTile(tuned));
    return coopmatTileUsable(tuned, device) ? tuned : t;
}

// ggml's pick, refined per shape and CU count on the coopmat kernel at wave64.
// Wherever no rule applies, ggml's pick stands. Every rule wants the CU count
// and was measured on whole k-blocks (the picks step k32 / k64).
inline Linear::PipeParamsTile amdLinearTile(uint32_t M, uint32_t N, uint32_t K, Dtype iDtype, Dtype wDtype,
                                            eva::Device device)
{
    const Linear::PipeParamsTile t = ggmlMulmmTileParams(M, N, iDtype, wDtype, device);
    const uint32_t cores = device.computeBlockCount();
    if (t.tK == 0 || t.sgSize != 64 || cores == 0 || K % 64 != 0)
        return t;

    uint32_t wgM, wgN, wgK, sgM, sgN;
    uint32_t wgSize = 128;

    // UMA iGPUs: the weight stream dominates, so tiles go M-fat with
    // k64 blocks and never split. Discrete parts price weight re-reads
    // with VRAM + caches instead and stay on ggml's pick, as do low CU
    // counts.
    if (device.deviceType() != eva::DEVICE_TYPE::INTEGRATED_GPU || cores < 16u)
        return t;

    if (N <= 512u && M <= 256u)   // narrow layers, small M: at any CU count
    {
        wgM = 32;
        wgN = 32;
        wgK = 64;
        sgM = 32;
        sgN = 32;
        wgSize = 64;
    }
    else if (cores >= 32u)   // >= 32 CUs: one tile per layer width
    {
        if (N >= 8192u)   // widest layers: one fat M band of 32-wide
        {                 // columns, so the weight streams once per band
            wgK = (M >= 1024u) ? 32u : 64u;
            if (M <= 192u)
            {
                wgM = 192;
                wgN = 32;
                sgM = 96;
                sgN = 16;
                wgSize = 256;
            }
            else
            {
                wgM = 256;
                wgN = 32;
                sgM = 256;
                sgN = 16;
            }
        }
        else if (N <= 512u)   // narrow layers, bigger M
        {
            wgM = 128;
            wgN = 128;
            wgK = 64;
            sgM = 128;
            sgN = 16;
            wgSize = 512;
        }
        else   // mid widths (q/o, down-proj class)
        {
            // One M band pays only while the grid stays wide and deep;
            // narrow or shallow layers keep the small square.
            if (M <= 256u)
            {
                if (K >= 2048u && N >= 2048u)
                {
                    wgM = 192;
                    wgN = 64;
                    wgK = 64;
                    sgM = 96;
                    sgN = 16;
                    wgSize = 512;
                }
                else
                {
                    wgM = 64;
                    wgN = 64;
                    wgK = 64;
                    sgM = 64;
                    sgN = 16;
                    wgSize = 256;
                }
            }
            else if (M < 1024u)
            {
                wgM = 256;
                wgN = 128;
                wgK = 64;
                sgM = 256;
                sgN = 16;
                wgSize = 512;
            }
            else
            {
                wgM = 96;
                wgN = 128;
                wgK = 64;
                sgM = 96;
                sgN = 32;
                wgSize = 256;
            }
        }
    }
    else                 // < 32 CUs: one general tile, plus a
    {                    // narrow-N and a mid-M case
        if (N <= 512u)   // narrow layers, bigger M
        {
            wgM = 64;
            wgN = 128;
            wgK = 64;
            sgM = 32;
            sgN = 128;
        }
        else if (M > 256u && M < 2048u && (M < 1024u || K >= 8192u))
        {   // mid M rides a taller-N tile
            wgM = 128;
            wgN = 256;
            wgK = 64;
            sgM = 32;
            sgN = 256;
            wgSize = 256;
        }
        else   // the general tile
        {
            wgM = 192;
            wgN = 128;
            wgK = 64;
            sgM = 48;
            sgN = 128;
            wgSize = 256;
        }
    }

    return refineCoopmatTile(t, wgSize, wgM, wgN, wgK, sgM, sgN, device);
}

// A refined pick never splits (the picks exist only on the UMA iGPUs, where
// splits lose); ggml's own picks take ggml's guess.
inline uint32_t amdLinearSplitCount(uint32_t M, uint32_t N, uint32_t K, const Linear::PipeParamsTile& t,
                                    eva::Device device)
{
    if (cmRefinedTile(t))
        return 1;
    return ggmlGuessSplitCount(M, N, K, t, device);
}

// NOTE (vs llama.cpp): a deliberate deviation from ggml's slab rule — RDNA3+
// picks LARGE for K >= 1024, no N cap. Everything else is ggml's rule.
inline uint32_t amdLinearSlabWgSize(uint32_t K, uint32_t N, uint32_t sgSize, eva::Device device)
{
    if (device.architectureID() == eva::Architecture::AMD_POST_RDNA2 && 1024u <= K)
        return sgSize * 4u;   // LARGE (RDNA3+)
    return ggmlSlabWgSize(K, N, sgSize, device);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Intel
/////////////////////////////////////////////////////////////////////////////////////////

// linear-tile-intel runs at subgroup 16 and broadcasts fp16 across it
// (subgroup extended types).
inline bool useIntelTile(eva::Device device)
{
    return device.vendorID() == eva::VENDOR_ID::INTEL && !ggmlCoopmatShape(device).valid() &&
           device.minSubgroupSize() <= 16u && 16u <= device.maxSubgroupSize() && device.supportsSubgroupExtendedTypes();
}

// Workgroups the GPU keeps resident on the Intel tile path (one subgroup each):
// 16 XVEs x 8 threads per Xe-core (Xe-LPG).
inline uint32_t intelTileWave(eva::Device device)
{
    return device.computeBlockCount() * 128u;
}

inline uint32_t intelTileMaxSplits(uint32_t K)
{
    uint32_t s = 1;
    while (s < 16 && K / (s * 2) >= 64)
        s *= 2;
    return s;
}

// Intel tile choice (tests/linear-tune-intel, linear-tile-21), a wave being
// intelTileWave() workgroups:
//   1. rows per subgroup, most FMAs per load first, within the registers a
//      lane holds two sets of (32 fp16 / 16 fp32 accumulators); those padding
//      M by at most 25% (failing all, the least padding);
//   2. the first of at least 8 rows that fills a whole wave without split-K;
//      failing that, the first that split-K can bring to half a wave;
//   3. order: N-first over bands of 256 rows.
inline Linear::PipeParamsIntel intelTileParams(uint32_t M, uint32_t N, uint32_t K, Dtype iDtype, bool accF32,
                                               eva::Device device)
{
    static constexpr uint32_t kRowsF16[] = { 16, 8, 4, 2, 1 };
    static constexpr uint32_t kRowsF32[] = { 8, 4, 2, 1 };
    const uint32_t* rows = accF32 ? kRowsF32 : kRowsF16;
    const uint32_t count = accF32 ? std::size(kRowsF32) : std::size(kRowsF16);
    const uint32_t wave = intelTileWave(device);

    const auto pad = [&](uint32_t b) { return float(M) / float(ROUNDUP(M, b)); };
    float bestPad = 0.0f;
    for (uint32_t i = 0; i < count; i++)
        bestPad = std::max(bestPad, pad(rows[i]));
    const auto usable = [&](uint32_t b) { return pad(b) >= std::min(0.75f, bestPad - 1e-6f); };
    const auto wgs = [&](uint32_t b) { return CEIL_DIV(M, b) * CEIL_DIV(N, 32u); };

    uint32_t tileB = 0, last = 0;
    for (uint32_t i = 0; i < count && !tileB; i++)
        if (usable(rows[i]) && rows[i] >= 8 && wgs(rows[i]) >= wave)
            tileB = rows[i];
    for (uint32_t i = 0; i < count && !tileB; i++)
        if (usable(rows[i]))
        {
            last = rows[i];
            if (wgs(rows[i]) * intelTileMaxSplits(K) >= wave / 2)
                tileB = rows[i];
        }
    if (!tileB)
        tileB = last;

    return { .tileB = tileB,
             .bandTiles = std::min(CEIL_DIV(M, tileB), std::max(1u, 256u / tileB)),
             .accDtype = accF32 ? Dtype::Fp32 : Dtype::Fp16,
             .iDtype = iDtype };
}

// Split-K for an Intel tile: the least count reaching half a wave, then doubled
// while it cuts the whole waves per unit of work, ceil(wgs s / wave) / s, and
// K / s stays >= 1024.
inline uint32_t intelTileSplitCount(uint32_t M, uint32_t N, uint32_t K, const Linear::PipeParamsIntel& t,
                                    eva::Device device)
{
    const uint32_t wave = intelTileWave(device), wgs = CEIL_DIV(M, t.tileB) * CEIL_DIV(N, 32u);
    uint32_t s = 1;
    while (wgs * s < wave / 2 && s < intelTileMaxSplits(K))
        s *= 2;
    float perWork = float(CEIL_DIV(wgs * s, wave)) / float(s);
    for (uint32_t s2 = s * 2; s2 <= 16 && K / s2 >= 1024; s2 *= 2)
    {
        const float w = float(CEIL_DIV(wgs * s2, wave)) / float(s2);
        if (w < perWork - 1e-6f)
        {
            s = s2;
            perWork = w;
        }
    }
    return s;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Other vendors (Qualcomm, Apple, software, unknown): ggml's rules
/////////////////////////////////////////////////////////////////////////////////////////

// ggml's tile pick as such. Intel's parts without its own kernel (useIntelTile)
// take this pick too.
inline Linear::PipeParamsTile otherLinearTile(uint32_t M, uint32_t N, Dtype iDtype, Dtype wDtype, eva::Device device)
{
    return ggmlMulmmTileParams(M, N, iDtype, wDtype, device);
}

// ggml's split guess as such.
inline uint32_t otherLinearSplitCount(uint32_t M, uint32_t N, uint32_t K, const Linear::PipeParamsTile& t,
                                      eva::Device device)
{
    return ggmlGuessSplitCount(M, N, K, t, device);
}

// ggml's slab rule as such.
inline uint32_t otherLinearSlabWgSize(uint32_t K, uint32_t N, uint32_t sgSize, eva::Device device)
{
    return ggmlSlabWgSize(K, N, sgSize, device);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Tile, split count and slab size: the vendor decides
/////////////////////////////////////////////////////////////////////////////////////////

// The tile kernel's params for one shape and their split count (Intel's own
// kernel is prepare()'s branch, see useIntelTile). NVIDIA's Fill selector picks
// the two together; AMD and the others split the picked tile by their rule.
inline Linear::PipeParamsTile selectLinearTile(uint32_t M, uint32_t N, uint32_t K, Dtype iDtype, Dtype wDtype,
                                               eva::Device device, uint32_t& splitCount)
{
    Linear::PipeParamsTile t;
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::NVIDIA:
        return nvLinearTile(M, N, K, iDtype, wDtype, device, splitCount);
    case eva::VENDOR_ID::AMD:
        t = amdLinearTile(M, N, K, iDtype, wDtype, device);
        splitCount = amdLinearSplitCount(M, N, K, t, device);
        return t;
    default:
        t = otherLinearTile(M, N, iDtype, wDtype, device);
        splitCount = otherLinearSplitCount(M, N, K, t, device);
        return t;
    }
}

// The slab kernel's workgroup size (M <= 8): AMD deviates, the others keep ggml's rule.
inline uint32_t selectLinearSlabWgSize(uint32_t K, uint32_t N, uint32_t sgSize, eva::Device device)
{
    switch (device.vendorID())
    {
    case eva::VENDOR_ID::AMD:
        return amdLinearSlabWgSize(K, N, sgSize, device);
    default:
        return otherLinearSlabWgSize(K, N, sgSize, device);
    }
}


inline Linear::Linear(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    addSlot("partial", NodeSlot::temporary);   // assigned only when the reduce pass runs

    addSlot("bias", NodeSlot::persistent);
    if (config_.acceptResidual) addSlot("residual", NodeSlot::input);
}

inline bool Linear::weightPackedIntel() const
{
    return (*this)["weight"].isShapeOf(CEIL_DIV(config_.outDim, 32u), CEIL_DIV(config_.inDim, 2u), 64u);
}

// W [N, K] -> W'[n / 32, k / 2, n % 16, (k % 2) * 2 + (n % 32) / 16]
// (linear-pack-intel), into a buffer from the net's pool that the slot then
// holds: the [N, K] tensor goes back to the pool, where the next weight of its
// size can take it, unless something else still holds it.
inline void Linear::packWeightIntel()
{
    const uint32_t N = config_.outDim, K = config_.inDim;
    Tensor& wTen = (*this)["weight"];
    Tensor packed({ CEIL_DIV(N, 32u), CEIL_DIV(K, 2u), 64u }, wTen.dtype());
    packed.bindBuffer(bufferPool().requestBuffer(eva::BUFFER_USAGE::STORAGE_BUFFER | eva::BUFFER_USAGE::TRANSFER_DST,
                                                 eva::MEMORY_PROPERTY::DEVICE_LOCAL,
                                                 packed.numBytes(),
                                                 packed.numBytes()));

    // clang-format off
    eva::ComputePipeline pipe = pipelinePool().request("linear-pack-intel", {
        eva::constant_id<SPEC_ID_TYPE_W>(wTen.dtype())   // typeW
    });
    // clang-format on
    eva::DescriptorSet descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    descSet.write({ packed.buffer(), wTen.buffer() });   // 0 = W', 1 = W

    const uint32_t total = packed.numElements();
    const uint32_t groups = CEIL_DIV(total, 256u), maxX = device().maxComputeWorkGroupCount()[0];
    const uint32_t gy = CEIL_DIV(groups, maxX), gx = CEIL_DIV(groups, gy);
    const uint32_t constants[] = { N, K, total };

    auto cmd = device().newCommandBuffer(eva::queue_compute);
    cmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipe)
        .bindDescSets({ descSet })
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(gx, gy, 1)
        .end();
    device().queue() << cmd << eva::waiting;

    wTen = packed;
}

inline void Linear::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    const Tensor& wTen  = (*this)["weight"];

    uint32_t N = config_.outDim;
    uint32_t K = config_.inDim;
    VAI_ASSERT(inTen.shape().back() == K);

    const Dtype iDtype = inTen.dtype();
    const Dtype wDtype = wTen.dtype();
    const Dtype accDtype = (accDtype_ == Dtype::None) ? wDtype : accDtype_;

    const auto& inShape = inTen.shape();
    uint32_t M = outerSize(inShape);
    intel_ = useIntelTile(device());
    if (intel_ && wTen.isShapeOf(N, K))
        packWeightIntel();
    VAI_ASSERT(intel_ ? weightPackedIntel() : wTen.isShapeOf(N, K));

    // An operand block that covers O is 0, so the shader indexes it directly;
    // a shorter one ([N], or one image's rows) is its length, read modulo.
    auto repeatOf = [&](const Tensor& t)
    {
        const uint32_t outElems = M * N, elems = t.numElements();
        VAI_ASSERT(t.shape().back() == N && elems > 0 && outElems % elems == 0);
        return (elems == outElems) ? 0u : elems;
    };

    const Dtype bDtype = hasBias() ? (*this)["bias"].dtype() : Dtype::None;
    uint32_t biasElems = 0;
    if (hasBias())
    {
        biasElems = repeatOf((*this)["bias"]);
    }

    uint32_t residualElems = 0;
    if (config_.acceptResidual)
    {
        const Tensor& rTen = (*this)["residual"];
        VAI_ASSERT(rTen.dtype() == oDtype);
        residualElems = repeatOf(rTen);
    }

    if (useSlabPath(M))
    {
        // Slab policy (device / option-derived) — recomputed per prepare like
        // the tile params below; run() only re-derives wgN.
        const uint32_t sgSize = ggmlSlabSgSize(device());
        const uint32_t wgSize = selectLinearSlabWgSize(K, N, sgSize, device());
        const uint32_t wgN = (wDtype == Dtype::Fp16) ? 2u : 1u;   // fp16 W pairs N positions
        const bool     aligned = (K % 4 == 0);   // K_PER_ITER = 4 wide loads

        const PipeParamsSlab p = { .M = M, .biasRepeat = biasElems, .residualRepeat = residualElems, .iDtype = iDtype };
        if (!pipeSlab || p != pipeParamsSlab)
        {
            pipeParamsSlab = p;

            // clang-format off
            pipeSlab = pipelinePool().request("linear-slab", {
                eva::constant_id<0>(wgSize),                          // wgSize (= local_size_x)
                eva::constant_id<1>(sgSize),                          // sgSize (pinned subgroup size)
                eva::constant_id<2>(M),                               // M (flattened batch)
                eva::constant_id<3>(wgN),                             // wgN
                eva::constant_id<4>(hasBias() ? 1u : 0u),             // hasBias
                eva::constant_id<5>(biasElems),                       // bias repeat (0: covers O)
                eva::constant_id<6>(config_.acceptResidual ? 1u : 0u),   // hasResidual
                eva::constant_id<7>(residualElems),                   // residual repeat (0: covers O)
                eva::constant_id<8>((uint32_t) config_.residualOrder),   // residualOrder
                eva::constant_id<9>(bDtype),                          // typeB (bias)
                eva::constant_id<SPEC_ID_TYPE_O>(oDtype),     // typeO
                eva::constant_id<SPEC_ID_TYPE_I>(iDtype),             // typeI
                eva::constant_id<SPEC_ID_TYPE_W>(wDtype),             // typeW
                eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u), // aligned (K % 4 == 0)
                eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)   // epilogue
            }, {
                .requiredSubgroupSize = sgSize,
            });
            // clang-format on

            if (!descSetSlab)
                descSetSlab = gVaiDestSetPool(pipeSlab.descSetLayout(0));
        }
    }
    else
    {
        if (intel_)
        {
            const bool accF32 = (iDtype == Dtype::Fp32 && wDtype == Dtype::Fp32) || accDtype == Dtype::Fp32;
            const PipeParamsIntel pi = intelTileParams(M, N, K, iDtype, accF32, device());

            // Split-K: explicit splitSize, or the Intel rule; spans of whole 16-k trips.
            const uint32_t splits =
                (config_.splitSize > 0) ? CEIL_DIV(K, config_.splitSize) : intelTileSplitCount(M, N, K, pi, device());
            splitSize = (splits > 1) ? ROUNDUP(CEIL_DIV(K, splits), 16u) : K;
            splitCount = CEIL_DIV(K, splitSize);

            if (!pipeTile || pi != pipeParamsIntel)
            {
                pipeParamsIntel = pi;
                // clang-format off
                pipeTile = pipelinePool().request("linear-tile-intel", {
                    eva::constant_id<7>(pi.tileB),                          // tileB
                    eva::constant_id<10>(pi.bandTiles),                     // bandTiles
                    eva::constant_id<11>(pi.accDtype),                      // accType
                    eva::constant_id<SPEC_ID_TYPE_O>(oDtype),       // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),               // typeI
                    eva::constant_id<SPEC_ID_TYPE_W>(wDtype)        // typeW
                }, {
                    .requiredSubgroupSize = 16,
                });
                // clang-format on

                if (!descSetTile)
                    descSetTile = gVaiDestSetPool(pipeTile.descSetLayout(0));
            }
        }
        else
        {
            // fp32 x fp32 on the scalar tile kernel stages and accumulates fp32;
            // coopmat (like llama.cpp) and any fp16 operand keep fp16 staging.
            const bool stageF32 = iDtype == Dtype::Fp32 && wDtype == Dtype::Fp32;
            uint32_t pickedSplits = 1;
            const PipeParamsTile t = selectLinearTile(M, N, K, iDtype, wDtype, device(), pickedSplits);

            // Resolve split-K for this shape: explicit splitSize, or the
            // vendor's count when 0.
            if (config_.splitSize > 0)
            {
                VAI_ASSERT(config_.splitSize >= 32);
                splitSize = config_.splitSize;
                splitCount = CEIL_DIV(K, splitSize);
            }
            else
            {
                splitCount = pickedSplits;
                splitSize = (splitCount > 1) ? ggmlSplitSize(K, splitCount) : K;
            }

            if (!pipeTile || t != pipeParamsTile)
            {
                pipeParamsTile = t;

                // Whole k-blocks need K % wgK == 0 (and each split starting on a
                // k-block boundary); otherwise the kernel's scalar masked-load path
                // is selected. Heuristic splits are multiples of 256, so only wgK
                // among the key moves this. linear-cm-nv masks the K tail per
                // k-block instead and takes a load-width code (nvCmLoadCode).
                const bool aligned = (K % t.wgK == 0) && (splitSize % t.wgK == 0);
                const uint32_t loadCode = t.kTailMasked ? nvCmLoadCode(K, splitSize) : (aligned ? 1u : 0u);

                // Robustness folds the loaders' row-tail guards away (spec 12):
                // garbage from a robust OOB tail read only feeds outputs the
                // stores mask off (the K-tail masks stay — that axis is summed).
                // Guarded loads stall staging: 17-33% measured with device-local
                // input (pinned inputs invert that — see tests/linear-cm).
                // NOTE (vs llama.cpp): ggml relies on core robustBufferAccess,
                // always on; vai needs VK_EXT_pipeline_robustness, else guards stay.
                const bool robust = device().supportsPipelineRobustness();

                // Loader constraints, common to both tile kernels (they share
                // the cooperative loaders).
                VAI_ASSERT(tileLoadersFit(t, loadCode));

                if (t.tK != 0)   // coopmat kernel
                {
                    // fp16 accumulation only when the locked shape reports it;
                    // otherwise promote to fp32 (ggml falls back to its f32acc
                    // pipeline the same way).
                    const Dtype acc =
                        (accDtype == Dtype::Fp16 && !ggmlCoopmatShape(device()).accF16) ? Dtype::Fp32 : accDtype;

                    // NOTE (vs llama.cpp): ggml passes REQUIRE_FULL_SUBGROUPS_BIT
                    // here; eva doesn't expose it (see ggmlCoopmatShape).
                    // clang-format off
                    pipeTile = pipelinePool().request(t.kTailMasked ? "linear-cm-nv" : "linear-cm", {
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
                        eva::constant_id<SPEC_ID_TYPE_O>(oDtype),    // typeO
                        eva::constant_id<SPEC_ID_TYPE_I>(iDtype),            // typeI
                        eva::constant_id<SPEC_ID_TYPE_W>(wDtype),    // typeW
                        eva::constant_id<SPEC_ID_ALIGNED>(loadCode)          // aligned (linear-cm-nv: load-width code)
                    }, {
                        .robustBufferAccess = robust,
                    });
                    // clang-format on
                }
                else   // scalar tile kernel
                {
                    // Force the tile's assumed subgroup size onto the pipeline when the
                    // device supports that size; otherwise leave the driver's choice.
                    // (measured on Intel Arc iGPU: forcing 32 doubles the kernel time —
                    //  the driver prefers a narrower SIMD width for this kernel)
                    const bool forceSg =
                        t.sgSize >= device().minSubgroupSize() && t.sgSize <= device().maxSubgroupSize() && false;

                    // tN even: the kernel accumulates W-row pairs.
                    VAI_ASSERT(t.tN % 2 == 0);

                    // clang-format off
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
                        eva::constant_id<11>(stageF32 ? Dtype::Fp32 : accDtype),   // accType (fp32 over fp32 staging)
                        eva::constant_id<12>(robust ? 1u : 0u),                            // robustBufferAccess
                        eva::constant_id<SPEC_ID_TYPE_O>(oDtype),                  // typeO
                        eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                          // typeI
                        eva::constant_id<SPEC_ID_TYPE_W>(wDtype),                  // typeW
                        eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)               // aligned
                    }, {
                        .requiredSubgroupSize = forceSg ? t.sgSize : 0u,
                        .robustBufferAccess = robust,
                    });
                    // clang-format on
                }

                if (!descSetTile)
                    descSetTile = gVaiDestSetPool(pipeTile.descSetLayout(0));
            }
        }

        const PipeParamsReduce r = { .biasRepeat = biasElems, .residualRepeat = residualElems };
        if (needReducePass() && (!pipeReduce || r != pipeParamsReduce))
        {
            pipeParamsReduce = r;
            // clang-format off
            pipeReduce = pipelinePool().request("linear-reduce", {
                eva::constant_id<0>(hasBias() ? 1u : 0u),                // hasBias
                eva::constant_id<1>(biasElems),                          // bias repeat (0: covers O)
                eva::constant_id<2>(config_.acceptResidual ? 1u : 0u),   // hasResidual
                eva::constant_id<3>(residualElems),                      // residual repeat (0: covers O)
                eva::constant_id<4>((uint32_t) config_.residualOrder),   // residualOrder
                eva::constant_id<5>(bDtype),                             // typeB (bias)
                eva::constant_id<SPEC_ID_TYPE_O>(oDtype),        // typeO (O, P & residual)
                eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)   // epilogue
            });
            // clang-format on

            if (!descSetReduce)
                descSetReduce = gVaiDestSetPool(pipeReduce.descSetLayout(0));
        }

        // The partial buffer stays inside the tile branch: on the slab path
        // splitCount keeps the previous shape's value and must not be read.
        if (needReducePass())
            (*this)["partial"] = Tensor({ splitCount, M, N }, oDtype);
    }

    auto outShape = inShape;
    outShape.back() = N;
    (*this)["out0"] = Tensor(outShape, oDtype);
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
        const uint32_t wgN = ((*this)["weight"].dtype() == Dtype::Fp16) ? 2u : 1u;   // fp16 W pairs N positions
        auto bBuf = hasBias() ? (*this)["bias"].buffer() : eva::Buffer();
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
    uint32_t gx, gy;
    if (intel_)
    {
        // One flat tile index (linear-tile-intel), folded into y past the x limit.
        const uint32_t bands = CEIL_DIV(CEIL_DIV(M, pipeParamsIntel.tileB), pipeParamsIntel.bandTiles);
        const uint32_t total = CEIL_DIV(N, 32u) * bands * pipeParamsIntel.bandTiles;
        gy = (total <= device().maxComputeWorkGroupCount()[0])
                 ? 1u
                 : CEIL_DIV(total, CEIL_DIV(total, device().maxComputeWorkGroupCount()[1]));
        gx = CEIL_DIV(total, gy);
    }
    else
    {
        const TileGrid grid(M, N, pipeParamsTile.wgM, pipeParamsTile.wgN);
        gx = grid.tilesN;
        gy = grid.tilesM;
    }

    if (!needReducePass())
    {
        descSetTile.write({ outBuf, inBuf, wBuf });
        cmdBuff.bindPipeline(pipeTile)
            .bindDescSets({ descSetTile })
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(gx, gy, 1)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
        return;
    }

    // Split path: the tile kernel writes partials P[splitCount, M, N], then the
    // reduce pass folds them and applies bias / epilogue / the residual.
    auto pBuf = (*this)["partial"].buffer();
    auto bBuf = hasBias() ? (*this)["bias"].buffer() : eva::Buffer();
    descSetTile.write({ pBuf, inBuf, wBuf });
    descSetReduce.write({ outBuf, pBuf, bBuf, rBuf });   // 0 = O, 1 = P, 2 = bias, 3 = residual

    constexpr uint32_t reduceWgSize = 256;   // = local_size_x of linear-reduce
    constexpr uint32_t reduceBatch  = 4;     // elements folded per invocation
    struct { uint32_t MN, splitCount, N; float scalarScale; } reduceConstants = { M * N, splitCount, N, config_.scalarScale };

    cmdBuff.bindPipeline(pipeTile)
        .bindDescSets({ descSetTile })
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(gx, gy, splitCount)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / pBuf / eva::SYNC_SCOPE::COMPUTE_READ)
        .bindPipeline(pipeReduce)
        .bindDescSets({ descSetReduce })
        .setPushConstants(0, sizeof(reduceConstants), &reduceConstants)
        .dispatch(CEIL_DIV(M * N, reduceWgSize * reduceBatch), 1, 1)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}


}   // namespace vai::node

#endif // VAI_NODES_LINEAR_H
