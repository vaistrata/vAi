#ifndef VAI_NODES_CONVOLUTION_H
#define VAI_NODES_CONVOLUTION_H

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "node-common.h"

namespace vai::node {
/////////////////////////////////////////////////////////////////////////////////////////
// Common utilities, device capabilities, and workload
/////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t scaleWorkSupplyThreshold(uint32_t baseWorkgroups, uint32_t computeBlockCount)
{
    // computeBlock : (NVIDIA : SM), (AMD : CU)
    constexpr uint32_t referenceComputeBlocks = 64u;
    const uint32_t computeBlocks = computeBlockCount == 0u ? referenceComputeBlocks : computeBlockCount;

    return static_cast<uint32_t>((uint64_t(baseWorkgroups) * computeBlocks + referenceComputeBlocks - 1u) /
                                 referenceComputeBlocks);
}

enum class ConvAlgorithm : uint8_t {
    None,
    Direct,
    Implicit,
    Winograd,
};

enum class ConvolutionFusion : uint8_t {
    Bias,  // unscaled output; bias is optional
    Scale, // scaled output; bias is optional
};

inline bool supportsCMShape(const eva::Device& device, Dtype wDtype, Dtype accDtype)
{
    using CT = eva::COMPONENT_TYPE;

    if (wDtype != Dtype::Fp16 || (accDtype != Dtype::Fp16 && accDtype != Dtype::Fp32))
        return false;

    const CT accType = (accDtype == Dtype::Fp16) ? CT::FLOAT16 : CT::FLOAT32;

    for (const auto& shape : device.cooperativeMatrixProperties())
    {
        if (shape.aType != CT::FLOAT16 || shape.bType != CT::FLOAT16 ||
            shape.cType != accType ||
            shape.resultType != accType ||
            shape.scope != eva::SCOPE::SUBGROUP ||
            shape.saturatingAccumulation)
            continue;
        if (shape.M == 16u && shape.N == 16u && shape.K == 16u)
            return true;
    }
    return false;
}

inline bool hasCooperativeMatrixUnit(uint32_t vendorId, eva::Architecture architecture)
{
    switch (vendorId)
    {
    case eva::VENDOR_ID::NVIDIA:
        return architecture == eva::Architecture::NVIDIA_TURING ||
            architecture == eva::Architecture::NVIDIA_POST_TURING;
    case eva::VENDOR_ID::AMD:
        return architecture == eva::Architecture::AMD_POST_RDNA2;
    case eva::VENDOR_ID::INTEL:
        return architecture == eva::Architecture::INTEL_XE || architecture == eva::Architecture::INTEL_XE_LPG ||
               architecture == eva::Architecture::INTEL_XE_HPG || architecture == eva::Architecture::INTEL_XE2 ||
               architecture == eva::Architecture::INTEL_XE3;
    default:
        return false;
    }
}

constexpr uint32_t kAMDStrixPointDeviceId = 0x150eu;
constexpr uint32_t kAMDStrixHaloDeviceId = 0x1586u;
constexpr uint32_t kAMDNavi48DeviceId = 0x7550u;

enum class AMDArchitecture : uint8_t {
    Unknown,
    RDNA3,
    RDNA4,
};

// Refines EVA's AMD_POST_RDNA2 class for convolution tuning policy.
inline AMDArchitecture classifyAMDArchitecture(uint32_t vendorId, uint32_t deviceId)
{
    if (vendorId != eva::VENDOR_ID::AMD)
        return AMDArchitecture::Unknown;

    switch (deviceId)
    {
    case kAMDStrixPointDeviceId: // Strix Point
    case kAMDStrixHaloDeviceId: // Strix Halo
        return AMDArchitecture::RDNA3;
    case kAMDNavi48DeviceId: // Navi 48
        return AMDArchitecture::RDNA4;
    default:
        return AMDArchitecture::Unknown;
    }
}

struct ConvolutionCaps {
    bool hasCM = false;        // hardware-level matrix-core capability
    bool accFP16 = false;      // cooperative matrix accumulator support for fp16
};

inline bool subgroupSizeInRange(const eva::Device& device, uint32_t size)
{
    return device.minSubgroupSize() <= size && size <= device.maxSubgroupSize();
}

inline bool enableCM(const eva::Device& device, const ConvolutionCaps& caps)
{
    return device.supportsCooperativeMatrix() && caps.hasCM;
}

// Selects the accumulator inside an already established FP16 CM branch.
inline Dtype selectAccType(const eva::Device& device, const ConvolutionCaps& caps)
{
    if (device.vendorID() == eva::VENDOR_ID::AMD)
        return Dtype::Fp32;
    return caps.accFP16 ? Dtype::Fp16 : Dtype::Fp32;
}

struct Workload {
    uint32_t ci = 0u;
    uint32_t co = 0u;
    uint32_t kernel = 0u;
    uint32_t stride = 0u;
    uint32_t hi = 0u;
    uint32_t wi = 0u;
    uint32_t ho = 0u;
    uint32_t wo = 0u;
    uint32_t batch = 1u;   // images in in0's leading axes, stacked on the dispatch z axis
    Dtype iDtype = Dtype::Fp32;
    Dtype oDtype = Dtype::Fp32;
    Dtype wDtype = Dtype::None;
    ConvolutionFusion fusion = ConvolutionFusion::Bias;
    bool residual = false;

    constexpr uint32_t spatialSize() const { return ho * wo; }
    constexpr uint32_t winogradTileCount() const  // winograd tile count for 3x3 kernel, stride=1, padding=1
    {
        return CEIL_DIV(wo, 4u) * CEIL_DIV(ho, 4u);
    }
};

// The residual a pass adds to its output. Off for a pass that writes split-K
// partials: the reduce that finishes them adds it.
struct ResidualFusion
{
    bool enabled = false;
    uint32_t order = 0u;   // ResidualOrder
};

struct ShapeSpec
{
    uint32_t hi = 0u;
    uint32_t wi = 0u;
    uint32_t ho = 0u;
    uint32_t wo = 0u;
    uint32_t ci = 0u;
    uint32_t co = 0u;

    static constexpr ShapeSpec fromWorkload(const Workload& workload)
    {
        return {
            workload.hi,
            workload.wi,
            workload.ho,
            workload.wo,
            workload.ci,
            workload.co,
        };
    }

    bool empty() const
    {
        return hi == 0u && wi == 0u && ho == 0u && wo == 0u && ci == 0u && co == 0u;
    }

    bool matchSpatial(const Workload& workload) const
    {
        return hi == workload.hi && wi == workload.wi && ho == workload.ho && wo == workload.wo;
    }

    bool matchShape(const Workload& workload) const
    {
        return matchSpatial(workload) && ci == workload.ci && co == workload.co;
    }

    auto operator<=>(const ShapeSpec&) const = default;
};

struct ConvDispatchFeatures
{
    uint32_t R = 0u;
    uint32_t M = 0u;
    uint32_t P = 0u;
    uint64_t G = 0u;
};

inline ConvDispatchFeatures makeConvDispatchFeatures(
    uint32_t ci,
    uint32_t reductionTileElements,
    uint32_t co,
    uint32_t outputChannelTileElements,
    uint32_t n,
    uint32_t spatialTileElements)
{
    VAI_ASSERT(ci != 0u && reductionTileElements != 0u &&
               co != 0u && outputChannelTileElements != 0u &&
               n != 0u && spatialTileElements != 0u);
    const uint32_t r = CEIL_DIV(ci, reductionTileElements);
    const uint32_t m = CEIL_DIV(co, outputChannelTileElements);
    const uint32_t p = CEIL_DIV(n, spatialTileElements);
    return {r, m, p, uint64_t(m) * p};
}

/////////////////////////////////////////////////////////////////////////////////////////
// Common command-recording primitives
/////////////////////////////////////////////////////////////////////////////////////////

struct PushConstants {
    uint32_t hi;
    uint32_t wi;
    uint32_t ho;
    uint32_t wo;
    uint32_t ci;
    uint32_t co;
    uint32_t n;
    uint32_t splitCount;
    uint32_t kernel;
    uint32_t stride;
    uint32_t batch;
};

static_assert(std::is_standard_layout_v<PushConstants>);
static_assert(std::is_trivially_copyable_v<PushConstants>);
static_assert(sizeof(PushConstants) == 44u);
static_assert(alignof(PushConstants) == 4u);
static_assert(offsetof(PushConstants, hi) == 0u);
static_assert(offsetof(PushConstants, wi) == 4u);
static_assert(offsetof(PushConstants, ho) == 8u);
static_assert(offsetof(PushConstants, wo) == 12u);
static_assert(offsetof(PushConstants, ci) == 16u);
static_assert(offsetof(PushConstants, co) == 20u);
static_assert(offsetof(PushConstants, n) == 24u);
static_assert(offsetof(PushConstants, splitCount) == 28u);
static_assert(offsetof(PushConstants, kernel) == 32u);
static_assert(offsetof(PushConstants, stride) == 36u);
static_assert(offsetof(PushConstants, batch) == 40u);

constexpr PushConstants makePushConstants(const Workload& workload, uint32_t n = 0u, uint32_t splitCount = 0u)
{
    return {
        workload.hi,
        workload.wi,
        workload.ho,
        workload.wo,
        workload.ci,
        workload.co,
        n == 0u ? workload.spatialSize() : n,   // winograd  n = CEIL_DIV(Ho, 4) * CEIL_DIV(Wo, 4)
        splitCount,
        workload.kernel,
        workload.stride,
        workload.batch,
    };
}

struct PipelineBinding
{
    eva::ComputePipeline pipeline;
    eva::DescriptorSet descriptorSet;
};

inline PipelineBinding makePipelineBinding(eva::ComputePipeline pipeline)
{
    return {pipeline, gVaiDestSetPool(pipeline.descSetLayout(0))};
}

// groupCountZ is one image's; the images stack on z, outermost.
inline void recordDispatch(
    eva::CommandBuffer cmdBuff,
    const eva::ComputePipeline& pipeline,
    const eva::DescriptorSet& descriptorSet,
    const PushConstants& constants,
    uint32_t groupCountX,
    uint32_t groupCountY = 1u,
    uint32_t groupCountZ = 1u)
{
    cmdBuff.bindPipeline(pipeline)
        .bindDescSets({descriptorSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(groupCountX, groupCountY, groupCountZ * constants.batch);
}

inline void recordDispatch(
    eva::CommandBuffer cmdBuff,
    const PipelineBinding& binding,
    const PushConstants& constants,
    uint32_t groupCountX,
    uint32_t groupCountY = 1u,
    uint32_t groupCountZ = 1u)
{
    recordDispatch(
        cmdBuff, binding.pipeline, binding.descriptorSet, constants,
        groupCountX, groupCountY, groupCountZ);
}

struct AffineBindings
{
    eva::Descriptor bias;
    eva::Descriptor scale;
};

inline AffineBindings convAffineBindings(Node& node, bool hasBias, bool hasScale)
{
    // Bind an existing buffer for unused slots; specialization removes every read.
    const eva::Descriptor bias = hasBias ? node["bias"].buffer() : node["weight"].buffer();
    const eva::Descriptor scale = hasScale ? eva::Descriptor(node["scale"].buffer()) : bias;
    return {bias, scale};
}

inline bool ensureWeightTransform(
    eva::Device device,
    eva::CommandBuffer cmdBuff,
    PipelineBinding& binding,
    const eva::BufferRange& srcWeight,
    eva::Buffer& transformedWeight,
    const PushConstants& constants,
    size_t byteSize,
    uint32_t workItems,
    bool& transformed)
{
    if (transformed)
        return false;

    transformedWeight = device.createBuffer({
        .size = byteSize,
        .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::DEVICE_LOCAL,
    });
    binding.descriptorSet.write({transformedWeight, srcWeight});
    cmdBuff.bindPipeline(binding.pipeline)
        .bindDescSets({binding.descriptorSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(workItems)
        .barrier(
            eva::SYNC_SCOPE::COMPUTE_WRITE / transformedWeight /
            eva::SYNC_SCOPE::COMPUTE_READ);
    transformed = true;
    return true;
}

// Waits on the partial buffer, then reduces the split-K partials into the
// output, workItems per image. The caller keeps its own barrier on the reduce output, whose
// placement differs per path.
inline void recordConvSplitKReduce(eva::CommandBuffer cmdBuff, PipelineBinding& reduce, const eva::Descriptor& output,
                                   Tensor& partial, const AffineBindings& affine, const eva::Descriptor& residual,
                                   const PushConstants& reduceConstants, uint32_t workItems)
{
    cmdBuff.barrier(
        eva::SYNC_SCOPE::COMPUTE_WRITE / partial.buffer() /
        eva::SYNC_SCOPE::COMPUTE_READ);
    reduce.descriptorSet.write({
        output,
        partial.buffer(),
        affine.bias,
        affine.scale,
        residual,
    });
    cmdBuff.bindPipeline(reduce.pipeline)
        .bindDescSets({reduce.descriptorSet})
        .setPushConstants(0, sizeof(reduceConstants), &reduceConstants)
        .dispatch2(workItems, 1u, reduceConstants.batch);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Convolution plans
/////////////////////////////////////////////////////////////////////////////////////////

struct K1S1Plan {
    uint32_t sgSize = 32u;
    uint32_t wgM = 16u;
    uint32_t wgN = 16u;
    uint32_t wgK = 16u;
    uint32_t tM = 16u;
    uint32_t sgM = 0u; // M tile dimension computed by each subgroup.
    uint32_t sgN = 0u; // N tile dimension computed by each subgroup.
    uint32_t accStoreBatchSize = 1u;
    uint32_t loadMode = 0u;
    bool alignM = false;
    bool alignN = false;
    bool alignK = false;
    bool storeMode = false;
    uint32_t accType = 0u;
    bool reorder = false;
    uint32_t splitK = 1u;
    uint32_t swapXY = 0u;
    bool directOutput = false;
    uint32_t ciSpec = 0u;
    uint32_t nSpec = 0u;

    auto operator<=>(const K1S1Plan&) const = default;
};

constexpr uint32_t kK3S1TailWgN = 16u;

struct K3S1Plan {
    ConvAlgorithm algorithm = ConvAlgorithm::None;
    uint32_t wgSize = 0u;
    uint32_t wgM = 0u;
    uint32_t wgN = 0u;
    uint32_t wgK = 0u;
    uint32_t sgSize = 0u;
    uint32_t sgM = 0u; // M tile dimension computed by each subgroup.
    uint32_t sgN = 0u; // N tile dimension computed by each subgroup.
    uint32_t tM = 0u;
    uint32_t tN = 0u;
    uint32_t panelK = 0u;
    uint32_t accType = 1u;
    uint32_t splitK = 1u;
    uint32_t tailNOffset = 0u;
    uint32_t pad = 0u;
    ShapeSpec shape;
    bool alignM = false;
    bool alignN = false;
    bool alignK = false;
    bool loadMode = false;
    bool storeMode = false;
    bool reorder = false;
    bool inputReorder = false;
    bool swapXY = false;
    bool outputStoreMode = false;
    bool flatSpatial = false;

    bool valid() const { return algorithm != ConvAlgorithm::None; }
    bool hasTail() const { return tailNOffset != 0u; }
    auto operator<=>(const K3S1Plan&) const = default;
};

struct K3S2Plan {
    uint32_t splitK = 1u;
    uint32_t wgSize = 0u;
    uint32_t wgM = 0u;
    uint32_t wgN = 0u;
    uint32_t wgK = 0u;
    uint32_t tM = 0u;
    uint32_t sgM = 0u; // M tile dimension computed by each subgroup.
    uint32_t sgN = 0u; // N tile dimension computed by each subgroup.
    ShapeSpec shape;
    uint32_t accType = 1u;
    bool packedWeights = false;
    bool directOutput = false;
    bool alignM = false;
    bool alignN = false;
    bool alignK = false;

    bool valid() const { return wgSize != 0u && wgM != 0u && wgN != 0u; }
    bool cooperativeMatrix() const { return sgM != 0u && sgN != 0u; }
    bool firstLayer() const { return !cooperativeMatrix() && wgK == 27u; }
    bool split() const { return splitK > 1u; }

    auto operator<=>(const K3S2Plan&) const = default;
};

struct ImplicitGemmPlan
{
    uint32_t wgSize = 128u;
    uint32_t wgM = 64u;
    uint32_t wgN = 32u;
    uint32_t sgM = 0u; // M tile dimension computed by each subgroup.
    uint32_t sgN = 0u; // N tile dimension computed by each subgroup.
    uint32_t wgK = 8u;
    uint32_t tM = 4u;
    uint32_t panelK = 32u;
    uint32_t accType = 1u;
    bool cooperativeMatrix = false;
    bool structuredInputLoad = false;
    bool alignM = false;
    bool alignN = false;
    bool alignK = false;
    bool padA = true;
    bool wideFp32Tile = false;

    auto operator<=>(const ImplicitGemmPlan&) const = default;
};

struct ConvPlan
{
    enum class Path : uint8_t
    {
        General,
        K1S1,
        K3S1,
        K3S2,
    };

    Path path = Path::General;
    uint32_t vendorId = 0u;
    Dtype iDtype = Dtype::Fp32;
    Dtype oDtype = Dtype::Fp32;
    Dtype wDtype = Dtype::None;
    ConvolutionFusion fusion = ConvolutionFusion::Bias;
    K1S1Plan k1s1;
    K3S1Plan k3s1;
    K3S2Plan k3s2;
    ImplicitGemmPlan implicit;
    uint32_t requiredSubgroupSize = 0u;
    uint32_t splitRequiredSubgroupSize = 0u;
    bool useCM = false;

    ConvAlgorithm algorithm() const noexcept
    {
        switch (path)
        {
        case Path::K1S1:
        case Path::K3S2:
            return ConvAlgorithm::Direct;

        case Path::K3S1:
            return k3s1.algorithm;

        case Path::General:
            return ConvAlgorithm::Implicit;
        }

        return ConvAlgorithm::None;
    }

    auto operator<=>(const ConvPlan&) const = default;
};

/////////////////////////////////////////////////////////////////////////////////////////
// Common utilities, device capabilities, and workload: selection tables
/////////////////////////////////////////////////////////////////////////////////////////

struct K1S1CMExactRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype weightDtype;
    ConvolutionFusion fusion;
    ShapeSpec shape;
    K1S1Plan plan;
};

constexpr std::array<K1S1CMExactRule, 21> kK1S1CMExactTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Bias,
     {80u, 80u, 80u, 80u, 80u, 80u},
     {.wgM = 16u, .wgN = 32u, .sgM = 16u, .sgN = 16u,
      .accStoreBatchSize = 1u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Bias,
     {80u, 80u, 80u, 80u, 64u, 64u},
     {.wgM = 64u, .wgN = 32u, .sgM = 32u, .sgN = 16u,
      .accStoreBatchSize = 2u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Bias,
     {40u, 40u, 40u, 40u, 64u, 64u},
     {.wgM = 32u, .wgN = 32u, .sgM = 32u, .sgN = 16u,
      .accStoreBatchSize = 2u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Bias,
     {40u, 40u, 40u, 40u, 80u, 80u},
     {.wgM = 16u, .wgN = 32u, .sgM = 16u, .sgN = 16u,
      .accStoreBatchSize = 1u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 1024u, 256u},
     {.wgM = 64u, .wgN = 64u, .sgM = 64u, .sgN = 32u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 768u, 256u},
     {.wgM = 128u, .wgN = 32u, .sgM = 64u, .sgN = 32u,
      .accStoreBatchSize = 8u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 1280u, 512u},
     {.wgM = 128u, .wgN = 64u, .sgM = 32u, .sgN = 32u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 2048u, 512u},
     {.wgM = 128u, .wgN = 64u, .sgM = 32u, .sgN = 32u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {160u, 160u, 160u, 160u, 320u, 128u},
     {.wgM = 64u, .wgN = 128u, .sgM = 32u, .sgN = 64u,
      .accStoreBatchSize = 8u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 640u, 256u},
     {.wgM = 64u, .wgN = 128u, .sgM = 32u, .sgN = 64u,
      .accStoreBatchSize = 8u, .storeMode = true, .accType = 1u,
      .reorder = true, .swapXY = 1u, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {160u, 160u, 160u, 160u, 32u, 32u},
     {.wgM = 16u, .wgN = 64u, .sgM = 16u, .sgN = 16u,
      .accStoreBatchSize = 1u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 64u, 64u},
     {.wgM = 64u, .wgN = 64u, .sgM = 64u, .sgN = 16u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 128u, 64u},
     {.wgM = 64u, .wgN = 64u, .sgM = 64u, .sgN = 16u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 384u, 128u},
     {.wgM = 128u, .wgN = 32u, .sgM = 64u, .sgN = 16u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 192u, 128u},
     {.wgM = 128u, .wgN = 32u, .sgM = 64u, .sgN = 16u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 192u, 64u},
     {.wgM = 64u, .wgN = 64u, .sgM = 64u, .sgN = 16u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 96u, 64u},
     {.wgM = 64u, .wgN = 32u, .sgM = 64u, .sgN = 16u,
      .accStoreBatchSize = 4u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 768u, 512u},
     {.wgM = 32u, .wgN = 16u, .sgM = 32u, .sgN = 16u,
      .accStoreBatchSize = 2u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 1024u, 512u},
     {.wgM = 32u, .wgN = 16u, .sgM = 32u, .sgN = 16u,
      .accStoreBatchSize = 2u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 512u, 512u},
     {.wgM = 32u, .wgN = 16u, .sgM = 32u, .sgN = 16u,
      .accStoreBatchSize = 2u, .storeMode = true, .accType = 1u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16,
     ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 2048u, 1024u},
     {.wgM = 32u, .wgN = 16u, .sgM = 32u, .sgN = 16u,
      .accStoreBatchSize = 2u, .storeMode = true, .accType = 1u,
      .ciSpec = 2048u, .nSpec = 400u}},
}};

inline const K1S1CMExactRule* findK1S1CMExactRule(const eva::Device& device, const Workload& workload)
{
    for (const K1S1CMExactRule& rule : kK1S1CMExactTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.weightDtype == workload.wDtype &&
            rule.fusion == workload.fusion && rule.shape.matchShape(workload))
            return &rule;
    }
    return nullptr;
}

struct K1S1ExactRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype inputDtype;
    Dtype outputDtype;
    Dtype weightDtype;
    ConvolutionFusion fusion;
    ShapeSpec shape;
    K1S1Plan plan;
};

constexpr std::array<K1S1ExactRule, 39> kK1S1ExactTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 1024u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 1024u, 512u},
     {.wgM = 64u, .wgN = 64u, .wgK = 8u, .tM = 4u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 1024u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 1280u, 512u},
     {.wgM = 64u, .wgN = 64u, .wgK = 8u, .tM = 4u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 1280u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {160u, 160u, 160u, 160u, 128u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 2048u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 256u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {160u, 160u, 160u, 160u, 320u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 512u, 512u},
     {.wgM = 64u, .wgN = 64u, .wgK = 8u, .tM = 4u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 512u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 640u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 768u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 768u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 768u, 512u},
     {.wgM = 64u, .wgN = 64u, .wgK = 8u, .tM = 4u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 768u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 512u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 384u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 384u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 256u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 192u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Bias,
     {80u, 80u, 80u, 80u, 80u, 80u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Bias,
     {80u, 80u, 80u, 80u, 128u, 80u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 1024u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 1280u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 2048u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {160u, 160u, 160u, 160u, 320u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 640u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {80u, 80u, 80u, 80u, 768u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 1024u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 512u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {40u, 40u, 40u, 40u, 768u, 512u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 384u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u, .splitK = 8u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 512u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u, .splitK = 8u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 256u, 256u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u, .splitK = 8u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp32, Dtype::Fp32, Dtype::Fp32, ConvolutionFusion::Scale,
     {20u, 20u, 20u, 20u, 512u, 128u},
     {.wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u, .splitK = 8u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp16, Dtype::Fp16, Dtype::Fp16, ConvolutionFusion::Bias,
     {80u, 80u, 80u, 80u, 128u, 5u},
     {.wgM = 16u, .wgN = 64u, .wgK = 32u, .tM = 1u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp16, Dtype::Fp16, Dtype::Fp16, ConvolutionFusion::Bias,
     {40u, 40u, 40u, 40u, 128u, 5u},
     {.wgM = 16u, .wgN = 64u, .wgK = 32u, .tM = 1u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u,
     Dtype::Fp16, Dtype::Fp16, Dtype::Fp16, ConvolutionFusion::Bias,
     {20u, 20u, 20u, 20u, 128u, 5u},
     {.wgM = 16u, .wgN = 16u, .wgK = 32u, .tM = 1u}},
}};

inline const K1S1ExactRule* findK1S1ExactRule(const eva::Device& device, const Workload& workload)
{
    for (const K1S1ExactRule& rule : kK1S1ExactTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.inputDtype == workload.iDtype &&
            rule.outputDtype == workload.oDtype && rule.weightDtype == workload.wDtype &&
            rule.fusion == workload.fusion && rule.shape.matchShape(workload))
            return &rule;
    }
    return nullptr;
}

struct ConvBucketStep
{
    uint32_t upperBound;
    uint32_t step;
};

template <size_t N>
constexpr uint32_t convBucket(uint32_t value, const std::array<ConvBucketStep, N>& steps)
{
    for (const ConvBucketStep& entry : steps)
    {
        if (value < entry.upperBound)
            return CEIL_DIV(value, entry.step) * entry.step;
    }

    uint32_t step = steps.back().step;
    uint32_t boundary = steps.back().upperBound;
    while (value >= boundary)
    {
        step *= 2u;
        boundary *= 2u;
    }
    return CEIL_DIV(value, step) * step;
}

constexpr std::array kConvChannelBucketSteps{
    ConvBucketStep{16u, 4u},
    ConvBucketStep{128u, 16u},
    ConvBucketStep{256u, 32u},
    ConvBucketStep{512u, 64u},
    ConvBucketStep{1024u, 128u},
};

constexpr std::array kConvSpatialBucketSteps{
    ConvBucketStep{8u, 4u},
    ConvBucketStep{64u, 8u},
    ConvBucketStep{128u, 16u},
    ConvBucketStep{256u, 32u},
    ConvBucketStep{512u, 64u},
    ConvBucketStep{1024u, 128u},
};

constexpr uint32_t convChannelBucket(uint32_t value)
{
    if (value == 3u)
        return 3u;
    return convBucket(value, kConvChannelBucketSteps);
}

struct K3S1AlgorithmRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    ConvAlgorithm algorithm;
};

// Exact entries in this table choose among specialized K3S1 algorithms.
// General-path overrides are separate because they bypass K3S1 resources.
constexpr std::array<K3S1AlgorithmRule, 2> kK3S1AlgorithmTable{{
    {eva::VENDOR_ID::NVIDIA, 0x2c05u, 70u, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, ConvAlgorithm::Winograd},
}};

inline const K3S1AlgorithmRule* findK3S1AlgorithmRule(const eva::Device& device)
{
    for (const K3S1AlgorithmRule& rule : kK3S1AlgorithmTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount())
            return &rule;
    }
    return nullptr;
}

struct K3S1ExactAlgorithmRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype weightDtype;
    ShapeSpec shape;
    ConvAlgorithm algorithm;
};

constexpr std::array<K3S1ExactAlgorithmRule, 10> kK3S1ExactAlgorithmTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 256u, 256u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 512u, 64u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 64u, 64u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {160u, 160u, 160u, 160u, 64u, 64u}, ConvAlgorithm::Implicit},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 256u, 256u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 256u, 64u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 512u, 64u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 512u, 64u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 512u, 256u}, ConvAlgorithm::Winograd},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 512u, 512u}, ConvAlgorithm::Implicit},
}};

inline const K3S1ExactAlgorithmRule* findK3S1ExactAlgorithmRule(
    const eva::Device& device, const Workload& workload)
{
    for (const K3S1ExactAlgorithmRule& rule : kK3S1ExactAlgorithmTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.weightDtype == workload.wDtype &&
            rule.shape.matchShape(workload))
            return &rule;
    }
    return nullptr;
}

struct K3S1ImplicitRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype weightDtype;
};

constexpr std::array<K3S1ImplicitRule, 1> kK3S1ImplicitTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32},
}};

inline const K3S1ImplicitRule* findK3S1ImplicitRule(
    const eva::Device& device, Dtype weightDtype)
{
    for (const K3S1ImplicitRule& rule : kK3S1ImplicitTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.weightDtype == weightDtype)
            return &rule;
    }
    return nullptr;
}

struct K3S1ImplicitPlanRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype weightDtype;
    ShapeSpec shape;
    uint32_t requiredSubgroupSize;
    uint32_t residualRequiredSubgroupSize;
    ImplicitGemmPlan plan;
};

constexpr std::array<K3S1ImplicitPlanRule, 26> kK3S1ImplicitPlanTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 64u, 80u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 80u, .wgN = 32u, .wgK = 8u, .tM = 5u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 80u, 80u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 80u, .wgN = 32u, .wgK = 8u, .tM = 5u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 128u, 80u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 80u, .wgN = 32u, .wgK = 8u, .tM = 5u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 256u, 80u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 80u, .wgN = 32u, .wgK = 8u, .tM = 5u,
      .structuredInputLoad = true, .alignM = true, .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 128u, 64u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 64u, .wgN = 128u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 256u, 64u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 64u, .wgN = 64u, .wgK = 12u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 512u, 128u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 128u, .wgN = 32u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignK = true,
      .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 128u, 64u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 64u, .wgN = 64u, .wgK = 12u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 256u, 64u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 64u, .wgN = 32u, .wgK = 8u, .tM = 4u,
      .structuredInputLoad = true, .alignM = true, .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 128u, 128u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 128u, .wgN = 32u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 128u, 128u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 128u, .wgN = 32u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 32u, 32u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 32u, .wgN = 64u, .wgK = 8u, .tM = 4u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {160u, 160u, 160u, 160u, 32u, 32u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 32u, .wgN = 64u, .wgK = 8u, .tM = 4u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {160u, 160u, 160u, 160u, 16u, 16u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 16u, .wgN = 64u, .wgK = 8u, .tM = 4u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 256u, 256u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 128u, 128u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 128u, .wgN = 32u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 256u, 256u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 128u, .wgN = 32u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignK = true,
      .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {160u, 160u, 160u, 160u, 64u, 64u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 64u, .wgN = 64u, .wgK = 12u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 256u, 256u},
     0u, 0u,
     {.wgSize = 256u, .wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 512u, 256u},
     32u, 0u,
     {.wgSize = 128u, .wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 512u, 256u},
     32u, 0u,
     {.wgSize = 64u, .wgM = 128u, .wgN = 32u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignK = true,
      .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 512u, 64u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 64u, .wgN = 64u, .wgK = 12u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {20u, 20u, 20u, 20u, 512u, 64u},
     0u, 0u,
     {.wgSize = 256u, .wgM = 64u, .wgN = 16u, .wgK = 32u, .tM = 1u,
      .alignM = true, .alignN = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 256u, 64u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 64u, .wgN = 32u, .wgK = 8u, .tM = 4u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {80u, 80u, 80u, 80u, 64u, 64u},
     32u, 32u,
     {.wgSize = 128u, .wgM = 64u, .wgN = 128u, .wgK = 8u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32,
     {40u, 40u, 40u, 40u, 64u, 64u},
     32u, 32u,
     {.wgSize = 64u, .wgM = 64u, .wgN = 64u, .wgK = 12u, .tM = 8u,
      .structuredInputLoad = true, .alignM = true, .alignN = true,
      .alignK = true, .wideFp32Tile = true}},
}};

inline const K3S1ImplicitPlanRule* findK3S1ImplicitPlanRule(
    const eva::Device& device, const Workload& workload)
{
    if (workload.kernel != 3u || workload.stride != 1u)
        return nullptr;

    for (const K3S1ImplicitPlanRule& rule : kK3S1ImplicitPlanTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.weightDtype == workload.wDtype &&
            rule.shape.matchShape(workload))
            return &rule;
    }
    return nullptr;
}

struct K3S1CMImplicitGeometry
{
    uint32_t wgSize;
    uint32_t wgM;
    uint32_t wgN;
    uint32_t sgM;
    uint32_t sgN;
    uint32_t panelK;
};

struct K3S1CMImplicitRule
{
    uint32_t coBucket;
    uint32_t coDivisor;
    uint32_t maxCiBucket;
    uint32_t minWoBucket;
    uint32_t maxWoBucket;
    K3S1CMImplicitGeometry geometry;
};

constexpr uint32_t kAnyBucket = 0u;
constexpr uint32_t kUnboundedBucket = UINT32_MAX;

constexpr std::array<K3S1CMImplicitRule, 9> kK3S1CMImplicitTable{{
    {64u, 0u, 64u, 160u, kUnboundedBucket, {64u, 64u, 32u, 64u, 16u, 144u}},
    {64u, 0u, kUnboundedBucket, 160u, kUnboundedBucket, {128u, 64u, 32u, 32u, 16u, 144u}},
    {64u, 0u, kUnboundedBucket, 0u, 159u, {64u, 64u, 16u, 32u, 16u, 144u}},
    {128u, 0u, kUnboundedBucket, 160u, kUnboundedBucket, {256u, 128u, 32u, 32u, 16u, 144u}},
    {128u, 0u, kUnboundedBucket, 0u, 159u, {128u, 128u, 16u, 32u, 16u, 144u}},
    {kAnyBucket, 256u, kUnboundedBucket, 160u, kUnboundedBucket, {256u, 256u, 32u, 32u, 32u, 288u}},
    {kAnyBucket, 256u, kUnboundedBucket, 0u, 159u, {256u, 256u, 16u, 32u, 16u, 288u}},
    {kAnyBucket, 128u, kUnboundedBucket, 160u, kUnboundedBucket, {128u, 128u, 32u, 32u, 32u, 288u}},
    {kAnyBucket, 128u, kUnboundedBucket, 0u, 159u, {128u, 128u, 16u, 32u, 16u, 288u}},
}};

struct K3S1CMExactImplicitRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype weightDtype;
    ShapeSpec shape;
    K3S1CMImplicitGeometry geometry;
};

constexpr std::array<K3S1CMExactImplicitRule, 31> kK3S1CMExactImplicitTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 32u, 32u},
     {.wgSize = 64u, .wgM = 32u, .wgN = 32u,
      .sgM = 32u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {160u, 160u, 160u, 160u, 32u, 32u},
     {.wgSize = 64u, .wgM = 32u, .wgN = 32u,
      .sgM = 32u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {160u, 160u, 160u, 160u, 16u, 16u},
     {.wgSize = 64u, .wgM = 16u, .wgN = 32u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 64u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 80u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 128u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 256u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 64u, 144u},
     {.wgSize = 96u, .wgM = 144u, .wgN = 16u,
      .sgM = 48u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 128u, 144u},
     {.wgSize = 96u, .wgM = 144u, .wgN = 16u,
      .sgM = 48u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 256u, 144u},
     {.wgSize = 96u, .wgM = 144u, .wgN = 16u,
      .sgM = 48u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 64u, 64u},
     {.wgSize = 64u, .wgM = 64u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 64u, 64u},
     {.wgSize = 64u, .wgM = 64u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 128u, 64u},
     {.wgSize = 64u, .wgM = 64u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 256u, 64u},
     {.wgSize = 64u, .wgM = 64u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 64u, 144u},
     {.wgSize = 96u, .wgM = 144u, .wgN = 32u,
      .sgM = 48u, .sgN = 32u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 128u, 144u},
     {.wgSize = 96u, .wgM = 144u, .wgN = 16u,
      .sgM = 48u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 256u, 144u},
     {.wgSize = 96u, .wgM = 144u, .wgN = 16u,
      .sgM = 48u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 32u, 32u},
     {.wgSize = 128u, .wgM = 32u, .wgN = 64u,
      .sgM = 32u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {160u, 160u, 160u, 160u, 32u, 32u},
     {.wgSize = 128u, .wgM = 32u, .wgN = 64u,
      .sgM = 32u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {160u, 160u, 160u, 160u, 16u, 16u},
     {.wgSize = 32u, .wgM = 16u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 64u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 80u, 80u},
     {.wgSize = 320u, .wgM = 80u, .wgN = 64u,
      .sgM = 16u, .sgN = 32u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 128u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 256u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {40u, 40u, 40u, 40u, 80u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 80u, 80u},
     {.wgSize = 160u, .wgM = 80u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 720u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 128u, 128u},
     {.wgSize = 128u, .wgM = 128u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .panelK = 576u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 64u, 64u},
     {.wgSize = 128u, .wgM = 64u, .wgN = 16u,
      .sgM = 16u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {160u, 160u, 160u, 160u, 64u, 64u},
     {.wgSize = 32u, .wgM = 64u, .wgN = 16u,
      .sgM = 64u, .sgN = 16u, .panelK = 144u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16,
     {80u, 80u, 80u, 80u, 128u, 128u},
     {.wgSize = 64u, .wgM = 128u, .wgN = 16u,
      .sgM = 64u, .sgN = 16u, .panelK = 288u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16,
     {20u, 20u, 20u, 20u, 512u, 512u},
     {.wgSize = 256u, .wgM = 256u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .panelK = 288u}},
}};

inline const K3S1CMImplicitGeometry* findK3S1CMExactImplicitRule(const eva::Device& device, const Workload& workload)
{
    for (const K3S1CMExactImplicitRule& rule : kK3S1CMExactImplicitTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.weightDtype == workload.wDtype &&
            rule.shape.matchShape(workload))
            return &rule.geometry;
    }
    return nullptr;
}

inline const K3S1CMImplicitGeometry* findK3S1CMImplicitRule(const Workload& workload)
{
    const uint32_t ciBucket = convChannelBucket(workload.ci);
    const uint32_t coBucket = convChannelBucket(workload.co);
    const uint32_t woBucket = convBucket(workload.wo, kConvSpatialBucketSteps);
    for (const K3S1CMImplicitRule& rule : kK3S1CMImplicitTable)
    {
        const bool channelMatch = rule.coBucket != 0u
            ? coBucket == rule.coBucket
            : rule.coDivisor != 0u && (coBucket % rule.coDivisor) == 0u;
        if (channelMatch && ciBucket <= rule.maxCiBucket && woBucket >= rule.minWoBucket && woBucket <= rule.maxWoBucket)
            return &rule.geometry;
    }
    return nullptr;
}

inline bool useHaloK3S1DirectOutput(const Workload& workload)
{
    constexpr std::array<ShapeSpec, 7> rules{{
        {160u, 160u, 160u, 160u, 16u, 16u},
        {80u, 80u, 80u, 80u, 32u, 32u},
        {40u, 40u, 40u, 40u, 64u, 64u},
        {20u, 20u, 20u, 20u, 128u, 128u},
        {80u, 80u, 80u, 80u, 64u, 144u},
        {40u, 40u, 40u, 40u, 128u, 144u},
        {20u, 20u, 20u, 20u, 256u, 144u},
    }};
    for (const ShapeSpec& rule : rules)
        if (rule.matchShape(workload))
            return true;
    return false;
}

struct K3S2ExactRule
{
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t computeBlockCount;
    Dtype weightDtype;
    bool useCM;
    ShapeSpec shape;
    K3S2Plan plan;
};

constexpr std::array<K3S2ExactRule, 18> kK3S2ExactTable{{
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, false,
     {640u, 640u, 320u, 320u, 3u, 16u},
     {.wgSize = 128u, .wgM = 16u, .wgN = 64u, .wgK = 27u, .tM = 2u,
      .shape = {640u, 640u, 320u, 320u, 3u, 16u}}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {320u, 320u, 160u, 160u, 16u, 32u},
     {.wgSize = 32u, .wgM = 32u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u, .shape = {.wo = 160u},
      .accType = 1u, .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32, false,
     {80u, 80u, 40u, 40u, 256u, 256u},
     {.wgSize = 256u, .wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u,
      .shape = {80u, 80u, 40u, 40u, 256u, 256u}}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32, false,
     {80u, 80u, 40u, 40u, 256u, 512u},
     {.wgSize = 256u, .wgM = 128u, .wgN = 64u, .wgK = 8u, .tM = 8u,
      .shape = {80u, 80u, 40u, 40u, 256u, 512u}}},
    {eva::VENDOR_ID::AMD, kAMDStrixPointDeviceId, 16u, Dtype::Fp32, false,
     {40u, 40u, 20u, 20u, 512u, 512u},
     {.wgSize = 256u, .wgM = 64u, .wgN = 64u, .wgK = 8u, .tM = 4u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {320u, 320u, 160u, 160u, 64u, 128u},
     {.wgSize = 128u, .wgM = 128u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u,
      .shape = {320u, 320u, 160u, 160u, 64u, 128u}}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {80u, 80u, 40u, 40u, 256u, 256u},
     {.wgSize = 256u, .wgM = 256u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u,
      .shape = {80u, 80u, 40u, 40u, 256u, 256u}}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {160u, 160u, 80u, 80u, 32u, 64u},
     {.wgSize = 64u, .wgM = 64u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u, .shape = {.wo = 80u},
      .accType = 1u, .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {80u, 80u, 40u, 40u, 64u, 64u},
     {.splitK = 3u, .wgSize = 64u, .wgM = 64u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u, .shape = {.wo = 40u},
      .accType = 1u, .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {80u, 80u, 40u, 40u, 64u, 128u},
     {.splitK = 3u, .wgSize = 128u, .wgM = 128u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u, .shape = {.wo = 40u},
      .accType = 1u, .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {40u, 40u, 20u, 20u, 128u, 128u},
     {.splitK = 9u, .wgSize = 128u, .wgM = 128u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u, .shape = {.wo = 20u},
      .accType = 1u, .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp16, true,
     {40u, 40u, 20u, 20u, 128u, 256u},
     {.splitK = 9u, .wgSize = 128u, .wgM = 256u, .wgN = 32u,
      .sgM = 64u, .sgN = 32u, .shape = {.wo = 20u},
      .accType = 1u, .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp32, false,
     {160u, 160u, 80u, 80u, 128u, 256u},
     {.wgSize = 256u, .wgM = 128u, .wgN = 64u, .wgK = 16u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp32, false,
     {80u, 80u, 40u, 40u, 256u, 512u},
     {.wgSize = 256u, .wgM = 128u, .wgN = 64u, .wgK = 16u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDStrixHaloDeviceId, 40u, Dtype::Fp32, false,
     {40u, 40u, 20u, 20u, 512u, 512u},
     {.splitK = 4u, .wgSize = 256u, .wgM = 128u, .wgN = 64u,
      .wgK = 8u, .tM = 8u}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16, true,
     {160u, 160u, 80u, 80u, 64u, 128u},
     {.wgSize = 128u, .wgM = 128u, .wgN = 16u,
      .sgM = 32u, .sgN = 16u, .shape = {.wo = 80u},
      .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16, true,
     {80u, 80u, 40u, 40u, 128u, 256u},
     {.splitK = 3u, .wgSize = 128u, .wgM = 128u, .wgN = 32u,
      .sgM = 32u, .sgN = 32u, .shape = {.wo = 40u},
      .packedWeights = true, .directOutput = true}},
    {eva::VENDOR_ID::AMD, kAMDNavi48DeviceId, 64u, Dtype::Fp16, true,
     {40u, 40u, 20u, 20u, 256u, 256u},
     {.splitK = 9u, .wgSize = 128u, .wgM = 256u, .wgN = 16u,
      .sgM = 64u, .sgN = 16u, .shape = {.wo = 20u},
      .packedWeights = true, .directOutput = true}},
}};

inline const K3S2ExactRule* findK3S2ExactRule(const eva::Device& device, const Workload& workload)
{
    for (const K3S2ExactRule& rule : kK3S2ExactTable)
    {
        if (rule.vendorId == device.vendorID() && rule.deviceId == device.deviceID() &&
            rule.computeBlockCount == device.computeBlockCount() && rule.weightDtype == workload.wDtype &&
            rule.shape.matchShape(workload))
            return &rule;
    }
    return nullptr;
}

/*=============================================================================
  Convolution - 2D convolution with device-selected direct, implicit, or Winograd plans
  Input: in0 [..., Ci, Hi, Wi], optional residual [..., Co, Ho, Wo]
  Params: weight [Co, Ci, K, K], optional scale [Co] and bias [Co]
  Output: out0 [..., Co, ceil(Hi / stride), ceil(Wi / stride)]

  The "scale" and "bias" slots always exist; each applies only when the weight
  loader filled it. Odd kernels use symmetric same-padding, so each spatial
  output size is ceil(inputSize / stride).
  acceptResidual opens the "residual" input (input dtype), added after the epilogue
  or, with ResidualOrder::BeforeEpilogue, before it.
  The axes ahead of Ci are a batch of images. The plan is one image's; every
  dispatch stacks the images on its z axis, and each kernel offsets its buffers
  to its image.
=============================================================================*/
class Convolution : public vai::Node
{
public:
    struct Config {
        uint32_t inChannels;
        uint32_t outChannels;
        uint32_t kernelSize;
        uint32_t stride;

        bool acceptResidual = false;   // input "residual"
        vai::ResidualOrder residualOrder = vai::ResidualOrder::AfterEpilogue;
        vai::Epilogue epilogue = vai::Epilogue::None;
    };

    Convolution(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // The weight loader filled "bias" / "scale"; fixed once the weights are up.
    bool hasBias() const { return (*this)["bias"].validShape(); }
    bool hasScale() const { return (*this)["scale"].validShape(); }

private:
    Workload makeWorkload();
    void validateInputs(const Workload& workload);
    void preparePlan(const Workload& workload, const ConvPlan& nextPlan);
    void prepareK1S1(const Workload& workload);
    void prepareK3S1(const Workload& workload);
    void prepareK3S2(const Workload& workload);
    void prepareImplicit(const Workload& workload);
    void prepareTensorStorage(const Workload& workload);
    bool hasPendingWeightTransforms() const;
    bool recordK3S1WeightTransforms(eva::CommandBuffer cmdBuff, const Workload& workload);
    bool recordK1S1WeightTransform(eva::CommandBuffer cmdBuff, const Workload& workload);
    bool recordK3S2WeightTransform(eva::CommandBuffer cmdBuff, const Workload& workload);
    bool recordImplicitWeightTransform(eva::CommandBuffer cmdBuff, const Workload& workload);
    bool recordWeightTransforms(eva::CommandBuffer cmdBuff, const Workload& workload);
    void transformWeight(const Workload& workload);
    void runK3S1Implicit(eva::CommandBuffer cmdBuff, const Workload& workload);
    void runK3S1Winograd(eva::CommandBuffer cmdBuff, const Workload& workload);
    void runK1S1(eva::CommandBuffer cmdBuff, const Workload& workload);
    void runK3S2(eva::CommandBuffer cmdBuff, const Workload& workload);
    void runConv(eva::CommandBuffer cmdBuff, const Workload& workload);

    ConvPlan plan_;
    ShapeSpec preparedShape_;
    bool planPrepared_ = false;

    // Fusion-related pipeline inputs shared by every prepare path. The
    // batch-norm rule (affine parameters carry the weight dtype and
    // affineMode 1) lives only here.
    struct FusionSpec
    {
        Dtype iDtype = Dtype::None;
        Dtype oDtype = Dtype::None;
        Dtype affineDtype = Dtype::None;
        uint32_t hasScale = 0u;
        uint32_t epilogue = 0u;
    };
    FusionSpec makeFusionSpec(const Workload& workload) const
    {
        const bool hasScale = workload.fusion == ConvolutionFusion::Scale;
        return {
            workload.iDtype, workload.oDtype, workload.wDtype, hasScale ? 1u : 0u, (uint32_t)config_.epilogue,
        };
    }
    ResidualFusion residualFusion() const { return {config_.acceptResidual, (uint32_t)config_.residualOrder}; }

    struct ImplicitResources
    {
        PipelineBinding gemm;
        PipelineBinding weightTransform;
        eva::Buffer transformedWeight;
        bool weightTransformed = false;
    };

    // Shared by specialized K1S1, K3S1, and K3S2 paths. A node only
    // prepares one of these paths.
    struct CooperativeMatrixResources
    {
        eva::Buffer fragmentWeight;
        PipelineBinding transform;
        PipelineBinding reduce;
        bool weightTransformed = false;
    };

    // Shared by K1S1 and the base K3S2 fallback.
    struct TiledResources
    {
        PipelineBinding main;
        PipelineBinding split;
        PipelineBinding reduce;
    };

    struct K1S1Resources
    {
        CooperativeMatrixResources cm;
        TiledResources tiled;
    };

    struct WinogradResources
    {
        PipelineBinding filterTransform;
        PipelineBinding inputTransform;
        PipelineBinding gemm;
        PipelineBinding outputTransform;
        eva::ComputePipeline cmTailSharedDescriptorPipeline;
        PipelineBinding cmTailGemm;
        eva::Buffer transformedWeight;
        CooperativeMatrixResources cm;
        bool weightTransformed = false;
    };

    struct K3S1Resources
    {
        ImplicitResources implicit;
        WinogradResources winograd;
    };

    struct K3S2Resources
    {
        CooperativeMatrixResources cm;
        TiledResources tiled;
    };

    using PreparedState = std::variant<
        std::monostate,
        K1S1Resources,
        K3S1Resources,
        K3S2Resources,
        ImplicitResources>;
    PreparedState prepared_;

    // Shared by the fp32 and fp16 K1S1 split paths.
    void prepareK1S1Split(TiledResources& tiled, const Workload& workload);

    // One-shot weight transform for kernels whose work-item count equals
    // the fp16 element count of the transformed buffer.
    bool recordFp16WeightTransform(
        eva::CommandBuffer cmdBuff,
        PipelineBinding& transform,
        const eva::BufferRange& srcWeight,
        eva::Buffer& transformedWeight,
        bool& transformed,
        const Workload& workload,
        uint64_t elements)
    {
        const PushConstants constants = makePushConstants(workload);
        return ensureWeightTransform(
            device(), cmdBuff, transform, srcWeight, transformedWeight, constants,
            size_t(elements) * sizeof(uint16_t), uint32_t(elements),
            transformed);
    }

public:
    Config config_;
};

/////////////////////////////////////////////////////////////////////////////////////////
// K1S1 path selection and validation
/////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t selectK1S1SplitK(const eva::Device& device, const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const uint32_t computeBlockCount = device.computeBlockCount();
    const uint32_t ci = workload.ci;
    const uint32_t co = workload.co;
    const uint32_t n = workload.spatialSize();
    constexpr uint32_t wgM = 128u;
    constexpr uint32_t wgN = 64u;
    const uint32_t outputWorkgroups = CEIL_DIV(n, wgN) * CEIL_DIV(co, wgM);
    const uint32_t kTiles = CEIL_DIV(ci, 16u);
    const uint64_t outputElements = uint64_t(co) * uint64_t(n);

    uint32_t splitK = 1u;
    if (vendorId == eva::VENDOR_ID::NVIDIA)
    {
        const uint32_t lowWorkgroups = scaleWorkSupplyThreshold(24u, computeBlockCount);
        const uint32_t sufficientWorkgroups = scaleWorkSupplyThreshold(64u, computeBlockCount);

        splitK = outputWorkgroups < lowWorkgroups ? 8u : outputWorkgroups < sufficientWorkgroups ? 4u : 2u;
    }
    else if (vendorId == eva::VENDOR_ID::AMD)
    {
        if (ci < 512u || (ci <= 512u && outputElements <= uint64_t(256u) * 512u))
        {
            splitK = 1u;
        }
        else
        {
            splitK = outputWorkgroups < scaleWorkSupplyThreshold(64u, computeBlockCount) ? 8u : 4u;
        }
    }
    else
    {
        splitK = ci >= 1536u ? 4u : 2u;
    }

    while (splitK > 1u && CEIL_DIV(kTiles, splitK) < 4u)
        splitK >>= 1u;
    return splitK;
}

/////////////////////////////////////////////////////////////////////////////////////////
// K3S1 path selection and validation
/////////////////////////////////////////////////////////////////////////////////////////

inline bool validCMTopology(uint32_t wgSize, uint32_t wgM, uint32_t wgN, uint32_t sgM, uint32_t sgN, uint32_t subgroupSize)
{
    if (subgroupSize == 0u || wgSize < subgroupSize || (wgSize % subgroupSize) != 0u ||
        sgM < 16u || sgN < 16u || (sgM % 16u) != 0u || (sgN % 16u) != 0u ||
        (wgM % sgM) != 0u || (wgN % sgN) != 0u)
        return false;
    return (wgM / sgM) * (wgN / sgN) == wgSize / subgroupSize;
}

inline uint32_t k3s1SharedMemoryBytes(const K3S1Plan& plan, uint32_t dispatchWgN, Dtype weightDtype)
{
    VAI_ASSERT(plan.valid() && dispatchWgN != 0u);

    if (plan.algorithm == ConvAlgorithm::Implicit)
    {
        const uint32_t bsBytes = plan.panelK * dispatchWgN * sizeof(uint16_t);
        const uint32_t accStoreBytes = (plan.wgSize / 32u) * 16u * 16u * sizeof(float);
        const uint32_t affineBytes =
            (plan.wgSize / 32u) * 16u * 2u * sizeof(float);
        return bsBytes + accStoreBytes + affineBytes;
    }

    VAI_ASSERT(plan.algorithm == ConvAlgorithm::Winograd);
    if (plan.sgSize == 0u)
    {
        VAI_ASSERT(weightDtype == Dtype::Fp32 || weightDtype == Dtype::Fp16);
        const uint32_t elements = plan.wgK * (plan.wgM + 4u + 2u * dispatchWgN + plan.pad);
        return weightDtype == Dtype::Fp16
            ? elements * sizeof(uint16_t) * 2u + 6u * sizeof(float) * 2u
            : elements * sizeof(float) * 2u + 6u * sizeof(uint16_t) * 2u;
    }

    const uint32_t subgroupCount = plan.wgSize / plan.sgSize;
    const uint32_t nIter = dispatchWgN / 16u;
    const uint32_t inputElements = plan.loadMode ? 2u * (plan.panelK / 16u) * nIter * 16u * 16u : 1u;
    const uint32_t partialStoreElements = !plan.storeMode && nIter == 1u ? subgroupCount * 16u * 16u : 1u;
    const uint32_t fp16StoreElements = plan.storeMode ? plan.wgM * dispatchWgN : 1u;
    const uint32_t fp16Elements = std::max(inputElements, std::max(partialStoreElements, fp16StoreElements));
    const uint32_t fp32Elements = plan.accType == 1u ? plan.wgM * dispatchWgN : 1u;
    return sizeof(uint16_t) * fp16Elements + sizeof(float) * fp32Elements;
}

inline bool validK3S1Plan(const K3S1Plan& plan, const eva::Device& device, bool useCooperativeMatrix, const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const uint32_t ci = workload.ci;
    const uint32_t co = workload.co;
    const uint32_t numTiles = workload.winogradTileCount();
    if (ci == 0u || co == 0u || numTiles < 16u)
        return false;

    switch (plan.algorithm)
    {
        case ConvAlgorithm::Winograd:
            if (!useCooperativeMatrix)
                return plan.wgSize != 0u && plan.wgM != 0u &&
                    plan.wgN != 0u && plan.wgK != 0u &&
                    plan.tM != 0u && plan.tN != 0u &&
                    (plan.wgM % plan.tM) == 0u &&
                    (plan.wgN % plan.tN) == 0u &&
                    plan.wgSize ==
                        (plan.wgM / plan.tM) * (plan.wgN / plan.tN) &&
                    plan.sgSize == 0u && plan.sgM == 0u && plan.sgN == 0u &&
                    plan.alignM == ((co % plan.wgM) == 0u) &&
                    plan.alignN == ((numTiles % plan.wgN) == 0u) &&
                    plan.alignK == ((ci % plan.wgK) == 0u) &&
                    plan.storeMode == ((numTiles % 4u) == 0u) &&
                    !plan.hasTail();

            if (plan.accType > 1u || (plan.accType == 1u && plan.splitK != 1u) ||
                !validCMTopology(plan.wgSize, plan.wgM, plan.wgN, plan.sgM, plan.sgN, plan.sgSize) ||
                plan.sgN != plan.wgN || plan.splitK == 0u ||
                (co % plan.wgM) != 0u)
                return false;

            if (vendorId == eva::VENDOR_ID::NVIDIA)
            {
                return plan.sgSize == 32u && plan.panelK == 16u &&
                    !plan.loadMode && plan.storeMode && plan.reorder &&
                    (!plan.hasTail() ||
                     (plan.splitK == 1u && plan.wgN == 32u &&
                      plan.tailNOffset < numTiles));
            }
            if (vendorId != eva::VENDOR_ID::AMD)
                return false;

            if (plan.splitK != 1u || plan.panelK < 16u ||
                (plan.panelK % 16u) != 0u ||
                (plan.reorder &&
                 ((ci % 16u) != 0u || (co % 16u) != 0u)) ||
                (plan.inputReorder &&
                 ((ci % 16u) != 0u || plan.loadMode)))
                return false;
            if (plan.hasTail())
            {
                const uint32_t remainder = numTiles % plan.wgN;
                const uint32_t expectedTailOffset = numTiles - remainder;
                if (plan.wgN != 32u ||
                    remainder == 0u || remainder == 16u ||
                    plan.tailNOffset != expectedTailOffset ||
                    plan.tailNOffset >= numTiles ||
                    (plan.tailNOffset % 16u) != 0u ||
                    !validCMTopology(plan.wgSize, plan.wgM, kK3S1TailWgN, plan.sgM, kK3S1TailWgN, plan.sgSize))
                    return false;
            }
            if ((numTiles % 16u) != 0u && plan.wgN > 16u && !plan.hasTail())
                return plan.storeMode;
            return true;

        case ConvAlgorithm::Implicit:
        {
            if (!useCooperativeMatrix || plan.accType != 1u ||
                !validCMTopology(plan.wgSize, plan.wgM, plan.wgN, plan.sgM, plan.sgN, plan.sgSize) ||
                plan.sgSize != 32u ||
                plan.shape.ci != ci || plan.shape.co != co ||
                (co % plan.wgM) != 0u ||
                ((ci * 9u) % 16u) != 0u ||
                plan.panelK < 16u || (plan.panelK % 16u) != 0u ||
                (plan.panelK % 9u) != 0u ||
                ((ci * 9u) % plan.panelK) != 0u ||
                (plan.wgN % 16u) != 0u)
                return false;

            if (plan.flatSpatial &&
                ((workload.wo % 8u) != 0u ||
                 (workload.spatialSize() % plan.wgN) != 0u))
                return false;

            const uint32_t sgpMIter = plan.sgM / 16u;
            const uint32_t sgpNIter = plan.sgN / 16u;
            return sgpMIter <= 4u && sgpNIter <= 4u && sgpMIter * sgpNIter <= 16u;
        }

        case ConvAlgorithm::None:
            return false;
    }
    return false;
}

// K3S2 path selection and validation
/////////////////////////////////////////////////////////////////////////////////////////

inline bool validK3S2Plan(const K3S2Plan& plan, const Workload& workload)
{
    const uint32_t Ci = workload.ci;
    const uint32_t Co = workload.co;
    const uint32_t Ho = workload.ho;
    const uint32_t Wo = workload.wo;
    if (!plan.valid())
        return false;

    if (plan.splitK == 0u || plan.splitK > 16u ||
        Ci == 0u || Co == 0u || Ho == 0u || Wo == 0u)
        return false;

    if (plan.firstLayer())
        return !plan.split() && Ci == 3u && Co <= 64u &&
            plan.wgSize >= 32u && plan.wgSize <= 256u &&
            (plan.wgSize % 32u) == 0u &&
            plan.wgM == Co && (plan.wgN % 4u) == 0u &&
            plan.tM >= 1u && plan.tM <= 8u &&
            (plan.wgM % plan.tM) == 0u &&
            plan.wgSize == (plan.wgM / plan.tM) * (plan.wgN / 4u) &&
            plan.wgSize >= plan.wgN && (plan.wgSize % plan.wgN) == 0u &&
            plan.sgM == 0u && plan.sgN == 0u &&
            !plan.packedWeights && !plan.directOutput &&
            plan.alignM == ((Co % plan.wgM) == 0u) &&
            plan.alignN == (((Ho * Wo) % plan.wgN) == 0u) &&
            plan.alignK &&
            plan.shape.matchSpatial(workload);

    if (plan.cooperativeMatrix())
    {
        if (plan.accType > 1u ||
            (plan.directOutput && !plan.packedWeights) ||
            plan.wgK != 0u || plan.tM != 0u ||
            !validCMTopology(
                plan.wgSize, plan.wgM, plan.wgN,
                plan.sgM, plan.sgN, 32u))
            return false;

        const uint32_t kk = Ci * 9u;
        const bool alignK = plan.packedWeights ||
            ((kk % plan.splitK) == 0u && ((kk / plan.splitK) % 16u) == 0u);
        if (plan.alignM != ((Co % plan.wgM) == 0u) ||
            plan.alignN != (((Ho * Wo) % plan.wgN) == 0u) ||
            plan.alignK != alignK)
            return false;
    }

    if (plan.cooperativeMatrix() && !plan.packedWeights)
    {
        const uint32_t kk = Ci * 9u;
        const uint32_t sgpMIter = plan.sgM / 16u;
        const uint32_t sgpNIter = plan.sgN / 16u;
        if (plan.accType != 1u || (kk % 16u) != 0u ||
            sgpMIter > 2u || sgpNIter > 4u ||
            sgpMIter * sgpNIter > 4u)
            return false;

        if (plan.shape.empty())
            return !plan.split() && plan.sgM == 16u &&
                plan.sgN == 16u && plan.alignM && plan.alignN;

        if (!plan.shape.matchShape(workload))
            return false;
        if (plan.split())
            return plan.sgM == 16u &&
                plan.sgN == 64u && (kk % plan.splitK) == 0u &&
                ((kk / plan.splitK) % 16u) == 0u;
        return true;
    }

    if (plan.cooperativeMatrix() && plan.packedWeights)
    {
        const uint32_t tapsPerSplit = CEIL_DIV(9u, plan.splitK);
        const uint32_t wgSgN = plan.wgN / plan.sgN;
        return (Ci % 16u) == 0u && ((Ho * Wo) % 4u) == 0u &&
            (!plan.directOutput || ((Ho * Wo) % 16u) == 0u) &&
            (plan.splitK - 1u) * tapsPerSplit < 9u &&
            wgSgN == 1u && plan.wgSize >= plan.wgN &&
            (plan.wgSize % plan.wgN) == 0u &&
            (16u % (plan.wgSize / plan.wgN)) == 0u &&
            plan.shape.hi == 0u && plan.shape.wi == 0u &&
            plan.shape.ho == 0u && plan.shape.wo == Wo &&
            plan.shape.ci == 0u && plan.shape.co == 0u;
    }

    if (plan.sgM != 0u || plan.sgN != 0u || plan.packedWeights ||
        plan.directOutput || plan.wgSize < 32u || plan.wgSize > 256u ||
        (plan.wgSize % 32u) != 0u || plan.wgM == 0u ||
        plan.wgN == 0u || (plan.wgN % 4u) != 0u ||
        plan.wgK == 0u || plan.tM == 0u ||
        (plan.wgM % plan.tM) != 0u ||
        plan.wgSize != (plan.wgM / plan.tM) * (plan.wgN / 4u) ||
        plan.wgSize < plan.wgN || (plan.wgSize % plan.wgN) != 0u ||
        plan.wgSize < plan.wgK || (plan.wgSize % plan.wgK) != 0u)
        return false;

    const uint32_t kk = Ci * 9u;
    const uint32_t kPerSplit = CEIL_DIV(kk, plan.splitK);
    const bool alignK = (kk % plan.splitK) == 0u && (kPerSplit % plan.wgK) == 0u;
    if (plan.alignM != ((Co % plan.wgM) == 0u) ||
        plan.alignN != (((Ho * Wo) % plan.wgN) == 0u) ||
        plan.alignK != alignK)
        return false;

    if (workload.wDtype != Dtype::Fp32)
        return plan.shape.empty() &&
            plan.wgK == 8u && plan.tM == plan.wgM / 16u &&
            (plan.wgM == 64u || plan.wgM == 128u) &&
            (plan.splitK == 1u || plan.splitK == 4u || plan.splitK == 8u);

    if (plan.split())
        return plan.shape.empty() && plan.wgM == 128u && plan.wgK == 8u && plan.tM == 8u &&
            (plan.splitK == 4u || plan.splitK == 8u);
    if (!plan.shape.empty())
    {
        if (!plan.alignM || !plan.alignN ||
            !plan.shape.matchShape(workload))
            return false;
        if (plan.wgM == 64u)
            return plan.wgK == 8u && plan.tM == 4u;
        if (plan.wgM == 128u)
            return plan.alignK && plan.wgK == 8u && plan.tM == 8u;
        return plan.alignK && plan.wgM == 256u &&
            plan.wgK == 8u && plan.tM == 16u;
    }
    if (plan.alignM && plan.alignN && plan.alignK)
        return plan.wgM == 128u && plan.wgK == 16u && plan.tM == 8u;
    return plan.wgM == 64u && plan.wgK == 8u && plan.tM == 4u;
}

// Fallback convolution path selection and validation
/////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t convSharedMemoryBytes(const ImplicitGemmPlan& plan, Dtype weightDtype)
{
    if (plan.cooperativeMatrix)
    {
        const uint32_t accBytes = plan.accType == 0u ? 2u : 4u;
        const uint32_t fp16StoreAlignmentScratch = plan.accType == 0u ? 16u * 16u * sizeof(float) : 0u;
        return plan.panelK * plan.wgN * 2u + (plan.wgSize / 32u) * 16u * 16u * accBytes + fp16StoreAlignmentScratch + 16u;
    }
    const uint32_t elements = plan.wgK * (
        plan.wgM + (plan.padA ? 4u : 0u) + plan.wgN);

    if (plan.wideFp32Tile)
        return 2u * plan.wgK * (plan.wgM + plan.wgN) * sizeof(float);

    return weightDtype == Dtype::Fp16
        ? elements * sizeof(uint16_t) + 2u * sizeof(float)
        : elements * sizeof(float) + 2u * sizeof(uint16_t);
}

inline bool validImplicitGemmPlan(
    const ImplicitGemmPlan& plan,
    const eva::Device& device,
    const ConvolutionCaps& caps,
    Dtype weightDtype)
{
    if (weightDtype != Dtype::Fp16 && weightDtype != Dtype::Fp32)
        return false;

    if (plan.wgSize == 0u || plan.wgM == 0u || plan.wgN == 0u)
        return false;
    if (plan.wideFp32Tile)
    {
        if (weightDtype != Dtype::Fp32 ||
            plan.cooperativeMatrix || plan.sgM != 0u || plan.sgN != 0u ||
            plan.tM != 8u ||
            !plan.structuredInputLoad || !plan.alignM || !plan.alignK)
            return false;

        const bool validTopology =
            (plan.wgSize == 256u && plan.wgM == 128u &&
             plan.wgN == 128u && plan.wgK == 8u) ||
            (plan.wgSize == 128u && plan.wgM == 64u &&
             plan.wgN == 128u && plan.wgK == 8u) ||
            (plan.wgSize == 128u && plan.wgM == 128u &&
             plan.wgN == 64u && plan.wgK == 8u && plan.alignN) ||
            (plan.wgSize == 64u && plan.wgM == 128u &&
             plan.wgN == 32u && plan.wgK == 8u) ||
            (plan.wgSize == 64u && plan.wgM == 64u &&
             plan.wgN == 64u && plan.wgK == 12u && plan.alignN);
        if (!validTopology)
            return false;

        return convSharedMemoryBytes(plan, weightDtype) <=
            device.maxComputeSharedMemorySize();
    }
    if (plan.cooperativeMatrix)
    {
        if (weightDtype != Dtype::Fp16 || !enableCM(device, caps) ||
            plan.accType > 1u || (plan.accType == 0u && !caps.accFP16) ||
            plan.panelK == 0u ||
            plan.sgM != 16u || plan.sgN != 16u ||
            (plan.panelK % 16u) != 0u || (plan.wgM % 16u) != 0u ||
            (plan.wgN % 16u) != 0u ||
            !validCMTopology(
                plan.wgSize, plan.wgM, plan.wgN,
                plan.sgM, plan.sgN, 32u))
            return false;
    }
    else if (plan.sgM != 0u || plan.sgN != 0u ||
             plan.wgK == 0u || plan.tM == 0u ||
             (plan.wgN % 4u) != 0u || (plan.wgM % plan.tM) != 0u ||
             (plan.structuredInputLoad &&
              (plan.wgSize < plan.wgN || (plan.wgSize % plan.wgN) != 0u)) ||
             plan.wgSize != (plan.wgM / plan.tM) * (plan.wgN / 4u))
        return false;

    return convSharedMemoryBytes(plan, weightDtype) <= device.maxComputeSharedMemorySize();
}

inline constexpr std::array<ImplicitGemmPlan, 4>
    gConvFallbacks{{
        {},
        {.wgSize = 64u, .wgM = 32u, .wgN = 32u},
        {.wgSize = 32u, .wgM = 32u, .wgN = 16u},
        {.wgSize = 16u, .wgM = 16u, .wgN = 16u},
    }};

// Complete plan validation
/////////////////////////////////////////////////////////////////////////////////////////

inline bool validK1S1Plan(const ConvPlan& convPlan, const Workload& workload)
{
    const K1S1Plan& plan = convPlan.k1s1;
    const uint32_t ci = workload.ci;
    const uint32_t co = workload.co;
    const uint32_t n = workload.spatialSize();

    if (convPlan.path != ConvPlan::Path::K1S1 ||
        workload.kernel != 1u || workload.stride != 1u ||
        ci == 0u || co == 0u || n == 0u ||
        convPlan.iDtype != workload.iDtype ||
        convPlan.oDtype != workload.oDtype ||
        convPlan.wDtype != workload.wDtype ||
        convPlan.fusion != workload.fusion ||
        (plan.splitK != 1u && plan.splitK != 2u && plan.splitK != 4u && plan.splitK != 8u))
        return false;

    if (workload.wDtype == Dtype::Fp32)
    {
        if (convPlan.useCM)
            return false;
    }
    else if (convPlan.useCM)
    {
        if (workload.wDtype != Dtype::Fp16 ||
            (ci % 16u) != 0u || (co % 16u) != 0u ||
            plan.sgSize == 0u || plan.sgM < 16u || plan.sgN < 16u ||
            (plan.sgM % 16u) != 0u || (plan.sgN % 16u) != 0u ||
            plan.loadMode > 1u || plan.accType > 1u || plan.swapXY > 1u ||
            (plan.ciSpec != 0u && plan.ciSpec != ci) ||
            (plan.nSpec != 0u && plan.nSpec != n))
            return false;

        if (plan.wgM < plan.sgM || plan.wgN < plan.sgN ||
            (plan.wgM % plan.sgM) != 0u ||
            (plan.wgN % plan.sgN) != 0u)
            return false;

        const uint32_t sgpMIter = plan.sgM / 16u;
        const uint32_t sgpNIter = plan.sgN / 16u;
        const uint32_t accFrags = sgpMIter * sgpNIter;
        if (plan.accStoreBatchSize == 0u || plan.accStoreBatchSize > accFrags || plan.wgM > co)
            return false;
        if (plan.wgK != 16u ||
            plan.alignM != ((co % plan.wgM) == 0u) ||
            plan.alignN != ((n % plan.wgN) == 0u) ||
            plan.alignK != ((ci % plan.wgK) == 0u))
            return false;
        if (plan.storeMode && (n % 4u) != 0u)
            return false;
        if (plan.accType == 0u && sgpMIter > 4u)
            return false;
        if (plan.accType == 1u && plan.splitK != 1u)
            return false;
        if (plan.accType == 1u && (!plan.alignM || !plan.alignN) &&
            (sgpMIter != 1u || sgpNIter != 1u ||
             plan.accStoreBatchSize != 1u))
            return false;
        if (plan.directOutput && (plan.accType != 1u || !plan.alignM || !plan.alignN || plan.splitK != 1u))
            return false;
        return true;
    }

    if (plan.sgM != 0u || plan.sgN != 0u ||
        plan.reorder || plan.directOutput ||
        plan.ciSpec != 0u || plan.nSpec != 0u)
        return false;

    const uint32_t expectedWgM =
        convPlan.vendorId == eva::VENDOR_ID::AMD ? 128u : 64u;

    const uint32_t kPerSplit = CEIL_DIV(ci, plan.splitK);
    if (plan.wgM == 0u || plan.wgN == 0u || plan.wgK == 0u ||
        plan.alignM != ((co % plan.wgM) == 0u) ||
        plan.alignN != ((n % plan.wgN) == 0u) ||
        plan.alignK != ((ci % plan.splitK) == 0u && (kPerSplit % plan.wgK) == 0u))
        return false;

    if (plan.splitK > 1u)
    {
        if (workload.fusion != ConvolutionFusion::Scale)
            return false;
        return plan.wgM == 128u && plan.wgN == 64u && plan.tM == 8u &&
            (plan.wgK == 8u ||
             (convPlan.vendorId == eva::VENDOR_ID::AMD &&
              plan.wgK == 16u));
    }

    const bool validSmallOutputPlan =
        workload.wDtype == Dtype::Fp16 && workload.co <= 16u &&
        plan.wgM == 16u && plan.tM == 1u &&
        (plan.wgK == 16u || plan.wgK == 32u) &&
        (plan.wgN == 16u || plan.wgN == 32u || plan.wgN == 64u);
    const bool validDirectWgM = workload.wDtype == Dtype::Fp32
        ? plan.wgM == 64u ||
            (convPlan.vendorId == eva::VENDOR_ID::AMD &&
             plan.wgM == 128u)
        : plan.wgM == expectedWgM;
    const bool validDefaultPlan =
        validDirectWgM && plan.wgN == 64u && plan.wgK == 8u &&
        plan.tM == plan.wgM / 16u;
    return validDefaultPlan || validSmallOutputPlan;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Pipeline factories
/////////////////////////////////////////////////////////////////////////////////////////

// The accumulator spec constant keeps the shader's inverted polarity
// (0 = fp16, 1 = fp32); this is the only remaining uint32 dtype slot.
inline Dtype accDtypeFromShaderType(uint32_t type)
{
    VAI_ASSERT(type <= 1u);
    return type == 0u ? Dtype::Fp16 : Dtype::Fp32;
}

inline void withShape(eva::SpecializationConstant& specialization, const ShapeSpec& shape, bool includeChannels)
{
    if (includeChannels)
    {
        specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_CI>(shape.ci));
        specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_CO>(shape.co));
    }
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_HI>(shape.hi));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_HO>(shape.ho));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_WI>(shape.wi));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_WO>(shape.wo));
}

struct DTypeSpecConfig
{
    bool includeWeight = false;
    bool includeAcc = false;
};

inline void withDTypes(
    eva::SpecializationConstant& specialization,
    Dtype oDtype,
    Dtype iDtype,
    Dtype paramType,
    Dtype typeW,
    uint32_t accType,
    DTypeSpecConfig config)
{
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_O>(oDtype));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_I>(iDtype));
    if (config.includeWeight)
        specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_W>(typeW));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_P>(paramType));
    if (config.includeAcc)
        specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_ACC>(accType));
}

inline void withFusion(eva::SpecializationConstant& specialization, uint32_t epilogue, uint32_t hasScale, uint32_t hasBias)
{
    specialization.addConstant(eva::constant_id<SPEC_ID_EPILOGUE>(epilogue));
    specialization.addConstant(eva::constant_id<SPEC_ID_HAS_SCALE>(hasScale));
    specialization.addConstant(eva::constant_id<SPEC_ID_HAS_BIAS>(hasBias));
}

inline eva::ComputePipeline k1s1CMPipeline(PipelinePool& pool, const K1S1Plan& plan, uint32_t kPerSplit, uint32_t ep,
                                           Dtype paramType, uint32_t hasScale, uint32_t hasBias,
                                           ResidualFusion residual, Dtype iDtype, Dtype oDtype,
                                           uint32_t requiredSubgroupSize)
{
    const uint32_t subgroupCount =
        (plan.wgM / plan.sgM) * (plan.wgN / plan.sgN);
    const uint32_t wgSize = plan.sgSize * subgroupCount;

    return pool.request("conv-k1s1-gemm-cm",
                        {
                            eva::constant_id<0>(wgSize),
                            eva::constant_id<1>(plan.wgM),
                            eva::constant_id<2>(plan.wgN),
                            eva::constant_id<3>(plan.sgM),
                            eva::constant_id<4>(plan.sgN),
                            eva::constant_id<5>(plan.accStoreBatchSize),
                            eva::constant_id<6>(plan.alignM),
                            eva::constant_id<7>(plan.alignN),
                            eva::constant_id<8>(plan.alignK),
                            eva::constant_id<9>(kPerSplit),
                            eva::constant_id<10>(plan.loadMode),
                            eva::constant_id<11>(plan.storeMode),
                            eva::constant_id<12>(plan.reorder),
                            eva::constant_id<13>(plan.swapXY),
                            eva::constant_id<14>(plan.directOutput),
                            eva::constant_id<15>(residual.enabled),
                            eva::constant_id<16>(residual.order),
                            eva::constant_id<SPEC_ID_TYPE_CI>(plan.ciSpec),
                            eva::constant_id<SPEC_ID_TYPE_N>(plan.nSpec),
                            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
                            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
                            eva::constant_id<SPEC_ID_TYPE_P>(paramType),
                            eva::constant_id<SPEC_ID_TYPE_ACC>(plan.accType),
                            eva::constant_id<SPEC_ID_EPILOGUE>(ep),
                            eva::constant_id<SPEC_ID_HAS_SCALE>(hasScale),
                            eva::constant_id<SPEC_ID_HAS_BIAS>(hasBias),
                        },
                        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline k1s1Pipeline(PipelinePool& pool, const K1S1Plan& plan, Dtype wDtype, Dtype paramDtype,
                                         uint32_t epilogue, uint32_t hasScale, uint32_t hasBias,
                                         ResidualFusion residual, Dtype iDtype, Dtype oDtype, uint32_t kPerSplit,
                                         uint32_t requiredSubgroupSize)
{
    const uint32_t wgSize =
        (plan.wgM / plan.tM) * (plan.wgN / 4u);
    return pool.request("conv-k1s1-gemm",
                        {
                            eva::constant_id<0>(wgSize),
                            eva::constant_id<1>(plan.wgM),
                            eva::constant_id<2>(plan.wgN),
                            eva::constant_id<3>(plan.wgK),
                            eva::constant_id<4>(plan.tM),
                            eva::constant_id<5>(plan.alignM),
                            eva::constant_id<6>(plan.alignN),
                            eva::constant_id<7>(plan.alignK),
                            eva::constant_id<8>(kPerSplit),
                            eva::constant_id<9>(residual.enabled),
                            eva::constant_id<10>(residual.order),
                            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
                            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
                            eva::constant_id<SPEC_ID_TYPE_W>(wDtype),
                            eva::constant_id<SPEC_ID_TYPE_P>(paramDtype),
                            eva::constant_id<SPEC_ID_EPILOGUE>(epilogue),
                            eva::constant_id<SPEC_ID_HAS_SCALE>(hasScale),
                            eva::constant_id<SPEC_ID_HAS_BIAS>(hasBias),
                        },
                        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline k3s2CMPipeline(PipelinePool& pool, const K3S2Plan& plan, uint32_t ep, Dtype paramType,
                                           uint32_t hasScale, uint32_t hasBias, ResidualFusion residual, Dtype iDtype,
                                           Dtype oDtype, uint32_t requiredSubgroupSize)
{
    VAI_ASSERT(plan.cooperativeMatrix());
    const bool packed = plan.packedWeights;
    eva::SpecializationConstant specialization{
        eva::constant_id<0>(plan.wgSize),        eva::constant_id<1>(plan.wgM),
        eva::constant_id<2>(plan.wgN),           eva::constant_id<3>(plan.sgM),
        eva::constant_id<4>(plan.sgN),           eva::constant_id<5>(plan.splitK),
        eva::constant_id<6>(plan.alignM),        eva::constant_id<7>(plan.alignN),
        eva::constant_id<8>(plan.alignK),        eva::constant_id<9>(packed),
        eva::constant_id<10>(plan.directOutput), eva::constant_id<11>(residual.enabled),
        eva::constant_id<12>(residual.order),
    };
    withShape(specialization, plan.shape, true);
    withDTypes(specialization, oDtype, iDtype, paramType, Dtype::Fp32, plan.accType, {.includeAcc = true});
    withFusion(specialization, ep, hasScale, hasBias);

    return pool.request(
        "conv-k3s2-gemm-cm", std::move(specialization),
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline k3s1ImplicitGemmPipeline(PipelinePool& pool, const K3S1Plan& plan, uint32_t ep,
                                                     Dtype paramType, uint32_t hasScale, uint32_t hasBias, Dtype iDtype,
                                                     Dtype oDtype, ResidualFusion residual, bool directOutput,
                                                     uint32_t requiredSubgroupSize)
{
    VAI_ASSERT(plan.algorithm == ConvAlgorithm::Implicit && requiredSubgroupSize == 32u);
    const bool rowAlignedFlatSpatial = plan.flatSpatial &&
        plan.shape.wo != 0u && (plan.shape.wo % plan.wgN) == 0u;
    eva::SpecializationConstant specialization{
        eva::constant_id<0>(plan.wgSize),      eva::constant_id<1>(plan.wgM),
        eva::constant_id<2>(plan.wgN),         eva::constant_id<3>(plan.sgM),
        eva::constant_id<4>(plan.sgN),         eva::constant_id<5>(rowAlignedFlatSpatial),
        eva::constant_id<6>(residual.enabled), eva::constant_id<7>(plan.flatSpatial),
        eva::constant_id<8>(plan.panelK),      eva::constant_id<9>(directOutput),
        eva::constant_id<10>(residual.order),
    };
    withShape(specialization, plan.shape, true);
    withDTypes(specialization, oDtype, iDtype, paramType, Dtype::Fp32, 0u, {});
    withFusion(specialization, ep, hasScale, hasBias);

    return pool.request(
        "conv-k3s1-implicit-gemm-cm", std::move(specialization),
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline ConvPipeline(PipelinePool& pool, const ImplicitGemmPlan& plan, const Workload& workload,
                                         uint32_t epilogue, Dtype paramType, uint32_t hasScale, uint32_t hasBias,
                                         ResidualFusion residual, bool k3s1Variant, uint32_t requiredSubgroupSize = 0u)
{
    VAI_ASSERT(!plan.cooperativeMatrix);
    const ShapeSpec shape = ShapeSpec::fromWorkload(workload);
    eva::SpecializationConstant specialization{
        eva::constant_id<0>(plan.wgSize),     eva::constant_id<1>(plan.wgM),    eva::constant_id<2>(plan.wgN),
        eva::constant_id<3>(plan.wgK),        eva::constant_id<4>(plan.tM),     eva::constant_id<5>(plan.alignM),
        eva::constant_id<6>(plan.alignN),     eva::constant_id<7>(plan.alignK), eva::constant_id<8>(residual.enabled),
        eva::constant_id<12>(residual.order),
    };
    if (k3s1Variant)
    {
        specialization.addConstant(eva::constant_id<9>(plan.structuredInputLoad));
        specialization.addConstant(eva::constant_id<10>(plan.padA));
        specialization.addConstant(eva::constant_id<11>(plan.wideFp32Tile));
    }
    withShape(specialization, shape, true);
    withDTypes(specialization, workload.oDtype, workload.iDtype, paramType,
               workload.wDtype, 0u, {.includeWeight = true});
    withFusion(specialization, epilogue, hasScale, hasBias);

    VAI_ASSERT(!plan.wideFp32Tile || k3s1Variant);
    const char* shader = k3s1Variant
        ? "conv-k3s1-implicit-gemm"
        : "conv-implicit-gemm";
    return pool.request(
        shader, std::move(specialization),
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline ConvCMPipeline(PipelinePool& pool, const ImplicitGemmPlan& plan, const Workload& workload,
                                           uint32_t epilogue, Dtype paramType, uint32_t hasScale, uint32_t hasBias,
                                           ResidualFusion residual)
{
    VAI_ASSERT(plan.cooperativeMatrix);
    const ShapeSpec shape = ShapeSpec::fromWorkload(workload);
    eva::SpecializationConstant specialization{
        eva::constant_id<0>(plan.wgSize),      eva::constant_id<1>(plan.wgM),       eva::constant_id<2>(plan.wgN),
        eva::constant_id<3>(plan.panelK),      eva::constant_id<4>(plan.sgM),       eva::constant_id<5>(plan.sgN),
        eva::constant_id<6>(residual.enabled), eva::constant_id<7>(residual.order),
    };
    withShape(specialization, shape, true);
    withDTypes(specialization, workload.oDtype, workload.iDtype, paramType,
               Dtype::Fp32, plan.accType, {.includeAcc = true});
    withFusion(specialization, epilogue, hasScale, hasBias);

    return pool.request("conv-implicit-gemm-cm", std::move(specialization), {.requiredSubgroupSize = 32u});
}

// fp16Direct selects the plain fp16 non-CM kernel (preset "direct"), which
// carries the full constant set regardless of the plan's split/shape state.
inline eva::ComputePipeline k3s2SpecializedPipeline(PipelinePool& pool, const K3S2Plan& plan, uint32_t ep,
                                                    Dtype paramType, uint32_t hasScale, uint32_t hasBias,
                                                    ResidualFusion residual, Dtype typeW, Dtype iDtype, Dtype oDtype,
                                                    uint32_t requiredSubgroupSize, bool fp16Direct = false)
{
    VAI_ASSERT(plan.valid() && !plan.cooperativeMatrix());
    enum class Preset : uint8_t { FirstLayer, Aligned, Generic, Split, Shape, Direct };
    struct PresetValues
    {
        bool includeSplitK;
        bool includeSpatialShape;
        bool includeFullShape;
        bool includeEpilogue;
    };
    static constexpr std::array<PresetValues, 6> presets{{
        {
            .includeSpatialShape = true,
            .includeEpilogue = true,
        },
        {
            .includeEpilogue = true,
        },
        {
            .includeEpilogue = true,
        },
        {
            .includeSplitK = true,
        },
        {
            .includeFullShape = true, .includeEpilogue = true,
        },
        {
            .includeSplitK = true,
            .includeEpilogue = true,
        },
    }};

    Preset preset;
    if (fp16Direct)
        preset = Preset::Direct;
    else if (plan.firstLayer())
        preset = Preset::FirstLayer;
    else if (plan.split())
        preset = Preset::Split;
    else if (!plan.shape.empty())
        preset = Preset::Shape;
    else if (plan.alignM && plan.alignN && plan.alignK)
        preset = Preset::Aligned;
    else
        preset = Preset::Generic;

    const PresetValues& values = presets[static_cast<size_t>(preset)];
    eva::SpecializationConstant specialization{
        eva::constant_id<0>(plan.wgSize),
        eva::constant_id<1>(plan.wgM),
        eva::constant_id<2>(plan.wgN),
        eva::constant_id<3>(plan.wgK),
        eva::constant_id<4>(plan.tM),
    };
    if (preset != Preset::Split)
    {
        specialization.addConstant(eva::constant_id<5>(plan.alignM));
        specialization.addConstant(eva::constant_id<7>(plan.alignN));
        specialization.addConstant(eva::constant_id<8>(plan.alignK));
    }
    if (values.includeSplitK)
        specialization.addConstant(eva::constant_id<6>(plan.splitK));
    if (values.includeSpatialShape || values.includeFullShape)
        withShape(specialization, plan.shape, values.includeFullShape);
    withDTypes(specialization, oDtype, iDtype, paramType, typeW, 0u, {.includeWeight = true});
    if (values.includeEpilogue)
        specialization.addConstant(eva::constant_id<SPEC_ID_EPILOGUE>(ep));
    specialization.addConstant(eva::constant_id<9>(residual.enabled));
    specialization.addConstant(eva::constant_id<10>(residual.order));
    specialization.addConstant(eva::constant_id<SPEC_ID_HAS_SCALE>(hasScale));
    specialization.addConstant(eva::constant_id<SPEC_ID_HAS_BIAS>(hasBias));

    return pool.request(
        "conv-k3s2-gemm", std::move(specialization),
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline cmTransformPipeline(PipelinePool& pool, uint32_t mode, uint32_t requiredSubgroupSize)
{
    return pool.request(
        "cm-transform",
        {eva::constant_id<0>(mode)},
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline winogradFilterTransformPipeline(
    PipelinePool& pool,
    Dtype typeW,
    bool reorder,
    uint32_t requiredSubgroupSize)
{
    return pool.request(
        "conv-k3s1-winograd-filter-transform",
        {
            eva::constant_id<3>(reorder),
            eva::constant_id<SPEC_ID_TYPE_W>(typeW),
        },
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline winogradInputTransformPipeline(
    PipelinePool& pool,
    Dtype typeW,
    Dtype iDtype,
    uint32_t requiredSubgroupSize,
    bool tileMajor = false)
{
    return pool.request(
        "conv-k3s1-winograd-input",
        {
            eva::constant_id<3>(tileMajor),
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
            eva::constant_id<SPEC_ID_TYPE_W>(typeW),
        },
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline winogradOutputPipeline(PipelinePool& pool, bool storeMode, Dtype typeW, Dtype paramType,
                                                   uint32_t ep, ResidualFusion residual, uint32_t hasScale,
                                                   uint32_t hasBias, Dtype iDtype, Dtype oDtype,
                                                   uint32_t requiredSubgroupSize)
{
    const uint32_t localSize = storeMode ? 128u : 256u;
    return pool.request("conv-k3s1-winograd-output",
                        {
                            eva::constant_id<0>(localSize),
                            eva::constant_id<3>(residual.enabled),
                            eva::constant_id<4>(storeMode),
                            eva::constant_id<5>(residual.order),
                            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
                            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
                            eva::constant_id<SPEC_ID_TYPE_W>(typeW),
                            eva::constant_id<SPEC_ID_TYPE_P>(paramType),
                            eva::constant_id<SPEC_ID_EPILOGUE>(ep),
                            eva::constant_id<SPEC_ID_HAS_SCALE>(hasScale),
                            eva::constant_id<SPEC_ID_HAS_BIAS>(hasBias),
                        },
                        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline bnReducePipeline(PipelinePool& pool, Dtype typeW, Dtype typeP, uint32_t hasScale,
                                             uint32_t hasBias, uint32_t ep, ResidualFusion residual,
                                             Dtype residualDtype, bool aligned4, Dtype oDtype, bool reorder = false)
{
    eva::SpecializationConstant specialization{
        eva::constant_id<3>(aligned4),
        eva::constant_id<4>(reorder),
        eva::constant_id<5>(residual.enabled),
        eva::constant_id<6>(residual.order),
    };
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_O>(oDtype));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_I>(residualDtype));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_W>(typeW));
    specialization.addConstant(eva::constant_id<SPEC_ID_TYPE_P>(typeP));
    withFusion(specialization, ep, hasScale, hasBias);

    return pool.request("bn-reduce", std::move(specialization));
}

inline eva::ComputePipeline winogradPipeline(
    PipelinePool& pool,
    const K3S1Plan& plan,
    Dtype typeW,
    uint32_t requiredSubgroupSize)
{
    return pool.request(
        "conv-k3s1-winograd-gemm",
        {
            eva::constant_id<0>(plan.wgSize),
            eva::constant_id<1>(plan.wgM),
            eva::constant_id<2>(plan.wgN),
            eva::constant_id<3>(plan.wgK),
            eva::constant_id<4>(plan.tM),
            eva::constant_id<5>(plan.tN),
            eva::constant_id<6>(plan.pad),
            eva::constant_id<7>(plan.alignM),
            eva::constant_id<8>(plan.alignN),
            eva::constant_id<9>(plan.alignK),
            eva::constant_id<10>(plan.storeMode),
            eva::constant_id<11>(plan.loadMode),
            eva::constant_id<SPEC_ID_TYPE_W>(typeW),
        },
        {.requiredSubgroupSize = requiredSubgroupSize});
}

inline eva::ComputePipeline winogradCMPipeline(
    PipelinePool& pool,
    const K3S1Plan& plan,
    uint32_t wgN,
    uint32_t columnOffset,
    uint32_t kPerSplit,
    bool nvidiaMode)
{
    VAI_ASSERT(validCMTopology(plan.wgSize, plan.wgM, wgN, plan.sgM, wgN, plan.sgSize));
    const uint32_t loadMode = nvidiaMode ? 0u : plan.loadMode;
    const bool storeMode = nvidiaMode || plan.storeMode;
    const bool reorder = nvidiaMode || plan.reorder;
    const bool swapXY = nvidiaMode && plan.swapXY;
    eva::SpecializationConstant specialization{
        eva::constant_id<0>(plan.wgSize),
        eva::constant_id<1>(plan.wgM),
        eva::constant_id<2>(wgN),
        eva::constant_id<3>(plan.sgM),
        eva::constant_id<4>(columnOffset),
        eva::constant_id<5>(kPerSplit),
        eva::constant_id<6>(loadMode),
        eva::constant_id<7>(storeMode),
        eva::constant_id<8>(reorder),
        eva::constant_id<9>(swapXY),
        eva::constant_id<10>(nvidiaMode),
        eva::constant_id<12>(wgN),
        eva::constant_id<SPEC_ID_TYPE_ACC>(plan.accType),
    };
    if (!nvidiaMode)
    {
        specialization.addConstant(eva::constant_id<11>(plan.panelK));
        specialization.addConstant(eva::constant_id<13>(plan.inputReorder));
    }

    return pool.request(
        "conv-k3s1-winograd-gemm-cm",
        std::move(specialization),
        {.requiredSubgroupSize = plan.sgSize});
}

struct ConvLaunch
{
    enum Feature : uint8_t
    {
        CooperativeMatrix = 1u << 0,
    };

    std::array<uint32_t, 3> groupCount{1u, 1u, 1u};
    uint32_t sharedMemoryBytes = 0u;
    uint32_t requiredSubgroupSize = 0u;
    uint8_t requiredFeatures = 0u;
    bool dispatchRepresentable = true;
};

constexpr uint8_t kConvCMFeatures = ConvLaunch::CooperativeMatrix;

inline uint32_t convWorkgroupCount(uint64_t workItems, uint32_t localSize)
{
    VAI_ASSERT(localSize != 0u);
    const uint64_t count =
        workItems / localSize + (workItems % localSize != 0u);
    return count > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(count);
}

struct K3S2DispatchGeometry
{
    uint32_t spatialGroups;
    uint32_t outputChannelGroups;
    uint32_t splitK;
};

inline K3S2DispatchGeometry deriveK3S2DispatchGeometry(const K3S2Plan& plan, const Workload& workload)
{
    return {
        convWorkgroupCount(workload.spatialSize(), plan.wgN),
        convWorkgroupCount(workload.co, plan.wgM),
        plan.splitK,
    };
}

inline uint32_t k1s1SharedMemoryBytes(
    uint32_t wgM, uint32_t wgN, uint32_t wgK)
{
    return 8u * wgK * (wgM + 4u + wgN);
}

inline uint32_t k1s1CMSharedMemoryBytes(const K1S1Plan& plan)
{
    const uint32_t subgroupCount = (plan.wgM / plan.sgM) * (plan.wgN / plan.sgN);
    const uint32_t mIters = plan.sgM / 16u;
    const uint32_t nIters = plan.sgN / 16u;
    const uint32_t fp16AccElements = subgroupCount * mIters * nIters * 16u * 16u;
    const uint32_t fp32AccElements = subgroupCount * plan.accStoreBatchSize * 16u * 16u;
    const uint32_t inputElements = subgroupCount * 16u * nIters * 16u;
    const uint32_t fp32Bytes = sizeof(float) * (plan.accType == 1u && !plan.directOutput ? fp32AccElements : 16u * 16u);
    const uint32_t fp16Bytes = sizeof(uint16_t) * (plan.accType == 0u ? fp16AccElements : 1u);
    const uint32_t inputBytes = plan.loadMode != 0u ? sizeof(uint16_t) * inputElements : sizeof(uint16_t) * 4u;
    return fp32Bytes + fp16Bytes + inputBytes;
}

inline uint32_t k3s2SharedMemoryBytes(const K3S2Plan& plan, Dtype weightDtype)
{
    VAI_ASSERT(plan.valid() && !plan.cooperativeMatrix());
    const uint32_t aElements = plan.wgK * (plan.wgM + 4u);
    const uint32_t bElements = plan.wgK * plan.wgN;
    const uint32_t elements = aElements + bElements;
    const uint32_t tileBuffers = plan.split() ? 1u : (weightDtype == Dtype::Fp16 && plan.firstLayer() ? 1u : 2u);
    const uint32_t fp32Elements = ((weightDtype == Dtype::Fp32) ? tileBuffers : 1u) * elements + (plan.split() && (weightDtype == Dtype::Fp32) ? 2u * elements : 1u);
    const uint32_t fp16Elements = ((weightDtype == Dtype::Fp16) ? tileBuffers : 1u) * elements + (plan.split() && (weightDtype == Dtype::Fp16) ? 2u * elements : 1u);
    const uint32_t bytes = sizeof(float) * fp32Elements + sizeof(uint16_t) * fp16Elements;
    return (bytes + 3u) & ~3u;
}

inline uint32_t k3s2CMSharedMemoryBytes(const K3S2Plan& plan)
{
    VAI_ASSERT(plan.cooperativeMatrix());
    return plan.packedWeights
        ? 64u * (plan.wgN + 8u) + sizeof(uint16_t) * (plan.directOutput ? 1u : plan.wgM * plan.wgN)
        : 32u * (plan.wgN + plan.wgSize);
}

inline bool validConvLaunch(const ConvLaunch& launch, const eva::Device& device, const ConvolutionCaps& caps)
{
    if (!launch.dispatchRepresentable)
        return false;

    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (launch.groupCount[axis] > device.maxComputeWorkGroupCount()[axis])
            return false;
    }
    if (launch.sharedMemoryBytes > device.maxComputeSharedMemorySize())
        return false;
    if (launch.requiredSubgroupSize != 0u &&
        (launch.requiredSubgroupSize < device.minSubgroupSize() ||
         launch.requiredSubgroupSize > device.maxSubgroupSize()))
        return false;

    return (!(launch.requiredFeatures & ConvLaunch::CooperativeMatrix) ||
            enableCM(device, caps));
}

inline ConvLaunch deriveConvLaunch(const ConvPlan& convPlan, const Workload& workload)
{
    const uint32_t Co = workload.co;
    const uint32_t Ho = workload.ho;
    const uint32_t Wo = workload.wo;
    const uint64_t N = uint64_t(Ho) * Wo;
    ConvLaunch launch;

    if (convPlan.path == ConvPlan::Path::K1S1)
    {
        if (!convPlan.useCM)
        {
            const bool split = convPlan.k1s1.splitK > 1u;
            const K1S1Plan& plan = convPlan.k1s1;
            const uint32_t requiredSubgroupSize = split
                ? convPlan.splitRequiredSubgroupSize
                : convPlan.requiredSubgroupSize;
            launch = {
                .groupCount = {
                    convWorkgroupCount(N, plan.wgN),
                    convWorkgroupCount(Co, plan.wgM), plan.splitK},
                .sharedMemoryBytes = k1s1SharedMemoryBytes(
                    plan.wgM, plan.wgN, plan.wgK),
                .requiredSubgroupSize = requiredSubgroupSize,
            };
        }
        else
        {
            const K1S1Plan& plan = convPlan.k1s1;
            uint32_t gx = convWorkgroupCount(N, plan.wgN);
            uint32_t gy = convWorkgroupCount(Co, plan.wgM);
            if (plan.swapXY != 0u)
                std::swap(gx, gy);
            launch = {
                .groupCount = {gx, gy, plan.splitK},
                .sharedMemoryBytes = k1s1CMSharedMemoryBytes(plan),
                .requiredSubgroupSize = plan.sgSize,
                .requiredFeatures = kConvCMFeatures,
            };
        }
        return launch;
    }
    else if (convPlan.path == ConvPlan::Path::K3S1)
    {
        const K3S1Plan& plan = convPlan.k3s1;
        if (plan.algorithm == ConvAlgorithm::Implicit)
        {
            if (convPlan.useCM)
            {
                launch = {
                    .groupCount = {
                        plan.flatSpatial
                            ? convWorkgroupCount(N, plan.wgN)
                            : Ho * convWorkgroupCount(Wo, plan.wgN),
                        convWorkgroupCount(Co, plan.wgM), 1u},
                    .sharedMemoryBytes = k3s1SharedMemoryBytes(
                        plan, plan.wgN, convPlan.wDtype),
                    .requiredSubgroupSize = 32u,
                    .requiredFeatures = kConvCMFeatures,
                };
            }
            else
            {
                const ImplicitGemmPlan& implicitPlan = convPlan.implicit;
                VAI_ASSERT(!implicitPlan.cooperativeMatrix);
                const uint32_t requiredSubgroupSize =
                    convPlan.requiredSubgroupSize;
                launch = {
                    .groupCount = {
                        convWorkgroupCount(N, implicitPlan.wgN),
                        convWorkgroupCount(Co, implicitPlan.wgM), 1u},
                    .sharedMemoryBytes = convSharedMemoryBytes(
                        implicitPlan, workload.wDtype),
                    .requiredSubgroupSize = requiredSubgroupSize,
                };
            }
            return launch;
        }
        else if (plan.algorithm == ConvAlgorithm::Winograd)
        {
            const uint32_t sharedMemoryBytes = k3s1SharedMemoryBytes(plan, plan.wgN, convPlan.wDtype);
            const uint64_t numTiles =
                (uint64_t(Wo) / 4u + (Wo % 4u != 0u)) *
                (uint64_t(Ho) / 4u + (Ho % 4u != 0u));
            if (convPlan.useCM)
            {
                if (convPlan.vendorId == eva::VENDOR_ID::AMD)
                {
                    const uint64_t mainTiles = plan.hasTail() ? plan.tailNOffset : numTiles;
                    launch = {
                        .groupCount = {
                            convWorkgroupCount(mainTiles, plan.wgN),
                            convWorkgroupCount(Co, plan.wgM), 36u},
                        .sharedMemoryBytes = sharedMemoryBytes,
                        .requiredSubgroupSize = plan.sgSize,
                        .requiredFeatures = kConvCMFeatures,
                    };
                    return launch;
                }
                else if (convPlan.vendorId == eva::VENDOR_ID::NVIDIA)
                {
                    uint32_t gx = plan.hasTail() ? convWorkgroupCount(numTiles / plan.wgN, 1u) : convWorkgroupCount(numTiles, plan.wgN);
                    uint32_t gy = convWorkgroupCount(Co, plan.wgM);
                    if (plan.swapXY)
                        std::swap(gx, gy);
                    launch = {
                        .groupCount = {gx, gy, 36u * plan.splitK},
                        .sharedMemoryBytes = sharedMemoryBytes,
                        .requiredSubgroupSize = plan.sgSize,
                        .requiredFeatures = kConvCMFeatures,
                    };
                    return launch;
                }
                else
                {
                    VAI_ASSERT(false && "K3S1 Winograd CM requires an AMD or NVIDIA plan");
                }
            }
            else
            {
                launch = {
                    .groupCount = {
                        convWorkgroupCount(numTiles, plan.wgN),
                        convWorkgroupCount(Co, plan.wgM), 36u},
                    .sharedMemoryBytes = sharedMemoryBytes,
                    .requiredSubgroupSize = convPlan.requiredSubgroupSize,
                };
                return launch;
            }
        }

        VAI_ASSERT(false && "invalid K3S1 algorithm");
        return launch;
    }
    else if (convPlan.path == ConvPlan::Path::K3S2)
    {
        const K3S2Plan& plan = convPlan.k3s2;
        const bool cm = convPlan.useCM;
        const K3S2DispatchGeometry dispatchGeometry = deriveK3S2DispatchGeometry(plan, workload);
        if (!cm)
        {
            launch = {
                .groupCount = {dispatchGeometry.spatialGroups, dispatchGeometry.outputChannelGroups, dispatchGeometry.splitK},
                .sharedMemoryBytes = k3s2SharedMemoryBytes(plan, convPlan.wDtype),
                .requiredSubgroupSize = convPlan.requiredSubgroupSize,
            };
            return launch;
        }

        launch = {
            .groupCount = {dispatchGeometry.spatialGroups, dispatchGeometry.outputChannelGroups, dispatchGeometry.splitK},
            .sharedMemoryBytes = k3s2CMSharedMemoryBytes(plan),
            .requiredSubgroupSize = 32u,
            .requiredFeatures = kConvCMFeatures,
        };
        return launch;
    }
    else if (convPlan.path == ConvPlan::Path::General)
    {
        const ImplicitGemmPlan& plan = convPlan.implicit;
        const uint8_t cmFeatures = convPlan.useCM
            ? uint8_t(ConvLaunch::CooperativeMatrix)
            : 0u;
        launch = {
            .groupCount = {
                convWorkgroupCount(N, plan.wgN),
                convWorkgroupCount(Co, plan.wgM), 1u},
            .sharedMemoryBytes = convSharedMemoryBytes(plan, workload.wDtype),
            .requiredSubgroupSize = convPlan.useCM ? 32u : 0u,
            .requiredFeatures = cmFeatures,
        };
        return launch;
    }

    VAI_ASSERT(false && "invalid convolution plan path");
    return launch;
}

inline ConvLaunch convThreadLaunch(
    uint32_t localSize,
    uint64_t workItems,
    uint8_t features,
    uint32_t requiredSubgroupSize = 0u)
{
    return {
        .groupCount = {convWorkgroupCount(workItems, localSize), 1u, 1u},
        .requiredSubgroupSize = requiredSubgroupSize,
        .requiredFeatures = features,
        .dispatchRepresentable = workItems <= UINT32_MAX,
    };
}

inline void appendThreadRequirement(
    std::vector<ConvLaunch>& requirements,
    uint32_t localSize,
    uint64_t workItems,
    uint8_t features,
    uint32_t requiredSubgroupSize = 0u)
{
    requirements.push_back(convThreadLaunch(
        localSize, workItems, features, requiredSubgroupSize));
}

inline void appendReductionRequirement(
    std::vector<ConvLaunch>& requirements, uint64_t workItems)
{
    appendThreadRequirement(requirements, 256u, workItems, 0u);
}

inline void appendTransformRequirement(
    std::vector<ConvLaunch>& requirements,
    uint64_t workItems,
    uint32_t subgroupSize)
{
    appendThreadRequirement(
        requirements, 128u, workItems, 0u, subgroupSize);
}

// AMD launches the tail grid on the tile remainder; NVIDIA folds the tail
// into a single swapXY-aware spatial group.
inline ConvLaunch k3s1WinogradTailRequirement(
    const K3S1Plan& plan, uint32_t vendorId, Dtype weightDtype, uint64_t tiles, uint64_t co)
{
    uint32_t gx = 1u;
    uint32_t gy = convWorkgroupCount(co, plan.wgM);
    if (vendorId == eva::VENDOR_ID::AMD)
        gx = convWorkgroupCount(tiles - plan.tailNOffset, kK3S1TailWgN);
    else if (plan.swapXY)
        std::swap(gx, gy);
    return {
        .groupCount = {gx, gy, 36u},
        .sharedMemoryBytes = k3s1SharedMemoryBytes(
            plan, kK3S1TailWgN, weightDtype),
        .requiredSubgroupSize = plan.sgSize,
        .requiredFeatures = kConvCMFeatures,
    };
}

inline std::vector<ConvLaunch> deriveConvLaunchRequirements(const ConvPlan& convPlan, const Workload& workload)
{
    const uint64_t Co = workload.co;
    const uint64_t Ci = workload.ci;
    const uint64_t N = uint64_t(workload.ho) * workload.wo;
    std::vector<ConvLaunch> requirements;
    requirements.push_back(deriveConvLaunch(convPlan, workload));

    if (convPlan.path == ConvPlan::Path::General)
    {
        if (convPlan.useCM)
        {
            const uint64_t paddedM = ((Co + 15u) / 16u) * 16u;
            const uint64_t rawK = Ci * workload.kernel * workload.kernel;
            const uint64_t paddedK = ((rawK + 15u) / 16u) * 16u;
            appendTransformRequirement(requirements, paddedM * paddedK, 32u);
        }
        return requirements;
    }

    if (convPlan.path == ConvPlan::Path::K1S1)
    {
        const K1S1Plan& plan = convPlan.k1s1;
        if (!convPlan.useCM)
        {
            if (plan.splitK > 1u)
            {
                const uint64_t outputElements = Co * N;
                appendReductionRequirement(
                    requirements,
                    outputElements / 4u +
                    (outputElements % 4u != 0u));
            }
            return requirements;
        }

        if (plan.reorder)
            appendTransformRequirement(requirements, Co * Ci, plan.sgSize);
        if (plan.splitK > 1u)
            appendReductionRequirement(requirements, Co * N / 4u);
        return requirements;
    }

    if (convPlan.path == ConvPlan::Path::K3S2)
    {
        const K3S2Plan& plan = convPlan.k3s2;
        if (plan.packedWeights)
            appendTransformRequirement(requirements, 9u * Ci * Co, 32u);
        if (plan.split())
        {
            const uint64_t outputElements = Co * N;
            const bool fp16Partial =
                plan.packedWeights ||
                (convPlan.useCM && workload.oDtype == Dtype::Fp16);
            appendReductionRequirement(requirements,
                fp16Partial
                    ? outputElements / 4u
                    : outputElements / 4u + (outputElements % 4u != 0u));
        }
        return requirements;
    }

    VAI_ASSERT(convPlan.path == ConvPlan::Path::K3S1);
    if (convPlan.algorithm() == ConvAlgorithm::Implicit)
    {
        if (convPlan.useCM)
            appendTransformRequirement(requirements, 9u * Co * Ci, 32u);
        return requirements;
    }

    const uint64_t tiles =
        (uint64_t(workload.wo) / 4u + (workload.wo % 4u != 0u)) *
        (uint64_t(workload.ho) / 4u + (workload.ho % 4u != 0u));
    const K3S1Plan& plan = convPlan.k3s1;
    uint32_t auxiliarySubgroupSize = convPlan.requiredSubgroupSize;
    if (convPlan.useCM)
    {
        auxiliarySubgroupSize =
            convPlan.vendorId == eva::VENDOR_ID::AMD
            ? plan.sgSize
            : 32u;
    }
    appendThreadRequirement(
        requirements,
        256u, Co * Ci,
        0u,
        auxiliarySubgroupSize);

    uint64_t inputTiles = tiles;
    if (convPlan.useCM)
    {
        if (convPlan.vendorId == eva::VENDOR_ID::AMD &&
            plan.inputReorder)
            inputTiles = ((tiles + 15u) / 16u) * 16u;
    }
    const uint64_t transformedWeightElements = 36u * Co * Ci;
    appendThreadRequirement(
        requirements,
        128u, Ci * inputTiles, 0u,
        auxiliarySubgroupSize);

    const bool outputStoreMode = plan.outputStoreMode;
    if (convPlan.useCM)
    {
        if (plan.hasTail())
            requirements.push_back(
                k3s1WinogradTailRequirement(
                    plan, convPlan.vendorId, convPlan.wDtype, tiles, Co));
        if (convPlan.vendorId == eva::VENDOR_ID::NVIDIA)
        {
            appendTransformRequirement(
                requirements, transformedWeightElements, 32u);
            if (plan.splitK > 1u)
            {
                const uint64_t reductionElements = Co * tiles;
                appendReductionRequirement(
                    requirements, 36u * (reductionElements / 4u));
            }
        }
    }

    appendThreadRequirement(
        requirements,
        outputStoreMode ? 128u : 256u,
        Co * tiles, 0u,
        auxiliarySubgroupSize);
    return requirements;
}

inline bool validConvLaunches(
    const ConvPlan& convPlan,
    const Workload& workload,
    const eva::Device& device,
    const ConvolutionCaps& caps)
{
    const std::vector<ConvLaunch> requirements =
        deriveConvLaunchRequirements(convPlan, workload);
    return std::all_of(requirements.begin(), requirements.end(), [&](ConvLaunch requirement) {
        // The images stack on z, outermost.
        const uint64_t z = uint64_t(requirement.groupCount[2]) * workload.batch;
        requirement.groupCount[2] = static_cast<uint32_t>(std::min<uint64_t>(z, UINT32_MAX));
        requirement.dispatchRepresentable &= z <= UINT32_MAX;
        return validConvLaunch(requirement, device, caps);
    });
}

/////////////////////////////////////////////////////////////////////////////////////////
// Convolution node lifecycle
/////////////////////////////////////////////////////////////////////////////////////////

inline Convolution::Convolution(Config config)
: config_(config)
{
    VAI_ASSERT(config_.inChannels != 0u && config_.outChannels != 0u);
    VAI_ASSERT(config_.kernelSize != 0u && config_.kernelSize % 2u != 0u);
    VAI_ASSERT(config_.stride != 0u);
    VAI_ASSERT(config_.epilogue < Epilogue::Max);
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    addSlot("bias", NodeSlot::persistent);
    addSlot("scale", NodeSlot::persistent);
    if (config_.acceptResidual)
        addSlot("residual", NodeSlot::input);

    // Path-specific scratch tensors are assigned only when the selected plan
    // needs them. Registering the slots here keeps slot ownership in the
    // constructor without allocating inactive storage.
    addSlot("winV", NodeSlot::temporary);
    addSlot("winM", NodeSlot::temporary);
    addSlot("split", NodeSlot::temporary);
}

inline Workload Convolution::makeWorkload()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& input = (*this)["in0"];
    const std::vector<uint32_t>& inputShape = input.shape();
    const size_t rank = inputShape.size();
    VAI_ASSERT(rank >= 3u);
    uint32_t batch = 1u;
    for (size_t axis = 0; axis + 3u < rank; ++axis)
        batch *= inputShape[axis];
    const uint32_t kernel = config_.kernelSize;
    const uint32_t stride = config_.stride;
    const uint32_t inputHeight = inputShape[rank - 2u];
    const uint32_t inputWidth = inputShape[rank - 1u];
    VAI_ASSERT(batch != 0u && inputHeight != 0u && inputWidth != 0u);

    return {
        .ci = config_.inChannels,
        .co = config_.outChannels,
        .kernel = kernel,
        .stride = stride,
        .hi = inputHeight,
        .wi = inputWidth,
        // Odd kernels use symmetric same-padding, so the numerator reduces
        // to inputSize - 1 without an overflow-prone padding expression.
        .ho = (inputHeight - 1u) / stride + 1u,
        .wo = (inputWidth - 1u) / stride + 1u,
        .batch = batch,
        .iDtype = input.dtype(),
        .oDtype = oDtype,
        .wDtype = (*this)["weight"].dtype(),
        .fusion = hasScale() ? ConvolutionFusion::Scale : ConvolutionFusion::Bias,
        .residual = config_.acceptResidual,
    };
}

inline void Convolution::validateInputs(const Workload& workload)
{
    const Tensor& input = (*this)["in0"];
    const Tensor& weight = (*this)["weight"];
    VAI_ASSERT(workload.wDtype == Dtype::Fp32 || workload.wDtype == Dtype::Fp16);
    VAI_ASSERT(workload.wDtype == Dtype::Fp16 || workload.oDtype != Dtype::Fp16);
    VAI_ASSERT(workload.wDtype == Dtype::Fp16 || workload.iDtype != Dtype::Fp16);   // fp16 activations need fp16 weights
    VAI_ASSERT(input.shape()[input.shape().size() - 3u] == workload.ci);
    VAI_ASSERT(weight.isShapeOf(workload.co, workload.ci, workload.kernel, workload.kernel));

    if (hasBias())
    {
        const Tensor& bias = (*this)["bias"];
        VAI_ASSERT(bias.dtype() == workload.wDtype && bias.isShapeOf(workload.co));
    }
    if (hasScale())
    {
        const Tensor& scale = (*this)["scale"];
        VAI_ASSERT(scale.dtype() == workload.wDtype && scale.isShapeOf(workload.co));
    }
}

inline K1S1Plan selectK1S1TiledPlan(const eva::Device& device, const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const uint32_t computeBlockCount = device.computeBlockCount();
    const uint32_t Ci = workload.ci;
    const uint32_t Co = workload.co;
    const uint32_t spatialSize = workload.spatialSize();
    const uint32_t tiledWgM = vendorId == eva::VENDOR_ID::AMD ? 128u : 64u;
    K1S1Plan selectedPlan;
    if (const K1S1ExactRule* rule = findK1S1ExactRule(device, workload))
    {
        selectedPlan = rule->plan;
        selectedPlan.alignM = (Co % selectedPlan.wgM) == 0u;
        selectedPlan.alignN = (spatialSize % selectedPlan.wgN) == 0u;
        selectedPlan.alignK = (Ci % selectedPlan.wgK) == 0u;
        return selectedPlan;
    }

    if (workload.fusion == ConvolutionFusion::Scale &&
        Ci >= ((vendorId == eva::VENDOR_ID::AMD ||
                vendorId == eva::VENDOR_ID::NVIDIA) ? 256u : 1024u))
    {
        const uint32_t splitK = selectK1S1SplitK(device, workload);
        uint32_t thresholdBM = 0u;
        uint32_t baseSplitKThreshold = 0u;
        if (workload.wDtype == Dtype::Fp32)
        {
            thresholdBM = 128u;
            baseSplitKThreshold = vendorId == eva::VENDOR_ID::AMD || vendorId == eva::VENDOR_ID::NVIDIA ? 512u : 384u;
        }
        else if (workload.wDtype == Dtype::Fp16)
        {
            thresholdBM = tiledWgM;
            baseSplitKThreshold = 120u;
        }
        else
        {
            VAI_ASSERT(false && "unsupported K1S1 Split-K weight dtype");
        }
        const uint32_t splitKThreshold = scaleWorkSupplyThreshold(baseSplitKThreshold, computeBlockCount);
        if (splitK > 1u && CEIL_DIV(spatialSize, 64u) * CEIL_DIV(Co, thresholdBM) < splitKThreshold)
            selectedPlan.splitK = splitK;
    }

    selectedPlan.wgM = selectedPlan.splitK > 1u         ? 128u
                       : workload.wDtype == Dtype::Fp32 ? 64u
                                                        : tiledWgM;
    selectedPlan.wgN = 64u;
    selectedPlan.wgK = selectedPlan.splitK > 1u && vendorId == eva::VENDOR_ID::AMD ? 16u : 8u;
    selectedPlan.tM = selectedPlan.wgM / 16u;
    const uint32_t kPerSplit = CEIL_DIV(Ci, selectedPlan.splitK);
    selectedPlan.alignM = (Co % selectedPlan.wgM) == 0u;
    selectedPlan.alignN = (spatialSize % selectedPlan.wgN) == 0u;
    selectedPlan.alignK = (Ci % selectedPlan.splitK) == 0u && (kPerSplit % selectedPlan.wgK) == 0u;
    if ((!selectedPlan.alignM || !selectedPlan.alignN || !selectedPlan.alignK) &&
        selectedPlan.wgK == 16u)
    {
        selectedPlan.wgK = 8u;
        selectedPlan.alignK = (Ci % selectedPlan.splitK) == 0u && (kPerSplit % selectedPlan.wgK) == 0u;
    }
    return selectedPlan;
}

inline K1S1Plan selectK1S1CMPlan(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const AMDArchitecture architecture =
        classifyAMDArchitecture(vendorId, device.deviceID());
    const uint32_t computeBlockCount = device.computeBlockCount();
    const Dtype acc = selectAccType(device, caps);
    const uint32_t ci = workload.ci;
    const uint32_t co = workload.co;
    const uint32_t n = workload.spatialSize();
    const bool alignedN = (n % 16u) == 0u;
    const bool storeAlignedN = (n % 4u) == 0u;
    K1S1Plan cmPlan;

    if (vendorId == eva::VENDOR_ID::NVIDIA)
    {
        // -----------------------------------------------------
        // NVIDIA
        // -----------------------------------------------------
        const uint32_t spatialTileElements = (n % 32u == 0u) ? 32u : 16u;
        const ConvDispatchFeatures baseFeatures =
            makeConvDispatchFeatures(ci, 16u, co, 16u, n, spatialTileElements);

        constexpr uint32_t minReductionTilesForTwoSubgroups = 18u;
        constexpr uint32_t mediumReductionTiles = 32u;
        constexpr uint32_t deepReductionTiles = 48u;
        const uint32_t maxWgMFragments = baseFeatures.R >= deepReductionTiles     ? 8u
                                         : baseFeatures.R >= mediumReductionTiles ? 6u
                                                                                  : 4u;
        const uint32_t subgroupCountM = std::min(baseFeatures.M, baseFeatures.R >= minReductionTilesForTwoSubgroups ? 2u : 1u);
        const uint32_t targetWorkgroups = scaleWorkSupplyThreshold(128u, computeBlockCount);

        uint32_t wgMFragments = 1u;
        for (uint32_t fragments = maxWgMFragments; fragments >= subgroupCountM; --fragments)
        {
            if ((fragments % subgroupCountM) != 0u || fragments > baseFeatures.M || (baseFeatures.M % fragments) != 0u)
                continue;

            if ((baseFeatures.M / fragments) * baseFeatures.P >= targetWorkgroups)
            {
                wgMFragments = fragments;
                break;
            }
        }

        cmPlan = {
            .sgSize = 32u,
            .wgM = wgMFragments * 16u,
            .wgN = n >= 32u ? 32u : 16u,
            .sgM = wgMFragments * 16u / subgroupCountM,
            .sgN = n >= 32u ? 32u : 16u,
            .loadMode = workload.iDtype == Dtype::Fp16 ? 0u : 1u,
            .storeMode = storeAlignedN,
            .accType = acc == Dtype::Fp16 ? 0u : 1u,
            .reorder = true,
            .splitK = 1u,
            .swapXY = 1u,
        };

        if (acc == Dtype::Fp16 &&
            workload.fusion == ConvolutionFusion::Scale && alignedN)
        {
            constexpr uint32_t minOutputTilesForReduction = 8u;
            constexpr uint32_t minReductionTilesPerSplit = 8u;
            const uint64_t unsplitWorkgroups = makeConvDispatchFeatures(ci, 16u, co, wgMFragments * 16u, n, cmPlan.wgN).G;
            const uint32_t splitWorkgroupThreshold = scaleWorkSupplyThreshold(1024u, computeBlockCount);

            if (baseFeatures.M >= minOutputTilesForReduction && unsplitWorkgroups < splitWorkgroupThreshold)
            {
                if (baseFeatures.R >= 4u * minReductionTilesPerSplit)
                    cmPlan.splitK = 4u;
                else if (baseFeatures.R >= 2u * minReductionTilesPerSplit)
                    cmPlan.splitK = 2u;
            }
        }

        if (!alignedN)
        {
            cmPlan.wgN = 16u;
            cmPlan.sgN = 16u;
            cmPlan.loadMode = 1u;
            cmPlan.splitK = 1u;
        }
        else if ((n % cmPlan.wgN) != 0u)
        {
            cmPlan.wgN = 16u;
            cmPlan.sgN = 16u;
        }

        if (acc == Dtype::Fp32 && !alignedN)
        {
            cmPlan.wgM = 16u;
            cmPlan.wgN = 16u;
            cmPlan.sgM = 16u;
            cmPlan.sgN = 16u;
            cmPlan.accStoreBatchSize = 1u;
            cmPlan.loadMode = 1u;
            cmPlan.splitK = 1u;
        }

        cmPlan.accStoreBatchSize = (cmPlan.sgM / 16u) * (cmPlan.sgN / 16u);
        cmPlan.alignM = (co % cmPlan.wgM) == 0u;
        cmPlan.alignN = (n % cmPlan.wgN) == 0u;
        cmPlan.alignK = (ci % cmPlan.wgK) == 0u;
    }
    else if (vendorId == eva::VENDOR_ID::AMD)
    {
        // -----------------------------------------------------
        // AMD
        // -----------------------------------------------------
        const auto finalizePlan = [&](K1S1Plan candidate)
        {
            // CM operands are FP16, so FP32 activations must be staged and
            // converted even when the geometry came from an FP16-I/O exact rule.
            if (workload.iDtype == Dtype::Fp32)
                candidate.loadMode = 1u;
            candidate.alignM = (co % candidate.wgM) == 0u;
            candidate.alignN = (n % candidate.wgN) == 0u;
            candidate.alignK = (ci % candidate.wgK) == 0u;
            if (candidate.alignM && candidate.alignN && candidate.splitK == 1u)
            {
                const uint32_t fragmentsPerSubgroup = (candidate.sgM / 16u) * (candidate.sgN / 16u);
                candidate.directOutput = fragmentsPerSubgroup >= 4u;
            }
            return candidate;
        };

        if (const K1S1CMExactRule* rule = findK1S1CMExactRule(device, workload))
        {
            cmPlan = finalizePlan(rule->plan);
        }
        else if (architecture == AMDArchitecture::RDNA3 && (workload.iDtype == Dtype::Fp16 || computeBlockCount <= 16u))
        {
            // Small RDNA3 GPUs need the extra M
            // workgroups from this plan. Larger GPUs reuse
            // staged fp32 input more efficiently below.
            uint32_t wgM = 16u;
            uint32_t subgroupM = 1u;
            if ((co % 64u) == 0u)
            {
                wgM = 64u;
                subgroupM = 2u;
            }
            else if ((co % 32u) == 0u)
            {
                wgM = 32u;
            }

            uint32_t wgN = 16u;
            for (uint32_t candidate : {128u, 64u, 32u})
            {
                if ((n % candidate) == 0u)
                {
                    wgN = candidate;
                    break;
                }
            }
            const uint32_t subgroupN = std::max(1u, wgN / 32u);
            const uint32_t sgMIters = wgM / (subgroupM * 16u);
            const uint32_t sgNIters = wgN / (subgroupN * 16u);
            const uint32_t accFrags = sgMIters * sgNIters;
            cmPlan = finalizePlan(K1S1Plan{
                .sgSize = 32u,
                .wgM = wgM,
                .wgN = wgN,
                .sgM = wgM / subgroupM,
                .sgN = wgN / subgroupN,
                .accStoreBatchSize = accFrags,
                .loadMode = workload.iDtype == Dtype::Fp32 ? 1u : 0u,
                .storeMode = storeAlignedN,
                .accType = 1u,
                .reorder = true,
                .splitK = 1u,
                .swapXY = 1u,
            });
        }
        else
        {
            const auto makePlan = [&](uint32_t wgM, uint32_t wgN, uint32_t subgroupCount)
            {
                const uint32_t sgMIters = wgM / 16u;
                const uint32_t sgNIters = wgN / (subgroupCount * 16u);
                const uint32_t accFrags = sgMIters * sgNIters;
                return finalizePlan(K1S1Plan{
                    .sgSize = 32u,
                    .wgM = wgM,
                    .wgN = wgN,
                    .sgM = wgM,
                    .sgN = wgN / subgroupCount,
                    .accStoreBatchSize = sgMIters == 8u && sgNIters == 1u ? 2u : accFrags,
                    .loadMode = !alignedN || workload.iDtype == Dtype::Fp32 ? 1u : 0u,
                    .storeMode = storeAlignedN,
                    .accType = 1u,
                    .reorder = false,
                    .splitK = 1u,
                    .swapXY = 0u,
                });
            };

            if (!alignedN)
            {
                cmPlan = makePlan(16u, 16u, 1u);
            }
            else
            {
                struct CandidateGeometry
                {
                    uint32_t wgM;
                    uint32_t wgN;
                    uint32_t subgroupCount;
                };
                static constexpr std::array<CandidateGeometry, 7> candidateGeometries{{
                    {128u, 128u, 8u},
                    {128u, 64u, 4u},
                    {64u, 64u, 4u},
                    {64u, 32u, 2u},
                    {32u, 32u, 2u},
                    {16u, 32u, 2u},
                    {16u, 16u, 1u},
                }};
                const uint64_t targetWorkgroups = scaleWorkSupplyThreshold(64u, computeBlockCount);
                for (const CandidateGeometry& candidate : candidateGeometries)
                {
                    if ((co % candidate.wgM) != 0u || (n % candidate.wgN) != 0u) continue;
                    const ConvDispatchFeatures features =
                        makeConvDispatchFeatures(ci, 16u, co, candidate.wgM, n, candidate.wgN);
                    if (features.G >= targetWorkgroups || (candidate.wgM == 16u && candidate.wgN == 16u))
                    {
                        cmPlan = makePlan(candidate.wgM, candidate.wgN, candidate.subgroupCount);
                        break;
                    }
                }
            }
        }

        // The Halo K1 CM shader benefits consistently from folding the
        // reduction and spatial extents at pipeline compile time. Keep the
        // geometry selector unchanged; this only removes bounds/index work
        // from the fp16/IO-fp16 specialization.
        if (device.deviceID() == kAMDStrixHaloDeviceId && computeBlockCount == 40u && workload.iDtype == Dtype::Fp16 &&
            workload.oDtype == Dtype::Fp16 && workload.wDtype == Dtype::Fp16)
        {
            cmPlan.ciSpec = ci;
            cmPlan.nSpec = n;
        }
    }
    else
    {
        // -----------------------------------------------------
        // Generic
        // -----------------------------------------------------
        cmPlan = {
            .sgSize = device.subgroupSize(),
            .wgM = 16u,
            .wgN = 16u,
            .sgM = 16u,
            .sgN = 16u,
            .accStoreBatchSize = 1u,
            .loadMode = !alignedN || workload.iDtype == Dtype::Fp32 ? 1u : 0u,
            .alignM = (co % 16u) == 0u,
            .alignN = alignedN,
            .alignK = (ci % 16u) == 0u,
            .storeMode = storeAlignedN,
            .accType = acc == Dtype::Fp16 ? 0u : 1u,
            .reorder = false,
            .splitK = 1u,
            .swapXY = 0u,
        };
    }
    return cmPlan;
}

inline std::optional<ConvPlan> selectK1S1Candidate(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload,
    const ConvPlan& basePlan)
{
    const uint32_t vendorId = device.vendorID();
    ConvPlan candidate = basePlan;
    candidate.path = ConvPlan::Path::K1S1;

    K1S1Plan selectedPlan;
    bool selectedCM = false;
    if (workload.wDtype == Dtype::Fp16)
    {
        if (enableCM(device, caps) &&
            (workload.co % 16u) == 0u &&
            (workload.ci % 16u) == 0u)
        {
            selectedPlan = selectK1S1CMPlan(device, caps, workload);
            selectedCM = true;
        }

        if (!selectedCM)
            selectedPlan = selectK1S1TiledPlan(device, workload);
    }
    else if (workload.wDtype == Dtype::Fp32)
    {
        selectedPlan = selectK1S1TiledPlan(device, workload);
    }
    else
    {
        return std::nullopt;
    }

    candidate.k1s1 = selectedPlan;
    candidate.useCM = selectedCM;
    if (!selectedCM && subgroupSizeInRange(device, 32u))
    {
        if (workload.wDtype == Dtype::Fp32 && vendorId == eva::VENDOR_ID::AMD &&
            device.deviceID() == kAMDStrixPointDeviceId && device.computeBlockCount() == 16u)
            candidate.requiredSubgroupSize = 32u;

        if (workload.wDtype == Dtype::Fp16 &&
            vendorId == eva::VENDOR_ID::NVIDIA)
            candidate.requiredSubgroupSize = 32u;

        if (selectedPlan.splitK > 1u)
        {
            if (vendorId == eva::VENDOR_ID::NVIDIA)
            {
                candidate.splitRequiredSubgroupSize = 32u;
            }
            else if (vendorId == eva::VENDOR_ID::AMD &&
                     selectedPlan.alignM && selectedPlan.alignN &&
                     selectedPlan.alignK)
            {
                candidate.splitRequiredSubgroupSize = 32u;
            }
        }
    }

    if (!validK1S1Plan(candidate, workload)) return std::nullopt;
    return candidate;
}

inline std::optional<K3S1Plan> selectK3S1ImplicitCMPlan(
    const eva::Device& device,
    const Workload& workload)
{
    const K3S1CMImplicitGeometry* rule = findK3S1CMExactImplicitRule(device, workload);
    if (rule == nullptr) rule = findK3S1CMImplicitRule(workload);
    if (rule == nullptr) return std::nullopt;

    K3S1Plan candidate{
        .algorithm = ConvAlgorithm::Implicit,
        .wgSize = rule->wgSize,
        .wgM = rule->wgM,
        .wgN = rule->wgN,
        .sgSize = 32u,
        .sgM = rule->sgM,
        .sgN = rule->sgN,
        .panelK = rule->panelK,
        .accType = 1u,
        .shape = ShapeSpec::fromWorkload(workload),
    };
    candidate.flatSpatial = device.vendorID() == eva::VENDOR_ID::AMD && device.deviceID() == kAMDStrixHaloDeviceId &&
                            device.computeBlockCount() == 40u && (workload.wo % 8u) == 0u &&
                            (workload.spatialSize() % candidate.wgN) == 0u;
    return candidate;
}

inline std::optional<K3S1Plan> selectK3S1WinogradPlan(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const AMDArchitecture architecture =
        classifyAMDArchitecture(vendorId, device.deviceID());
    const uint32_t ci = workload.ci;
    const uint32_t co = workload.co;
    const uint32_t ho = workload.ho;
    const uint32_t wo = workload.wo;
    const uint32_t numTiles = workload.winogradTileCount();
    K3S1Plan selectedPlan;

    if (workload.wDtype == Dtype::Fp32)
    {
        const K3S1ExactAlgorithmRule* exactAlgorithmRule =
            findK3S1ExactAlgorithmRule(device, workload);
        selectedPlan.algorithm = exactAlgorithmRule != nullptr
            ? exactAlgorithmRule->algorithm
            : ConvAlgorithm::Winograd;
        selectedPlan.wgK = 8u;
        selectedPlan.tM = 4u;
        selectedPlan.tN = 4u;
        selectedPlan.storeMode = (numTiles % 4u) == 0u;

        const auto setAlignment = [&]()
        {
            selectedPlan.alignM = (co % selectedPlan.wgM) == 0u;
            selectedPlan.alignN = (numTiles % selectedPlan.wgN) == 0u;
            selectedPlan.alignK = (ci % selectedPlan.wgK) == 0u;
        };

        if (vendorId == eva::VENDOR_ID::AMD)
        {
            if (device.deviceID() == kAMDStrixHaloDeviceId && device.computeBlockCount() == 40u && co > 64u &&
                numTiles >= 25u)
            {
                selectedPlan.wgSize = 128u;
                selectedPlan.wgM = 64u;
                selectedPlan.wgN = 32u;
                selectedPlan.wgK = 16u;
                selectedPlan.tM = 4u;
                setAlignment();
                selectedPlan.loadMode = true;
                selectedPlan.outputStoreMode =
                    (ho % 4u) == 0u && (wo % 4u) == 0u;
                return selectedPlan;
            }

            // ---------------------------------------------------------
            // AMD
            // ---------------------------------------------------------
            if (co <= 64u)
            {
                if (device.deviceID() == kAMDStrixHaloDeviceId && device.computeBlockCount() == 40u)
                {
                    selectedPlan.wgM = co <= 32u ? co : 64u;
                    selectedPlan.wgN = 32u;
                    selectedPlan.wgSize =
                        (selectedPlan.wgM / selectedPlan.tM) *
                        (selectedPlan.wgN / selectedPlan.tN);
                }
                else
                {
                    selectedPlan.wgSize = 256u;
                    selectedPlan.wgM = 64u;
                    selectedPlan.wgN = 64u;
                }
                setAlignment();
                if (selectedPlan.alignM && selectedPlan.alignK)
                    selectedPlan.loadMode = true;
                else
                    selectedPlan.pad = 1u;
            }
            else
            {
                selectedPlan.wgSize = 256u;
                selectedPlan.wgM = 128u;
                selectedPlan.wgN = 64u;
                selectedPlan.tM = 8u;
                setAlignment();
                if (selectedPlan.alignM && selectedPlan.alignK && numTiles > 25u)
                {
                    selectedPlan.loadMode = true;
                }
                else
                {
                    selectedPlan.wgSize = 128u;
                    selectedPlan.wgM = 64u;
                    selectedPlan.wgN = 32u;
                    selectedPlan.tM = 4u;
                    setAlignment();
                    selectedPlan.loadMode = selectedPlan.alignM && selectedPlan.alignK;
                    selectedPlan.pad = selectedPlan.loadMode ? 0u : 1u;
                }
            }
        }
        else
        {
            // ---------------------------------------------------------
            // NVIDIA and generic vendors
            // ---------------------------------------------------------
            if (co <= 64u)
            {
                selectedPlan.wgSize = 256u;
                selectedPlan.wgM = 64u;
                selectedPlan.wgN = 64u;
                setAlignment();
                selectedPlan.pad = 1u;
            }
            else
            {
                selectedPlan.wgSize = 128u;
                selectedPlan.wgM = 64u;
                selectedPlan.wgN = 32u;
                setAlignment();
                selectedPlan.pad = selectedPlan.alignM && selectedPlan.alignK ? 0u : 1u;
            }
        }
        selectedPlan.outputStoreMode = device.vendorID() == eva::VENDOR_ID::AMD &&
                                       device.deviceID() == kAMDStrixHaloDeviceId &&
                                       device.computeBlockCount() == 40u && (ho % 4u) == 0u && (wo % 4u) == 0u;
        return selectedPlan;
    }

    if (workload.wDtype != Dtype::Fp16) return std::nullopt;
    const Dtype acc = selectAccType(device, caps);

    if (vendorId == eva::VENDOR_ID::NVIDIA)
    {
        // ---------------------------------------------------------
        // NVIDIA cooperative-matrix Winograd
        // ---------------------------------------------------------
        selectedPlan.algorithm = ConvAlgorithm::Winograd;
        selectedPlan.sgSize = 32u;
        selectedPlan.sgN = 32u;
        selectedPlan.wgN = 32u;
        selectedPlan.panelK = 16u;
        selectedPlan.accType = acc == Dtype::Fp16 ? 0u : 1u;
        selectedPlan.storeMode = true;
        selectedPlan.reorder = true;

        uint32_t subgroupCount = 1u;
        const bool alignM = (co % 32u) == 0u;
        const bool alignK = (ci % 32u) == 0u;
        if (alignM && alignK)
        {
            for (const uint32_t candidate : {8u, 4u, 2u, 1u})
            {
                if ((co % (candidate * 32u)) == 0u)
                {
                    subgroupCount = candidate;
                    break;
                }
            }
            selectedPlan.sgM = 32u;
        }
        else
        {
            selectedPlan.sgM = alignM ? 32u : 16u;
        }

        selectedPlan.wgSize = 32u * subgroupCount;
        selectedPlan.wgM = selectedPlan.sgM * subgroupCount;
        if (selectedPlan.accType == 0u &&
            ci >= 512u && subgroupCount <= 2u && alignM && alignK)
        {
            while (selectedPlan.splitK < 8u && ci / selectedPlan.splitK > 128u &&
                   (ci % (selectedPlan.splitK * 2u)) == 0u)
                selectedPlan.splitK *= 2u;
        }
        const uint32_t spatialGroups = CEIL_DIV(numTiles, selectedPlan.wgN);
        selectedPlan.swapXY = alignM && alignK &&
            (spatialGroups >= 16u || subgroupCount <= 2u);

        const uint32_t tailN = numTiles % selectedPlan.wgN;
        if (selectedPlan.splitK == 1u && numTiles >= selectedPlan.wgN && tailN != 0u && tailN <= 16u)
        {
            selectedPlan.tailNOffset = (numTiles / selectedPlan.wgN) * selectedPlan.wgN;
        }
        selectedPlan.outputStoreMode = (ho % 4u) == 0u && (wo % 4u) == 0u;
    }
    else if (vendorId == eva::VENDOR_ID::AMD)
    {
        // ---------------------------------------------------------
        // AMD cooperative-matrix Winograd
        // ---------------------------------------------------------
        selectedPlan.algorithm = ConvAlgorithm::Winograd;
        selectedPlan.panelK = 16u;
        selectedPlan.accType = 1u;
        selectedPlan.loadMode = true;

        if ((co % 128u) == 0u && (ci % 16u) == 0u)
        {
            if (architecture == AMDArchitecture::RDNA3 &&
                numTiles >= 64u && numTiles < 128u)
            {
                selectedPlan.wgSize = 256u;
                selectedPlan.wgM = 128u;
                selectedPlan.wgN = 32u;
                selectedPlan.sgSize = 64u;
                selectedPlan.storeMode = true;
                selectedPlan.panelK = 64u;
            }
            else if (architecture == AMDArchitecture::RDNA4 && numTiles >= 64u)
            {
                selectedPlan.wgSize = 256u;
                selectedPlan.wgM = 128u;
                selectedPlan.wgN = 32u;
                selectedPlan.sgSize = 32u;
                const uint32_t remainder = numTiles % selectedPlan.wgN;
                if (remainder != 0u && remainder != 16u)
                    selectedPlan.tailNOffset = numTiles - remainder;
            }
            else
            {
                selectedPlan.wgSize = 512u;
                selectedPlan.wgM = 128u;
                selectedPlan.wgN = 16u;
                selectedPlan.sgSize = 64u;
            }
        }
        else if ((co % 64u) == 0u && (ci % 16u) == 0u)
        {
            selectedPlan.wgSize = 256u;
            selectedPlan.wgM = 64u;
            selectedPlan.wgN = 16u;
            selectedPlan.sgSize = 64u;
        }
        else
        {
            selectedPlan.wgSize = 32u;
            selectedPlan.wgM = 16u;
            selectedPlan.wgN = 16u;
            selectedPlan.sgSize = 32u;
        }

        if (device.deviceID() == kAMDStrixHaloDeviceId && device.computeBlockCount() == 40u && ci == 256u &&
            co == 256u && ho == 80u && wo == 80u)
        {
            selectedPlan.wgSize = 256u;
            selectedPlan.wgM = 128u;
            selectedPlan.wgN = 16u;
            selectedPlan.sgSize = 32u;
        }
        selectedPlan.sgM = selectedPlan.wgM / (selectedPlan.wgSize / selectedPlan.sgSize);
        selectedPlan.sgN = selectedPlan.wgN;
        selectedPlan.reorder = architecture == AMDArchitecture::RDNA3 ||
                               architecture == AMDArchitecture::RDNA4;
        selectedPlan.inputReorder = architecture == AMDArchitecture::RDNA4;
        if (selectedPlan.inputReorder) selectedPlan.loadMode = false;
        if (architecture == AMDArchitecture::RDNA3 &&
            selectedPlan.loadMode && ci >= 256u)
            selectedPlan.panelK = 64u;
    }
    else
    {
        return std::nullopt;
    }

    return selectedPlan;
}

inline std::optional<ConvPlan> selectK3S1Candidate(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload,
    const ConvPlan& basePlan)
{
    const K3S1ExactAlgorithmRule* exactAlgorithmRule =
        findK3S1ExactAlgorithmRule(device, workload);
    if (findK3S1ImplicitRule(device, workload.wDtype) != nullptr &&
        exactAlgorithmRule == nullptr)
        return std::nullopt;
    if (workload.residual && workload.ci != workload.co) return std::nullopt;
    if (workload.winogradTileCount() < 16u) return std::nullopt;

    K3S1Plan selectedPlan;
    bool selectedCM = false;
    if (workload.wDtype == Dtype::Fp32)
    {
        if (const auto winogradPlan = selectK3S1WinogradPlan(device, caps, workload))
        {
            if (validK3S1Plan(*winogradPlan, device, false, workload))
                selectedPlan = *winogradPlan;
        }
    }
    else if (workload.wDtype == Dtype::Fp16)
    {
        if (!enableCM(device, caps)) return std::nullopt;
        if ((workload.co % 16u) != 0u) return std::nullopt;
        const uint64_t reductionSize = uint64_t(workload.ci) * 9u;
        if ((reductionSize % 16u) != 0u) return std::nullopt;
        const K3S1AlgorithmRule* algorithmRule = findK3S1AlgorithmRule(device);
        const ConvAlgorithm preferredAlgorithm = exactAlgorithmRule != nullptr
            ? exactAlgorithmRule->algorithm
            : algorithmRule != nullptr
                ? algorithmRule->algorithm
                : ConvAlgorithm::None;
        if (preferredAlgorithm != ConvAlgorithm::Winograd &&
            subgroupSizeInRange(device, 32u))
        {
            if (const auto implicitPlan =
                    selectK3S1ImplicitCMPlan(device, workload))
            {
                if (k3s1SharedMemoryBytes(
                        *implicitPlan, implicitPlan->wgN,
                        workload.wDtype) <=
                        device.maxComputeSharedMemorySize() &&
                    validK3S1Plan(*implicitPlan, device, true, workload))
                {
                    selectedPlan = *implicitPlan;
                    selectedCM = true;
                }
            }
        }

        if (!selectedPlan.valid())
        {
            if (const auto winogradPlan =
                    selectK3S1WinogradPlan(device, caps, workload))
            {
                if (validK3S1Plan(*winogradPlan, device, true, workload))
                {
                    selectedPlan = *winogradPlan;
                    selectedCM = true;
                }
            }
        }
    }
    else
    {
        return std::nullopt;
    }

    if (!selectedPlan.valid()) return std::nullopt;

    ConvPlan candidate = basePlan;
    candidate.path = ConvPlan::Path::K3S1;
    candidate.k3s1 = selectedPlan;
    candidate.useCM = selectedCM;
    if (!selectedCM &&
        device.vendorID() == eva::VENDOR_ID::AMD &&
        subgroupSizeInRange(
            device,
            device.deviceID() == kAMDStrixHaloDeviceId ? 64u : 32u))
    {
        candidate.requiredSubgroupSize =
            device.deviceID() == kAMDStrixHaloDeviceId ? 64u : 32u;
    }
    return candidate;
}

inline void setK3S2Alignment(K3S2Plan& plan, const Workload& workload)
{
    plan.alignM = plan.wgM != 0u && (workload.co % plan.wgM) == 0u;
    plan.alignN = plan.wgN != 0u && (workload.spatialSize() % plan.wgN) == 0u;
    if (plan.cooperativeMatrix())
    {
        const uint32_t kk = workload.ci * 9u;
        plan.alignK = plan.packedWeights ||
            ((kk % plan.splitK) == 0u && ((kk / plan.splitK) % 16u) == 0u);
    }
    else if (plan.wgK != 0u)
    {
        const uint32_t kk = workload.ci * 9u;
        const uint32_t kPerSplit = CEIL_DIV(kk, plan.splitK);
        plan.alignK = (kk % plan.splitK) == 0u &&
            (kPerSplit % plan.wgK) == 0u;
    }
}

inline std::optional<K3S2Plan> selectK3S2DirectPlan(
    const eva::Device& device,
    const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const uint32_t Ci = workload.ci;
    const uint32_t Co = workload.co;
    const uint32_t N = workload.spatialSize();

    K3S2Plan selectedPlan;
    if (workload.wDtype == Dtype::Fp32)
    {
        if (vendorId == eva::VENDOR_ID::NVIDIA)
        {
            if ((N % 64u) == 0u && (Co % 64u) == 0u)
            {
                selectedPlan = {
                    .wgSize = 256u,
                    .wgM = 64u,
                    .wgN = 64u,
                    .wgK = 8u,
                    .tM = 4u,
                    .shape = ShapeSpec::fromWorkload(workload),
                };
            }
            else if (Ci >= 128u &&
                     CEIL_DIV(N, 64u) * CEIL_DIV(Co, 64u) < scaleWorkSupplyThreshold(60u, device.computeBlockCount()))
            {
                selectedPlan.splitK = Ci >= 512u ? 8u : 4u;
            }
        }
        else
        {
            const AMDArchitecture architecture =
                classifyAMDArchitecture(vendorId, device.deviceID());
            const uint32_t splitWorkgroupThreshold =
                architecture == AMDArchitecture::RDNA4 ? 60u : 512u;
            const uint32_t unsplitWorkgroups =
                CEIL_DIV(N, 64u) * CEIL_DIV(Co, 128u);
            if (Ci >= 128u &&
                unsplitWorkgroups < scaleWorkSupplyThreshold(splitWorkgroupThreshold, device.computeBlockCount()))
                selectedPlan.splitK = Ci >= 512u ? 8u : 4u;

            selectedPlan.wgSize = 256u;
            selectedPlan.wgN = 64u;
            if (selectedPlan.split())
            {
                selectedPlan.wgM = 128u;
                selectedPlan.wgK = 8u;
                selectedPlan.tM = 8u;
            }
            else
            {
                selectedPlan.wgM = 128u;
                selectedPlan.wgK = 16u;
                selectedPlan.tM = 8u;
                setK3S2Alignment(selectedPlan, workload);
                if (!selectedPlan.alignM || !selectedPlan.alignN || !selectedPlan.alignK)
                {
                    selectedPlan.wgM = 64u;
                    selectedPlan.wgK = 8u;
                    selectedPlan.tM = 4u;
                }
            }
        }

        if (!selectedPlan.valid())
        {
            selectedPlan.wgSize = 256u;
            selectedPlan.wgM = selectedPlan.split() ||
                vendorId != eva::VENDOR_ID::NVIDIA ? 128u : 64u;
            selectedPlan.wgN = 64u;
            selectedPlan.wgK = 8u;
            selectedPlan.tM = selectedPlan.wgM == 128u ? 8u : 4u;
        }
        setK3S2Alignment(selectedPlan, workload);
        return selectedPlan;
    }

    if (workload.wDtype != Dtype::Fp16) return std::nullopt;

    if (vendorId != eva::VENDOR_ID::NVIDIA &&
        Ci == 3u && Co <= 64u && N >= 64u)
    {
        selectedPlan = {
            .wgSize = 256u,
            .wgM = Co,
            .wgN = 64u,
            .wgK = 27u,
            .tM = Co / 16u,
            .shape = ShapeSpec::fromWorkload(workload),
        };
    }
    else
    {
        selectedPlan = {
            .wgSize = 256u,
            .wgM = vendorId == eva::VENDOR_ID::NVIDIA ? 64u : 128u,
            .wgN = 64u,
            .wgK = 8u,
            .tM = vendorId == eva::VENDOR_ID::NVIDIA ? 4u : 8u,
        };
    }
    setK3S2Alignment(selectedPlan, workload);
    return selectedPlan;
}

inline std::optional<K3S2Plan> selectK3S2CMPlan(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload)
{
    const uint32_t vendorId = device.vendorID();
    const AMDArchitecture architecture =
        classifyAMDArchitecture(vendorId, device.deviceID());
    const uint32_t Co = workload.co;
    const uint32_t N = workload.spatialSize();
    const uint32_t reductionTiles = 9u * (workload.ci / 16u);

    if (vendorId == eva::VENDOR_ID::AMD &&
        architecture == AMDArchitecture::RDNA3)
    {
        K3S2Plan candidate;
        candidate.shape = ShapeSpec::fromWorkload(workload);
        candidate.wgN = 32u;
        candidate.sgM = 16u;
        candidate.sgN = 16u;
        if (reductionTiles >= 72u && (Co % 32u) == 0u)
        {
            candidate.sgM = 32u;
            candidate.sgN = 32u;
        }
        for (uint32_t wgM : {512u, 256u, 128u, 64u, 32u, 16u})
        {
            if (wgM > Co) continue;
            if ((Co % wgM) != 0u) continue;
            if ((wgM % candidate.sgM) != 0u) continue;

            const uint32_t subgroupCount =
                (wgM / candidate.sgM) * (candidate.wgN / candidate.sgN);
            if (subgroupCount <= 16u)
            {
                candidate.wgM = wgM;
                candidate.wgSize = subgroupCount * 32u;
                break;
            }
        }

        const uint32_t unsplitWorkgroups =
            CEIL_DIV(N, candidate.wgN) * (Co / candidate.wgM);
        const uint32_t unsplitWaves =
            unsplitWorkgroups * (candidate.wgSize / 32u);
        if (unsplitWaves < scaleWorkSupplyThreshold(512u, device.computeBlockCount()) && reductionTiles >= 144u &&
            (Co % 128u) == 0u)
        {
            const uint32_t targetWaves = scaleWorkSupplyThreshold(1024u, device.computeBlockCount());
            const uint32_t splitBaseWorkgroups =
                CEIL_DIV(N, 64u) * (Co / 128u);
            uint32_t selectedSplitK = 1u;
            for (uint32_t splitK : {2u, 4u, 8u})
            {
                if ((reductionTiles % splitK) != 0u) continue;
                if (reductionTiles / splitK < 36u) continue;
                selectedSplitK = splitK;
                if (splitBaseWorkgroups * 8u * splitK >= targetWaves) break;
            }
            if (selectedSplitK > 1u)
            {
                candidate.splitK = selectedSplitK;
                candidate.wgSize = 256u;
                candidate.wgM = 128u;
                candidate.wgN = 64u;
                candidate.sgM = 16u;
                candidate.sgN = 64u;
            }
        }
        setK3S2Alignment(candidate, workload);
        if (candidate.split() && workload.oDtype == Dtype::Fp16 &&
            (N % 4u) != 0u)
            return std::nullopt;
        return candidate;
    }

    if (vendorId != eva::VENDOR_ID::NVIDIA &&
        vendorId != eva::VENDOR_ID::AMD)
        return std::nullopt;

    K3S2Plan candidate;
    candidate.packedWeights = true;
    candidate.accType = selectAccType(device, caps) == Dtype::Fp16 ? 0u : 1u;
    candidate.directOutput = architecture == AMDArchitecture::RDNA4;
    candidate.sgM = reductionTiles >= 144u && (Co % 64u) == 0u ? 64u
                    : (Co % 32u) == 0u                         ? 32u
                                                               : 16u;
    candidate.sgN = 32u;
    candidate.wgM = candidate.sgM;
    for (uint32_t wgM : {256u, 128u, 64u, 32u, 16u})
    {
        if (wgM < candidate.sgM) continue;
        if ((wgM % candidate.sgM) != 0u) continue;
        if ((Co % wgM) != 0u) continue;
        if (wgM / candidate.sgM > 8u) continue;
        candidate.wgM = wgM;
        break;
    }
    candidate.wgSize = 32u * (candidate.wgM / candidate.sgM);
    candidate.wgN = candidate.sgN;
    const uint32_t unsplitWorkgroups =
        CEIL_DIV(N, candidate.wgN) * (Co / candidate.wgM);
    const uint32_t targetWorkgroups = scaleWorkSupplyThreshold(256u, device.computeBlockCount());
    if (architecture == AMDArchitecture::RDNA4)
    {
        for (uint32_t splitK : {1u, 3u, 9u})
        {
            const uint32_t kTilesPerSplit = reductionTiles / splitK;
            if (kTilesPerSplit < 8u) continue;
            if (unsplitWorkgroups * splitK < targetWorkgroups && splitK != 9u)
                continue;
            candidate.splitK = splitK;
            break;
        }
    }
    else if (unsplitWorkgroups < targetWorkgroups)
    {
        if (reductionTiles >= 288u)
            candidate.splitK = 9u;
        else if (reductionTiles >= 144u)
            candidate.splitK = 5u;
        else if (reductionTiles >= 72u)
            candidate.splitK = 3u;
    }
    candidate.shape.wo = workload.wo;
    setK3S2Alignment(candidate, workload);
    return candidate;
}

inline std::optional<ConvPlan> selectK3S2Candidate(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload,
    const ConvPlan& basePlan)
{
    K3S2Plan selectedPlan;
    bool selectedCM = false;

    const K3S2ExactRule* rule = nullptr;
    if (workload.wDtype == Dtype::Fp32)
    {
        if (workload.iDtype != Dtype::Fp32) return std::nullopt;
        if (workload.oDtype != Dtype::Fp32) return std::nullopt;
        rule = findK3S2ExactRule(device, workload);
    }
    else if (workload.wDtype == Dtype::Fp16)
    {
        rule = findK3S2ExactRule(device, workload);
    }
    else
    {
        return std::nullopt;
    }

    if (rule != nullptr)
    {
        selectedPlan = rule->plan;
        setK3S2Alignment(selectedPlan, workload);
        if (!validK3S2Plan(selectedPlan, workload)) return std::nullopt;
        selectedCM = rule->useCM;
    }
    else if (workload.wDtype == Dtype::Fp32)
    {
        const auto directPlan = selectK3S2DirectPlan(device, workload);
        if (directPlan && validK3S2Plan(*directPlan, workload))
            selectedPlan = *directPlan;
    }
    else
    {
        std::optional<K3S2Plan> cmPlan;
        const uint64_t reductionSize = uint64_t(workload.ci) * 9u;
        if (enableCM(device, caps) &&
            (workload.co % 16u) == 0u &&
            (reductionSize % 16u) == 0u)
            cmPlan = selectK3S2CMPlan(device, caps, workload);
        if (cmPlan && validK3S2Plan(*cmPlan, workload))
        {
            selectedPlan = *cmPlan;
            selectedCM = true;
        }
        else
        {
            const auto directPlan = selectK3S2DirectPlan(device, workload);
            if (directPlan && validK3S2Plan(*directPlan, workload))
                selectedPlan = *directPlan;
        }
    }

    if (!selectedPlan.valid()) return std::nullopt;

    ConvPlan candidate = basePlan;
    candidate.path = ConvPlan::Path::K3S2;
    candidate.k3s2 = selectedPlan;
    candidate.useCM = selectedCM;
    if (!selectedCM && subgroupSizeInRange(device, 32u))
    {
        if (workload.wDtype == Dtype::Fp32)
        {
            if (device.vendorID() == eva::VENDOR_ID::AMD &&
                selectedPlan.split() &&
                classifyAMDArchitecture(device.vendorID(), device.deviceID()) ==
                    AMDArchitecture::RDNA3)
                candidate.requiredSubgroupSize = 32u;
        }
        else if (workload.wDtype == Dtype::Fp16)
        {
            if ((device.vendorID() == eva::VENDOR_ID::AMD &&
                 selectedPlan.firstLayer()) ||
                device.vendorID() == eva::VENDOR_ID::NVIDIA)
                candidate.requiredSubgroupSize = 32u;
        }
    }
    if (!selectedCM && workload.wDtype == Dtype::Fp32 && device.vendorID() == eva::VENDOR_ID::AMD &&
        device.deviceID() == kAMDStrixPointDeviceId && device.computeBlockCount() == 16u &&
        subgroupSizeInRange(device, 32u))
        candidate.requiredSubgroupSize = 32u;
    return candidate;
}

inline ConvPlan selectGeneralFallbackPlan(
    const eva::Device& device,
    const ConvolutionCaps& caps,
    const Workload& workload,
    const ConvPlan& basePlan)
{
    ConvPlan fallback = basePlan;
    fallback.path = ConvPlan::Path::General;
    const uint32_t implicitK = workload.ci * workload.kernel * workload.kernel;
    const K3S1ImplicitPlanRule* implicitPlan = findK3S1ImplicitPlanRule(device, workload);
    if (implicitPlan != nullptr)
    {
        fallback.implicit = implicitPlan->plan;
        fallback.requiredSubgroupSize = workload.residual
            ? implicitPlan->residualRequiredSubgroupSize
            : implicitPlan->requiredSubgroupSize;
    }
    else
    {
        bool implicitSelected = false;
        if (workload.wDtype == Dtype::Fp16)
        {
            const Dtype acc = selectAccType(device, caps);
            if (enableCM(device, caps))
            {
                ImplicitGemmPlan candidate{
                    .wgSize = 128u,
                    .wgM = 32u,
                    .wgN = 32u,
                    .sgM = 16u,
                    .sgN = 16u,
                    .wgK = 0u,
                    .tM = 0u,
                    .panelK = 32u,
                    .accType = acc == Dtype::Fp16 ? 0u : 1u,
                    .cooperativeMatrix = true,
                };
                if (validImplicitGemmPlan(candidate, device, caps, workload.wDtype))
                {
                    fallback.implicit = candidate;
                    implicitSelected = true;
                }
            }
        }

        if (!implicitSelected)
        {
            for (const ImplicitGemmPlan& candidate : gConvFallbacks)
            {
                if (!validImplicitGemmPlan(candidate, device, caps, workload.wDtype)) continue;
                fallback.implicit = candidate;
                implicitSelected = true;
                break;
            }
        }

        if (!implicitSelected)
        {
            throw std::runtime_error("Convolution has no implicit GEMM plan compatible "
                                     "with this device");
        }
    }
    fallback.useCM = fallback.implicit.cooperativeMatrix;
    if (!fallback.useCM)
    {
        VAI_ASSERT(fallback.implicit.wgK != 0u);
        fallback.implicit.alignK = (implicitK % fallback.implicit.wgK) == 0u;
        if (workload.kernel == 3u && workload.stride == 1u && workload.wDtype == Dtype::Fp32 &&
            device.vendorID() == eva::VENDOR_ID::AMD && device.deviceID() == kAMDStrixPointDeviceId &&
            device.computeBlockCount() == 16u)
        {
            fallback.implicit.structuredInputLoad = true;
            fallback.implicit.alignM =
                (workload.co % fallback.implicit.wgM) == 0u;
            fallback.implicit.alignN =
                (workload.spatialSize() % fallback.implicit.wgN) == 0u;
            if (subgroupSizeInRange(device, 32u))
                fallback.requiredSubgroupSize = 32u;
        }
    }
    if (!fallback.useCM && workload.kernel == 3u && workload.stride == 1u)
    {
        fallback.path = ConvPlan::Path::K3S1;
        fallback.k3s1.algorithm = ConvAlgorithm::Implicit;
    }
    if (validConvLaunches(fallback, workload, device, caps)) return fallback;

    // Retry every fallback tile in descending order. A larger tile can satisfy
    // its static topology yet exceed a dispatch or subgroup limit that a
    // smaller fallback tile satisfies.
    if (workload.kernel == 3u && workload.stride == 1u)
    {
        fallback.path = ConvPlan::Path::K3S1;
        fallback.k3s1.algorithm = ConvAlgorithm::Implicit;
    }
    for (const ImplicitGemmPlan& plan : gConvFallbacks)
    {
        if (!validImplicitGemmPlan(plan, device, caps, workload.wDtype)) continue;
        fallback.implicit = plan;
        fallback.useCM = false;
        fallback.implicit.alignK = (implicitK % fallback.implicit.wgK) == 0u;
        fallback.requiredSubgroupSize = 0u;
        if (validConvLaunches(fallback, workload, device, caps)) return fallback;
    }

    throw std::runtime_error("No convolution path satisfies the device "
                             "capabilities and launch limits");
}

inline ConvPlan selectConvPlan(const eva::Device& device, const ConvolutionCaps& caps, const Workload& workload)
{
    VAI_ASSERT(workload.ci != 0u && workload.co != 0u);
    VAI_ASSERT(workload.kernel != 0u && workload.stride != 0u);
    VAI_ASSERT(workload.hi != 0u && workload.wi != 0u);
    VAI_ASSERT(workload.ho != 0u && workload.wo != 0u);

    const ConvPlan basePlan{
        .vendorId = device.vendorID(),
        .iDtype = workload.iDtype,
        .oDtype = workload.oDtype,
        .wDtype = workload.wDtype,
        .fusion = workload.fusion,
    };

    ConvPlan::Path preferredPath = ConvPlan::Path::General;
    if (workload.kernel == 1u && workload.stride == 1u)
        preferredPath = ConvPlan::Path::K1S1;
    else if (workload.kernel == 3u && workload.stride == 1u)
        preferredPath = ConvPlan::Path::K3S1;
    else if (workload.kernel == 3u && workload.stride == 2u)
        preferredPath = ConvPlan::Path::K3S2;

    std::optional<ConvPlan> candidate;
    if (preferredPath == ConvPlan::Path::K1S1)
        candidate = selectK1S1Candidate(device, caps, workload, basePlan);
    else if (preferredPath == ConvPlan::Path::K3S1)
        candidate = selectK3S1Candidate(device, caps, workload, basePlan);
    else if (preferredPath == ConvPlan::Path::K3S2)
        candidate = selectK3S2Candidate(device, caps, workload, basePlan);

    if (candidate && validConvLaunches(*candidate, workload, device, caps))
        return *candidate;

    return selectGeneralFallbackPlan(device, caps, workload, basePlan);
}

inline void Convolution::prepareK1S1Split(TiledResources& tiled, const Workload& workload)
{
    const auto [iDtype, oDtype, affineDtype, hasScale, epilogue] = makeFusionSpec(workload);
    const Dtype weightType = workload.wDtype;

    VAI_ASSERT(weightType != Dtype::None && iDtype != Dtype::None && oDtype != Dtype::None);
    const uint32_t kPerSplitSpec = CEIL_DIV(config_.inChannels, plan_.k1s1.splitK);
    tiled.split = makePipelineBinding(k1s1Pipeline(pipelinePool(), plan_.k1s1, weightType, Dtype::Fp32, 0u, 0u,
                                                   hasBias(), ResidualFusion{}, iDtype, Dtype::Fp32, kPerSplitSpec,
                                                   plan_.splitRequiredSubgroupSize));

    VAI_ASSERT(plan_.k1s1.splitK <= 16u);
    tiled.reduce =
        makePipelineBinding(bnReducePipeline(pipelinePool(), Dtype::Fp32, affineDtype, 1u, hasBias(), epilogue,
                                             residualFusion(), iDtype, (workload.spatialSize() % 4u) == 0u, oDtype));
}

inline void Convolution::prepareK1S1(const Workload& workload)
{
    K1S1Resources& resources = std::get<K1S1Resources>(prepared_);
    CooperativeMatrixResources& cm = resources.cm;
    TiledResources& tiled = resources.tiled;
    VAI_ASSERT(plan_.path == ConvPlan::Path::K1S1);
    const uint32_t Ci = workload.ci;
    const Dtype weightType = workload.wDtype;
    const auto [iDtype, oDtype, affineDtype, hasScale, epilogue] = makeFusionSpec(workload);
    const uint32_t requiredSubgroupSize = plan_.requiredSubgroupSize;

    if (plan_.wDtype == Dtype::Fp32)
    {
        if (plan_.k1s1.splitK > 1u)
        {
            prepareK1S1Split(tiled, workload);
            return;
        }

        tiled.main =
            makePipelineBinding(k1s1Pipeline(pipelinePool(), plan_.k1s1, weightType, affineDtype, epilogue, hasScale,
                                             hasBias(), residualFusion(), iDtype, oDtype, 0u, requiredSubgroupSize));
        return;
    }
    else if (plan_.wDtype == Dtype::Fp16)
    {
        if (plan_.useCM)
        {
            const K1S1Plan& plan = plan_.k1s1;
            VAI_ASSERT(plan.splitK == 1u || plan.accType == 0u);
            const uint32_t cmRequiredSubgroupSize = plan.sgSize;
            const uint32_t kPerSplit = plan.splitK > 1u ? CEIL_DIV(Ci / plan.splitK, 16u) * 16u : 0u;
            tiled.main = makePipelineBinding(k1s1CMPipeline(
                pipelinePool(), plan, kPerSplit, epilogue, affineDtype, hasScale, hasBias(),
                plan.splitK > 1u ? ResidualFusion{} : residualFusion(), iDtype, oDtype, cmRequiredSubgroupSize));

            if (plan.reorder)
            {
                cm.transform = makePipelineBinding(
                    cmTransformPipeline(pipelinePool(), 0u, cmRequiredSubgroupSize));
            }

            if (plan.splitK > 1u)
            {
                cm.reduce =
                    makePipelineBinding(bnReducePipeline(pipelinePool(), Dtype::Fp16, affineDtype, 1u, hasBias(),
                                                         epilogue, residualFusion(), iDtype, true, oDtype));
            }
        }
        else
        {
            if (plan_.k1s1.splitK > 1u)
            {
                prepareK1S1Split(tiled, workload);
            }
            else
            {
                tiled.main = makePipelineBinding(k1s1Pipeline(pipelinePool(), plan_.k1s1, weightType, affineDtype,
                                                              epilogue, hasScale, hasBias(), residualFusion(), iDtype,
                                                              oDtype, 0u, requiredSubgroupSize));
            }
        }
        return;
    }

    VAI_ASSERT(false && "K1S1 requires fp32 or fp16 weights");
}

inline void Convolution::prepareK3S1(const Workload& workload)
{
    K3S1Resources& resources = std::get<K3S1Resources>(prepared_);
    ImplicitResources& implicit = resources.implicit;
    WinogradResources& winograd = resources.winograd;
    CooperativeMatrixResources& cm = winograd.cm;
    VAI_ASSERT(plan_.path == ConvPlan::Path::K3S1);
    const uint32_t Ci = workload.ci;
    VAI_ASSERT(workload.kernel == 3u && workload.stride == 1u);
    const auto [iDtype, oDtype, affineDtype, hasScale, epilogue] = makeFusionSpec(workload);
    const uint32_t requiredSubgroupSize = plan_.requiredSubgroupSize;

    switch (plan_.k3s1.algorithm)
    {
    case ConvAlgorithm::Implicit:
    {
        if (plan_.useCM)
        {
            constexpr uint32_t implicitSubgroupSize = 32u;
            implicit.gemm = makePipelineBinding(
                k3s1ImplicitGemmPipeline(pipelinePool(), plan_.k3s1, epilogue, affineDtype, hasScale, hasBias(), iDtype,
                                         oDtype, residualFusion(),
                                         device().deviceID() == kAMDStrixHaloDeviceId && plan_.k3s1.flatSpatial &&
                                             useHaloK3S1DirectOutput(workload),
                                         implicitSubgroupSize));
            implicit.weightTransform = makePipelineBinding(
                cmTransformPipeline(
                    pipelinePool(), 3u, implicitSubgroupSize));
        }
        else
        {
            implicit.gemm = makePipelineBinding(ConvPipeline(pipelinePool(), plan_.implicit, workload, epilogue,
                                                             affineDtype, hasScale, hasBias(), residualFusion(), true,
                                                             plan_.requiredSubgroupSize));
        }
        break;
    }

    case ConvAlgorithm::Winograd:
    {
        auto prepareWinograd = [&](Dtype wDtype) {
            winograd.filterTransform = makePipelineBinding(winogradFilterTransformPipeline(
                pipelinePool(), wDtype, false, requiredSubgroupSize));

            winograd.inputTransform = makePipelineBinding(winogradInputTransformPipeline(
                pipelinePool(), wDtype, iDtype, requiredSubgroupSize));

            winograd.outputTransform = makePipelineBinding(
                winogradOutputPipeline(pipelinePool(), plan_.k3s1.outputStoreMode, wDtype, affineDtype, epilogue,
                                       residualFusion(), hasScale, hasBias(), iDtype, oDtype, requiredSubgroupSize));

            winograd.gemm = makePipelineBinding(winogradPipeline(
                pipelinePool(), plan_.k3s1, wDtype, requiredSubgroupSize));
        };

        if (workload.wDtype == Dtype::Fp32)
        {
            prepareWinograd(Dtype::Fp32);
        }
        else if (workload.wDtype == Dtype::Fp16)
        {
            if (plan_.useCM)
            {
                if (plan_.vendorId == eva::VENDOR_ID::AMD)
                {
                    const uint32_t cmRequiredSubgroupSize = plan_.k3s1.sgSize;
                    winograd.filterTransform = makePipelineBinding(winogradFilterTransformPipeline(
                        pipelinePool(), Dtype::Fp16, plan_.k3s1.reorder,
                        cmRequiredSubgroupSize));

                    winograd.inputTransform = makePipelineBinding(winogradInputTransformPipeline(
                        pipelinePool(), Dtype::Fp16, iDtype, cmRequiredSubgroupSize,
                        plan_.k3s1.inputReorder));

                    winograd.outputTransform = makePipelineBinding(winogradOutputPipeline(
                        pipelinePool(), false, Dtype::Fp16, affineDtype, epilogue, residualFusion(), hasScale,
                        hasBias(), iDtype, oDtype, cmRequiredSubgroupSize));

                    winograd.gemm = makePipelineBinding(winogradCMPipeline(
                        pipelinePool(), plan_.k3s1, plan_.k3s1.wgN, 0u, 0u, false));
                    if (plan_.k3s1.hasTail())
                    {
                        winograd.cmTailSharedDescriptorPipeline = winogradCMPipeline(
                            pipelinePool(), plan_.k3s1, kK3S1TailWgN, plan_.k3s1.tailNOffset, 0u, false);
                    }
                }
                else if (plan_.vendorId == eva::VENDOR_ID::NVIDIA)
                {
                    constexpr uint32_t cmRequiredSubgroupSize = 32u;
                    winograd.filterTransform = makePipelineBinding(winogradFilterTransformPipeline(
                        pipelinePool(), Dtype::Fp16, false, cmRequiredSubgroupSize));

                    winograd.inputTransform = makePipelineBinding(winogradInputTransformPipeline(
                        pipelinePool(), Dtype::Fp16, iDtype, cmRequiredSubgroupSize));

                    const uint32_t kPerSplit = plan_.k3s1.splitK > 1u ? Ci / plan_.k3s1.splitK : 0u;
                    winograd.gemm = makePipelineBinding(winogradCMPipeline(
                        pipelinePool(), plan_.k3s1, plan_.k3s1.wgN, 0u, kPerSplit, true));

                    if (plan_.k3s1.hasTail())
                    {
                        winograd.cmTailGemm = makePipelineBinding(winogradCMPipeline(
                            pipelinePool(), plan_.k3s1, kK3S1TailWgN, plan_.k3s1.tailNOffset, 0u, true));
                    }

                    cm.transform = makePipelineBinding(cmTransformPipeline(pipelinePool(), 1u, cmRequiredSubgroupSize));

                    if (plan_.k3s1.splitK > 1u)
                    {
                        cm.reduce = makePipelineBinding(bnReducePipeline(pipelinePool(), Dtype::Fp32, Dtype::Fp32, 1u,
                                                                         hasBias(), 0u, ResidualFusion{}, iDtype, true,
                                                                         Dtype::Fp16, true));
                    }

                    winograd.outputTransform = makePipelineBinding(winogradOutputPipeline(
                        pipelinePool(), plan_.k3s1.outputStoreMode, Dtype::Fp16, affineDtype, epilogue,
                        residualFusion(), hasScale, hasBias(), iDtype, oDtype, cmRequiredSubgroupSize));
                }
                else
                {
                    VAI_ASSERT(false && "K3S1 Winograd CM requires an AMD or NVIDIA plan");
                }
            }
            else
            {
                prepareWinograd(Dtype::Fp16);
            }
        }
        else
        {
            VAI_ASSERT(false && "K3S1 Winograd requires fp32 or fp16 weights");
        }
        break;
    }

    case ConvAlgorithm::None:
        VAI_ASSERT(false && "K3S1 algorithm not selected");
        break;
    }
}

inline void Convolution::prepareK3S2(const Workload& workload)
{
    K3S2Resources& resources = std::get<K3S2Resources>(prepared_);
    CooperativeMatrixResources& cm = resources.cm;
    TiledResources& tiled = resources.tiled;
    VAI_ASSERT(plan_.path == ConvPlan::Path::K3S2);
    const uint32_t spatialSize = workload.spatialSize();
    const auto [iDtype, oDtype, affineDtype, hasScale, epilogue] = makeFusionSpec(workload);
    const uint32_t requiredSubgroupSize = plan_.requiredSubgroupSize;
    const K3S2Plan& plan = plan_.k3s2;

    // firstLayer() implies a non-CM plan, so hoisting it above the CM
    // branch keeps the dispatch order intact.
    if (workload.wDtype == Dtype::Fp32 || plan.firstLayer())
    {
        tiled.main = makePipelineBinding(k3s2SpecializedPipeline(
            pipelinePool(), plan, epilogue, affineDtype, hasScale, hasBias(),
            plan.split() ? ResidualFusion{} : residualFusion(), workload.wDtype, iDtype, oDtype, requiredSubgroupSize));
        if (plan.split())
        {
            tiled.reduce = makePipelineBinding(bnReducePipeline(pipelinePool(), Dtype::Fp32, affineDtype, hasScale,
                                                                hasBias(), epilogue, residualFusion(), iDtype,
                                                                (spatialSize % 4u) == 0u, oDtype));
        }
        return;
    }

    if (plan_.useCM)
    {
        constexpr uint32_t cmRequiredSubgroupSize = 32u;
        tiled.main = makePipelineBinding(k3s2CMPipeline(pipelinePool(), plan, epilogue, affineDtype, hasScale,
                                                        hasBias(), plan.split() ? ResidualFusion{} : residualFusion(),
                                                        iDtype, oDtype, cmRequiredSubgroupSize));

        if (plan.packedWeights)
        {
            cm.transform = makePipelineBinding(
                cmTransformPipeline(pipelinePool(), 2u, cmRequiredSubgroupSize));

            if (plan.split())
            {
                tiled.reduce =
                    makePipelineBinding(bnReducePipeline(pipelinePool(), Dtype::Fp16, affineDtype, hasScale, hasBias(),
                                                         epilogue, residualFusion(), iDtype, true, oDtype));
            }
        }
        else if (plan.split())
        {
            tiled.reduce =
                makePipelineBinding(bnReducePipeline(pipelinePool(), oDtype, affineDtype, hasScale, hasBias(), epilogue,
                                                     residualFusion(), iDtype, (spatialSize % 4u) == 0u, oDtype));
        }
        return;
    }

    if (plan.splitK > 1u)
    {
        VAI_ASSERT(plan.splitK == 4u || plan.splitK == 8u);
        tiled.split = makePipelineBinding(k3s2SpecializedPipeline(pipelinePool(), plan, 0u, Dtype::Fp32, 0u, hasBias(),
                                                                  ResidualFusion{}, workload.wDtype, iDtype,
                                                                  Dtype::Fp32, requiredSubgroupSize, true));

        tiled.reduce =
            makePipelineBinding(bnReducePipeline(pipelinePool(), Dtype::Fp32, affineDtype, hasScale, hasBias(),
                                                 epilogue, residualFusion(), iDtype, (spatialSize % 4u) == 0u, oDtype));
        return;
    }

    tiled.main = makePipelineBinding(k3s2SpecializedPipeline(pipelinePool(), plan, epilogue, affineDtype, hasScale,
                                                             hasBias(), residualFusion(), workload.wDtype, iDtype,
                                                             oDtype, requiredSubgroupSize, true));
}

inline void Convolution::prepareImplicit(const Workload& workload)
{
    ImplicitResources& implicit = std::get<ImplicitResources>(prepared_);
    VAI_ASSERT(plan_.path == ConvPlan::Path::General);
    VAI_ASSERT(plan_.algorithm() == ConvAlgorithm::Implicit);
    const auto [iDtype, oDtype, affineDtype, hasScale, epilogue] = makeFusionSpec(workload);
    const ImplicitGemmPlan& plan = plan_.implicit;

    if (plan_.useCM)
    {
        implicit.gemm = makePipelineBinding(ConvCMPipeline(pipelinePool(), plan, workload, epilogue, affineDtype,
                                                           hasScale, hasBias(), residualFusion()));
        implicit.weightTransform = makePipelineBinding(
            cmTransformPipeline(pipelinePool(), 4u, 32u));
    }
    else
    {
        implicit.gemm = makePipelineBinding(ConvPipeline(pipelinePool(), plan, workload, epilogue, affineDtype,
                                                         hasScale, hasBias(), residualFusion(), false));
    }
}

inline void Convolution::preparePlan(const Workload& workload, const ConvPlan& nextPlan)
{
    const bool shapeSpecializedImplicit =
        nextPlan.path == ConvPlan::Path::General ||
        (nextPlan.path == ConvPlan::Path::K3S1 &&
         nextPlan.k3s1.algorithm == ConvAlgorithm::Implicit &&
         !nextPlan.useCM);
    const bool implicitShapeUnchanged =
        !shapeSpecializedImplicit || preparedShape_.matchShape(workload);

    if (planPrepared_ && plan_ == nextPlan && implicitShapeUnchanged)
        return;

    const bool pathChanged = std::holds_alternative<std::monostate>(prepared_) || plan_.path != nextPlan.path;
    plan_ = nextPlan;

    switch (plan_.path)
    {
    case ConvPlan::Path::K1S1:
        if (pathChanged)
            prepared_.emplace<K1S1Resources>();
        prepareK1S1(workload);
        break;
    case ConvPlan::Path::K3S1:
        if (pathChanged)
            prepared_.emplace<K3S1Resources>();
        prepareK3S1(workload);
        break;
    case ConvPlan::Path::K3S2:
        if (pathChanged)
            prepared_.emplace<K3S2Resources>();
        prepareK3S2(workload);
        break;
    case ConvPlan::Path::General:
        if (pathChanged)
            prepared_.emplace<ImplicitResources>();
        prepareImplicit(workload);
        break;
    }
    preparedShape_ = ShapeSpec::fromWorkload(workload);
    planPrepared_ = true;
}

inline void Convolution::prepareTensorStorage(const Workload& workload)
{
    const uint32_t Ci = workload.ci;
    const uint32_t Co = workload.co;
    const uint32_t Ho = workload.ho;
    const uint32_t Wo = workload.wo;
    const uint32_t B = workload.batch;

    const std::vector<uint32_t>& inputShape = (*this)["in0"].shape();
    std::vector<uint32_t> outputShape(inputShape.begin(), inputShape.end() - 3);
    outputShape.insert(outputShape.end(), {Co, Ho, Wo});
    (*this)["out0"] = Tensor(outputShape, workload.oDtype);
    if (config_.acceptResidual)
    {
        const Tensor& residual = (*this)["residual"];
        VAI_ASSERT(residual.shape() == (*this)["out0"].shape());
        VAI_ASSERT(residual.dtype() == workload.iDtype);
    }

    switch (plan_.path)
    {
    case ConvPlan::Path::K3S1:
        if (plan_.algorithm() != ConvAlgorithm::Winograd)
            break;
        {
            const uint32_t numTiles = workload.winogradTileCount();
            VAI_ASSERT(numTiles >= 16u);
            if (workload.wDtype == Dtype::Fp32)
            {
                (*this)["winV"] = Tensor({B * 36u * Ci, numTiles}, Dtype::Fp32);
                (*this)["winM"] = Tensor({B * 36u * Co, numTiles}, Dtype::Fp32);
            }
            else if (workload.wDtype == Dtype::Fp16)
            {
                if (plan_.useCM)
                {
                    if (plan_.vendorId == eva::VENDOR_ID::AMD)
                    {
                        const uint32_t winVCols = plan_.k3s1.inputReorder
                            ? CEIL_DIV(numTiles, 16u) * 16u
                            : numTiles;
                        (*this)["winV"] = Tensor({B * 36u * Ci, winVCols}, Dtype::Fp16);
                        (*this)["winM"] = Tensor({B * 36u * Co, numTiles}, Dtype::Fp16);
                    }
                    else if (plan_.vendorId == eva::VENDOR_ID::NVIDIA)
                    {
                        (*this)["winV"] = Tensor({B * 36u * Ci, numTiles + 1u}, Dtype::Fp16);
                        (*this)["winM"] = Tensor({B * 36u * Co, numTiles}, Dtype::Fp16);
                        if (plan_.k3s1.splitK > 1u)
                        {
                            (*this)["split"] = Tensor({B * 36u * plan_.k3s1.splitK * Co, numTiles}, Dtype::Fp32);
                        }
                    }
                    else
                    {
                        VAI_ASSERT(false && "K3S1 Winograd CM requires an AMD or NVIDIA plan");
                    }
                }
                else
                {
                    (*this)["winV"] = Tensor({B * 36u * Ci, numTiles}, Dtype::Fp16);
                    (*this)["winM"] = Tensor({B * 36u * Co, numTiles}, Dtype::Fp16);
                }
            }
            else
            {
                VAI_ASSERT(false && "K3S1 Winograd requires fp32 or fp16 weights");
            }
        }
        break;

    case ConvPlan::Path::K1S1:
        if (plan_.k1s1.splitK > 1u)
        {
            const bool cm = plan_.useCM;
            (*this)["split"] =
                Tensor({B * plan_.k1s1.splitK * Co, workload.spatialSize()}, cm ? Dtype::Fp16 : Dtype::Fp32);
        }
        break;

    case ConvPlan::Path::K3S2:
        if (plan_.k3s2.splitK > 1u)
        {
            const K3S2Plan& plan = plan_.k3s2;
            const bool packedCM = plan.packedWeights;
            const Dtype partialDtype = (plan_.wDtype == Dtype::Fp32)         ? Dtype::Fp32
                : packedCM                           ? Dtype::Fp16
                : plan_.useCM                        ? plan_.oDtype
                                                     : Dtype::Fp32;
            (*this)["split"] = Tensor({B * plan.splitK * Co, workload.spatialSize()}, partialDtype);
        }
        break;

    case ConvPlan::Path::General:
        break;
    }
}

inline void Convolution::prepare()
{
    const eva::Device currentDevice = device();
    const uint32_t vendorId = currentDevice.vendorID();
    const ConvolutionCaps caps{
        .hasCM = hasCooperativeMatrixUnit(vendorId, currentDevice.architectureID()),
        .accFP16 = supportsCMShape(currentDevice, Dtype::Fp16, Dtype::Fp16),
    };
    const Workload workload = makeWorkload();
    validateInputs(workload);

    const ConvPlan nextPlan = selectConvPlan(currentDevice, caps, workload);
    preparePlan(workload, nextPlan);
    prepareTensorStorage(workload);
}

inline bool Convolution::hasPendingWeightTransforms() const
{
    switch (plan_.path)
    {
    case ConvPlan::Path::K1S1:
    {
        const CooperativeMatrixResources& cm = std::get<K1S1Resources>(prepared_).cm;
        return plan_.useCM && plan_.k1s1.reorder && !cm.weightTransformed;
    }
    case ConvPlan::Path::K3S1:
    {
        const K3S1Resources& resources = std::get<K3S1Resources>(prepared_);
        const ImplicitResources& implicit = resources.implicit;
        const WinogradResources& winograd = resources.winograd;
        if (plan_.algorithm() == ConvAlgorithm::Implicit)
        {
            return plan_.useCM && !implicit.weightTransformed;
        }
        else if (plan_.algorithm() == ConvAlgorithm::Winograd)
        {
            return !winograd.weightTransformed ||
                (plan_.useCM &&
                 plan_.vendorId == eva::VENDOR_ID::NVIDIA &&
                 !winograd.cm.weightTransformed);
        }

        VAI_ASSERT(false && "invalid K3S1 convolution algorithm");
        return false;
    }
    case ConvPlan::Path::K3S2:
    {
        const CooperativeMatrixResources& cm = std::get<K3S2Resources>(prepared_).cm;
        return plan_.k3s2.packedWeights && !cm.weightTransformed;
    }
    case ConvPlan::Path::General:
    {
        const ImplicitResources& implicit = std::get<ImplicitResources>(prepared_);
        return plan_.useCM && !implicit.weightTransformed;
    }
    }
    VAI_ASSERT(false && "invalid convolution plan");
    return false;
}

inline bool Convolution::recordK3S1WeightTransforms(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    K3S1Resources& resources = std::get<K3S1Resources>(prepared_);
    ImplicitResources& implicit = resources.implicit;
    WinogradResources& winograd = resources.winograd;
    CooperativeMatrixResources& cm = winograd.cm;
    const uint32_t Ci = workload.ci;
    const uint32_t Co = workload.co;
    const eva::BufferRange weight = (*this)["weight"].buffer();
    const PushConstants constants = makePushConstants(workload);
    bool recorded = false;

    if (plan_.algorithm() == ConvAlgorithm::Implicit &&
        plan_.useCM &&
        !implicit.weightTransformed)
    {
        recorded |= recordFp16WeightTransform(
            cmdBuff, implicit.weightTransform, weight,
            implicit.transformedWeight,
            implicit.weightTransformed, workload,
            uint64_t(Co) * Ci * 9u);
    }

    if (plan_.algorithm() == ConvAlgorithm::Winograd)
    {
        if (!winograd.weightTransformed)
        {
            const size_t elementSize = plan_.wDtype == Dtype::Fp32 ? sizeof(float) : sizeof(uint16_t);
            recorded |= ensureWeightTransform(
                device(), cmdBuff, winograd.filterTransform, weight, winograd.transformedWeight, constants,
                size_t(Co) * Ci * 36u * elementSize, Co * Ci,
                winograd.weightTransformed);
        }

        if (plan_.useCM &&
            plan_.vendorId == eva::VENDOR_ID::NVIDIA && !cm.weightTransformed)
        {
            recorded |= recordFp16WeightTransform(
                cmdBuff, cm.transform, winograd.transformedWeight,
                cm.fragmentWeight, cm.weightTransformed, workload,
                uint64_t(36u) * Co * Ci);
        }
    }
    return recorded;
}

inline bool Convolution::recordK1S1WeightTransform(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    CooperativeMatrixResources& cm = std::get<K1S1Resources>(prepared_).cm;
    if (!plan_.useCM || !plan_.k1s1.reorder || cm.weightTransformed)
        return false;
    return recordFp16WeightTransform(
        cmdBuff, cm.transform, (*this)["weight"].buffer(),
        cm.fragmentWeight, cm.weightTransformed, workload,
        uint64_t(workload.co) * workload.ci);
}

inline bool Convolution::recordK3S2WeightTransform(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    CooperativeMatrixResources& cm = std::get<K3S2Resources>(prepared_).cm;
    if (!plan_.k3s2.packedWeights || cm.weightTransformed)
        return false;
    return recordFp16WeightTransform(
        cmdBuff, cm.transform, (*this)["weight"].buffer(),
        cm.fragmentWeight, cm.weightTransformed, workload,
        uint64_t(workload.co) * workload.ci * 9u);
}

inline bool Convolution::recordImplicitWeightTransform(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    ImplicitResources& implicit =
        std::get<ImplicitResources>(prepared_);
    if (!plan_.useCM || implicit.weightTransformed)
        return false;

    const uint32_t rawK = workload.ci * workload.kernel * workload.kernel;
    const uint32_t paddedK = CEIL_DIV(rawK, 16u) * 16u;
    const uint32_t paddedM = CEIL_DIV(workload.co, 16u) * 16u;
    return recordFp16WeightTransform(
        cmdBuff, implicit.weightTransform, (*this)["weight"].buffer(),
        implicit.transformedWeight, implicit.weightTransformed, workload,
        uint64_t(paddedM) * paddedK);
}

inline bool Convolution::recordWeightTransforms(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    switch (plan_.path)
    {
    case ConvPlan::Path::K1S1:
        return recordK1S1WeightTransform(cmdBuff, workload);
    case ConvPlan::Path::K3S1:
        return recordK3S1WeightTransforms(cmdBuff, workload);
    case ConvPlan::Path::K3S2:
        return recordK3S2WeightTransform(cmdBuff, workload);
    case ConvPlan::Path::General:
        return recordImplicitWeightTransform(cmdBuff, workload);
    }
    VAI_ASSERT(false && "invalid convolution plan");
    return false;
}

inline void Convolution::transformWeight(const Workload& workload)
{
    if (!hasPendingWeightTransforms())
        return;

    auto transformCmd = device().newCommandBuffer(eva::queue_compute);
    transformCmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
    const bool recorded = recordWeightTransforms(transformCmd, workload);
    VAI_ASSERT(recorded);
    transformCmd.end();
    device().queue() << transformCmd << eva::waiting;
}

inline void Convolution::runK3S1Implicit(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    K3S1Resources& resources = std::get<K3S1Resources>(prepared_);
    ImplicitResources& implicit = resources.implicit;
    VAI_ASSERT(config_.kernelSize == 3u && config_.stride == 1u);
    VAI_ASSERT(plan_.k3s1.algorithm == ConvAlgorithm::Implicit);
    Tensor& outTen = (*this)["out0"];
    Tensor& inTen = (*this)["in0"];
    const auto weight = (*this)["weight"].buffer();
    const AffineBindings affine = convAffineBindings(*this, hasBias(), hasScale());
    const uint32_t Co = workload.co;
    const eva::Descriptor shaderWeight = plan_.useCM
        ? eva::Descriptor(implicit.transformedWeight)
        : eva::Descriptor(weight);

    implicit.gemm.descriptorSet.write({
        outTen.buffer(),
        inTen.buffer(),
        shaderWeight,
        affine.bias,
        affine.scale,
        workload.residual ? (*this)["residual"].buffer() : weight,
    });

    const PushConstants constants = makePushConstants(workload);
    if (plan_.useCM)
    {
        const K3S1Plan& plan = plan_.k3s1;
        VAI_ASSERT(plan.shape.matchSpatial(workload));
        recordDispatch(
            cmdBuff, implicit.gemm, constants,
            plan.flatSpatial
                ? CEIL_DIV(workload.spatialSize(), plan.wgN)
                : workload.ho * CEIL_DIV(workload.wo, plan.wgN),
            CEIL_DIV(Co, plan.wgM));
    }
    else
    {
        const ImplicitGemmPlan& plan = plan_.implicit;
        VAI_ASSERT(!plan_.useCM);
        recordDispatch(
            cmdBuff, implicit.gemm, constants,
            CEIL_DIV(workload.spatialSize(), plan.wgN),
            CEIL_DIV(Co, plan.wgM));
    }
    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

inline void Convolution::runK3S1Winograd(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    K3S1Resources& resources = std::get<K3S1Resources>(prepared_);
    WinogradResources& winograd = resources.winograd;
    CooperativeMatrixResources& cm = winograd.cm;
    VAI_ASSERT(config_.kernelSize == 3u && config_.stride == 1u);
    const uint32_t Ci = workload.ci;
    const uint32_t Co = workload.co;
    const uint32_t numTiles = workload.winogradTileCount();
    Tensor& outTen = (*this)["out0"];
    Tensor& inTen = (*this)["in0"];
    Tensor& winV = (*this)["winV"];
    Tensor& winM = (*this)["winM"];
    const AffineBindings affine = convAffineBindings(*this, hasBias(), hasScale());
    const PushConstants constants = makePushConstants(workload, numTiles);
    const PushConstants transformConstants = makePushConstants(workload);
    const K3S1Plan& plan = plan_.k3s1;
    VAI_ASSERT(plan.algorithm == ConvAlgorithm::Winograd);

    auto recordInputTransform = [&](uint32_t tileCount) {
        winograd.inputTransform.descriptorSet.write({
            winV.buffer(),
            inTen.buffer(),
        });
        recordDispatch(cmdBuff, winograd.inputTransform, transformConstants, CEIL_DIV(Ci * tileCount, 128u));
        cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / winV.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
    };
    auto recordOutputTransform = [&](uint32_t workgroupWidth) {
        winograd.outputTransform.descriptorSet.write({
            outTen.buffer(),
            winM.buffer(),
            affine.bias,
            affine.scale,
            workload.residual ? (*this)["residual"].buffer()
                              : winM.buffer(),
        });
        recordDispatch(cmdBuff, winograd.outputTransform, transformConstants, CEIL_DIV(Co * numTiles, workgroupWidth));
        cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
    };
    auto recordWinograd = [&] {
        recordInputTransform(numTiles);
        winograd.gemm.descriptorSet.write({
            winM.buffer(),
            winV.buffer(),
            winograd.transformedWeight,
        });
        recordDispatch(cmdBuff, winograd.gemm, constants, CEIL_DIV(numTiles, plan.wgN), CEIL_DIV(Co, plan.wgM), 36u);
        cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / winM.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
        recordOutputTransform(plan.outputStoreMode ? 128u : 256u);
    };

    if (workload.wDtype == Dtype::Fp32)
    {
        recordWinograd();
        return;
    }
    else if (workload.wDtype == Dtype::Fp16)
    {
        if (plan_.useCM)
        {
            if (plan_.vendorId == eva::VENDOR_ID::AMD)
            {
                const uint32_t inputTiles = plan.inputReorder ? CEIL_DIV(numTiles, 16u) * 16u : numTiles;
                recordInputTransform(inputTiles);
                VAI_ASSERT(winM.dtype() == Dtype::Fp16);

                winograd.gemm.descriptorSet.write({
                    winM.buffer(),
                    winV.buffer(),
                    winograd.transformedWeight,
                });
                const uint32_t mainTiles = plan.hasTail() ? plan.tailNOffset : numTiles;
                recordDispatch(cmdBuff, winograd.gemm, constants, CEIL_DIV(mainTiles, plan.wgN), CEIL_DIV(Co, plan.wgM), 36u);

                if (plan.hasTail())
                {
                    // The tail pipeline intentionally borrows the main pipeline's descriptor set.
                    recordDispatch(
                        cmdBuff, winograd.cmTailSharedDescriptorPipeline, winograd.gemm.descriptorSet, constants,
                        CEIL_DIV(numTiles - plan.tailNOffset, kK3S1TailWgN),
                        CEIL_DIV(Co, plan.wgM), 36u);
                }
                cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / winM.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
                recordOutputTransform(256u);
            }
            else if (plan_.vendorId == eva::VENDOR_ID::NVIDIA)
            {
                recordInputTransform(numTiles);
                const bool split = plan.splitK > 1u;
                Tensor* partial = split ? &(*this)["split"] : nullptr;
                winograd.gemm.descriptorSet.write({
                    split ? partial->buffer() : winM.buffer(),
                    winV.buffer(),
                    cm.fragmentWeight,
                });

                const uint32_t wgN = plan.wgN;
                const uint32_t wgM = plan.wgM;
                const bool useTail = plan.hasTail();
                if (useTail)
                {
                    winograd.cmTailGemm.descriptorSet.write({
                        winM.buffer(),
                        winV.buffer(),
                        cm.fragmentWeight,
                    });
                }

                uint32_t gx = useTail ? numTiles / wgN : CEIL_DIV(numTiles, wgN);
                uint32_t gy = CEIL_DIV(Co, wgM);
                if (plan.swapXY)
                    std::swap(gx, gy);
                recordDispatch(cmdBuff, winograd.gemm, constants, gx, gy, 36u * plan.splitK);

                if (useTail)
                {
                    uint32_t tailGx = 1u;
                    uint32_t tailGy = CEIL_DIV(Co, wgM);
                    if (plan.swapXY)
                        std::swap(tailGx, tailGy);
                    recordDispatch(cmdBuff, winograd.cmTailGemm, constants, tailGx, tailGy, 36u);
                }

                if (split)
                {
                    recordConvSplitKReduce(cmdBuff, cm.reduce, winM.buffer(), *partial, affine, winM.buffer(),
                                           makePushConstants(workload, numTiles, plan.splitK),
                                           36u * Co * numTiles / 4u);
                }
                cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / winM.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
                recordOutputTransform(plan.outputStoreMode ? 128u : 256u);
            }
            else
            {
                VAI_ASSERT(false && "K3S1 Winograd CM requires an AMD or NVIDIA plan");
            }
        }
        else
        {
            recordWinograd();
        }
        return;
    }

    VAI_ASSERT(false && "K3S1 Winograd requires fp32 or fp16 weights");
}

inline void Convolution::runK1S1(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    VAI_ASSERT(plan_.algorithm() == ConvAlgorithm::Direct);
    K1S1Resources& resources = std::get<K1S1Resources>(prepared_);
    CooperativeMatrixResources& cm = resources.cm;
    TiledResources& tiled = resources.tiled;
    const uint32_t Co = workload.co;
    const uint32_t N = workload.spatialSize();
    Tensor& outTen = (*this)["out0"];
    Tensor& inTen = (*this)["in0"];
    const eva::BufferRange weight = (*this)["weight"].buffer();
    const AffineBindings affine = convAffineBindings(*this, hasBias(), hasScale());
    const uint32_t splitK = plan_.k1s1.splitK;
    const PushConstants constants = makePushConstants(workload);
    const eva::Descriptor residual =
        workload.residual ? eva::Descriptor((*this)["residual"].buffer()) : eva::Descriptor(weight);

    auto recordK1S1 = [&] {
        VAI_ASSERT(!plan_.useCM);
        if (splitK > 1u)
        {
            Tensor& partial = (*this)["split"];
            tiled.split.descriptorSet.write({
                partial.buffer(),
                inTen.buffer(),
                weight,
                affine.bias,
                affine.scale,
                residual,
            });

            recordDispatch(
                cmdBuff, tiled.split, constants,
                CEIL_DIV(N, plan_.k1s1.wgN),
                CEIL_DIV(Co, plan_.k1s1.wgM), splitK);

            recordConvSplitKReduce(cmdBuff, tiled.reduce, outTen.buffer(), partial, affine, residual,
                                   makePushConstants(workload, N, splitK), CEIL_DIV(Co * N, 4u));
            cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
            return;
        }

        tiled.main.descriptorSet.write({
            outTen.buffer(),
            inTen.buffer(),
            weight,
            affine.bias,
            affine.scale,
            residual,
        });
        recordDispatch(
            cmdBuff, tiled.main, constants,
            CEIL_DIV(N, plan_.k1s1.wgN),
            CEIL_DIV(Co, plan_.k1s1.wgM));
        cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
    };

    if (plan_.wDtype == Dtype::Fp32)
    {
        recordK1S1();
        return;
    }
    else if (plan_.wDtype == Dtype::Fp16)
    {
        if (plan_.useCM)
        {
            const K1S1Plan& plan = plan_.k1s1;
            Tensor* partial = plan.splitK > 1u ? &(*this)["split"] : nullptr;
            tiled.main.descriptorSet.write({
                partial ? partial->buffer() : outTen.buffer(),
                inTen.buffer(),
                plan.reorder ? eva::Descriptor(cm.fragmentWeight) : eva::Descriptor(weight),
                affine.bias,
                affine.scale,
                residual,
            });

            uint32_t gx = CEIL_DIV(N, plan.wgN);
            uint32_t gy = CEIL_DIV(Co, plan.wgM);
            if (plan.swapXY != 0u)
                std::swap(gx, gy);

            recordDispatch(cmdBuff, tiled.main, constants, gx, gy, plan.splitK);

            if (partial)
            {
                recordConvSplitKReduce(cmdBuff, cm.reduce, outTen.buffer(), *partial, affine, residual,
                                       makePushConstants(workload, N, plan.splitK), Co * N / 4u);
            }
            cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
        }
        else
        {
            recordK1S1();
        }
        return;
    }

    VAI_ASSERT(false && "K1S1 requires fp32 or fp16 weights");
}

inline void Convolution::runK3S2(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    VAI_ASSERT(plan_.algorithm() == ConvAlgorithm::Direct);
    K3S2Resources& resources = std::get<K3S2Resources>(prepared_);
    CooperativeMatrixResources& cm = resources.cm;
    TiledResources& tiled = resources.tiled;
    const uint32_t Co = workload.co;
    const uint32_t N = workload.spatialSize();
    Tensor& outTen = (*this)["out0"];
    Tensor& inTen = (*this)["in0"];
    const eva::BufferRange weight = (*this)["weight"].buffer();
    const AffineBindings affine = convAffineBindings(*this, hasBias(), hasScale());
    const K3S2Plan& plan = plan_.k3s2;

    if (!plan.firstLayer() && !plan.shape.empty())
    {
        VAI_ASSERT((plan.cooperativeMatrix() && plan.packedWeights) ? plan.shape.wo == workload.wo : plan.shape.matchShape(workload));
    }
    if (plan.split() && !plan_.useCM)
        VAI_ASSERT(plan.splitK == 4u || plan.splitK == 8u);

    PipelineBinding& dispatchPipeline = workload.wDtype == Dtype::Fp16 &&
            !plan_.useCM && plan.split()
        ? tiled.split
        : tiled.main;
    const eva::Descriptor dispatchWeight = plan.packedWeights
        ? eva::Descriptor(cm.fragmentWeight)
        : eva::Descriptor(weight);
    Tensor* partialOutput = plan.split()
        ? &(*this)["split"] : nullptr;
    const eva::Descriptor dispatchOutput = partialOutput != nullptr ? partialOutput->buffer()
                                                                   : outTen.buffer();

    const eva::Descriptor residual =
        workload.residual ? eva::Descriptor((*this)["residual"].buffer()) : eva::Descriptor(weight);
    dispatchPipeline.descriptorSet.write({
        dispatchOutput,
        inTen.buffer(),
        dispatchWeight,
        affine.bias,
        affine.scale,
        residual,
    });

    const K3S2DispatchGeometry dispatchGeometry = deriveK3S2DispatchGeometry(plan, workload);
    recordDispatch(
        cmdBuff, dispatchPipeline, makePushConstants(workload),
        dispatchGeometry.spatialGroups, dispatchGeometry.outputChannelGroups, dispatchGeometry.splitK);

    if (partialOutput != nullptr)
    {
        VAI_ASSERT(plan.splitK <= 16u);
        const uint32_t outputElements = Co * N;
        const uint32_t reduceWorkItems = plan.packedWeights ? outputElements / 4u : CEIL_DIV(outputElements, 4u);
        recordConvSplitKReduce(cmdBuff, tiled.reduce, outTen.buffer(), *partialOutput, affine, residual,
                               makePushConstants(workload, N, plan.splitK), reduceWorkItems);
    }
    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

inline void Convolution::runConv(eva::CommandBuffer cmdBuff, const Workload& workload)
{
    VAI_ASSERT(plan_.algorithm() == ConvAlgorithm::Implicit);
    ImplicitResources& implicit = std::get<ImplicitResources>(prepared_);
    const ImplicitGemmPlan& plan = plan_.implicit;

    Tensor& outTen = (*this)["out0"];
    Tensor& inTen = (*this)["in0"];
    const eva::BufferRange weight = (*this)["weight"].buffer();
    const AffineBindings affine = convAffineBindings(*this, hasBias(), hasScale());
    implicit.gemm.descriptorSet.write({
        outTen.buffer(),
        inTen.buffer(),
        plan_.useCM
            ? eva::Descriptor(implicit.transformedWeight)
            : eva::Descriptor(weight),
        affine.bias,
        affine.scale,
        workload.residual ? (*this)["residual"].buffer()
                          : eva::Descriptor(weight),
    });

    const PushConstants constants = makePushConstants(workload);
    recordDispatch(
        cmdBuff, implicit.gemm, constants,
        CEIL_DIV(workload.spatialSize(), plan.wgN),
        CEIL_DIV(workload.co, plan.wgM));
    cmdBuff.barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outTen.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
}

inline void Convolution::run(eva::CommandBuffer cmdBuff)
{
    const Workload workload = makeWorkload();
    transformWeight(workload);
    switch (plan_.path)
    {
        case ConvPlan::Path::K1S1:
            runK1S1(cmdBuff, workload);
            return;

        case ConvPlan::Path::K3S1:
            if (plan_.algorithm() == ConvAlgorithm::Implicit)
            {
                runK3S1Implicit(cmdBuff, workload);
            }
            else if (plan_.algorithm() == ConvAlgorithm::Winograd)
            {
                runK3S1Winograd(cmdBuff, workload);
            }
            return;

        case ConvPlan::Path::K3S2:
            runK3S2(cmdBuff, workload);
            return;

        case ConvPlan::Path::General:
            runConv(cmdBuff, workload);
            return;
    }
}

}   // namespace vai::node

#endif // VAI_NODES_CONVOLUTION_H
