#ifndef VAI_NODES_COMMON_H
#define VAI_NODES_COMMON_H

// Shared base for every builtin-node header: the Node/Tensor engine API, the
// std headers the node implementations use, the vai::ActType enum, and the build
// macros (SHADER_SPV / CEIL_DIV / SPEC_ID_ACT_TYPE). Each nodes/*.h includes
// this so the domain headers are self-contained and independent of one another.
#include "vai/engine/neural-net.h"
#include "vai/engine/graph-flow.h"

#include <tuple>
#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>
#include "vai/shaders/common/spec_constants.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define ROUNDUP(M, N) (CEIL_DIV(M, N) * (N))
#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))   // N must be a power of 2
#define SHADER_SPV(name) SHADER_OUTPUT_DIR "/" name ".comp.spv"
#define SPV_BLOB(name) eva::SpvBlob::readFrom(SHADER_SPV(name))

namespace vai {

// Defined in neural-net.cpp; bound by initVAIEngine(). Every node header's
// descriptor setup references it. For the device, use Node::device().
extern eva::DescriptorPool gVaiDestSetPool;

enum class ActType : uint32_t {
    None,
    Relu,
    Gelu,       // exact, through erf
    GeluTanh,   // the tanh approximation (gelu_pytorch_tanh)
    Silu,
    Sigmoid,
    Max
};

// Where Linear and Convolution fold their "residual" in, relative to the activation.
enum class ResidualOrder : uint32_t {
    AfterActivation = 0,    // a following Add (e.g. YOLO's bottleneck)
    BeforeActivation = 1,   // an Add ahead of the activation (e.g. ResNet's block, pi0's time MLP)
};

// How each rotated pair gets its position; values match rope.comp's POS_*.
enum class RopePosType : uint32_t {
    Sequential,   // token tq at a start position + tq
    Grid2D,       // a ViT's raster grid: half the pairs turn by the row, half by the column
    MRoPE,        // (t, h, w) per token from the model; the pairs alternate t, h, w
};

// RoPE parameters, HF's rope_parameters. They describe a model, not a layer, so a
// model config carries one and each attention passes it with its own headDim.
struct RopeConfig
{
    RopePosType posType = RopePosType::Sequential;
    float base;
    float factor = 1.0f;             // 1: unscaled; else divides the frequencies (Llama3: the long wavelengths)
    uint32_t originalMaxPos = 0;     // 0: no Llama3 band; else Llama3 scales by the bounds below
    float lowFreqFactor = 1.0f;      // Llama3
    float highFreqFactor = 4.0f;     // Llama3
    std::array<uint32_t, 3> mropeSection = {};   // MRoPE: pairs on (t, h, w), interleaved
};


// The product of the axes ahead of the last innerAxes: the rows, sequences or
// images a node sweeps in one dispatch.
inline uint32_t outerSize(const std::vector<uint32_t>& shape, size_t innerAxes = 1)
{
    uint32_t n = 1;
    for (size_t i = 0; i + innerAxes < shape.size(); ++i)
        n *= shape[i];
    return n;
}

}   // namespace vai

#endif   // VAI_NODES_COMMON_H
