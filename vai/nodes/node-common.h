#ifndef VAI_NODES_NODE_COMMON_H
#define VAI_NODES_NODE_COMMON_H

// Shared base for every builtin-node header: the Node/Tensor engine API, the
// std headers the node implementations use, the Epilogue enum, and the build
// macros (SHADER_SPV / CEIL_DIV / SPEC_ID_EPILOGUE). Each nodes/*.h includes
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

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define ROUNDUP_POW2(M, N) (((M) + (N)-1) & ~((N)-1))   // N must be a power of 2
#define SHADER_SPV(name) SHADER_OUTPUT_DIR "/" name ".comp.spv"
#define SPV_BLOB(name) eva::SpvBlob::readFrom(SHADER_SPV(name))

namespace vai {

// Defined in neural-net.cpp; bound by initVAIEngine(). Every node header's
// descriptor setup references it. For the device, use Node::device().
extern eva::DescriptorPool gVaiDestSetPool;

enum class Epilogue : uint32_t {
    None,
    Relu,
    Gelu,       // exact, through erf
    GeluTanh,   // the tanh approximation (gelu_pytorch_tanh)
    Silu,
    Sigmoid,
    Max
};

// Where Linear and Convolution fold their "residual" in, relative to the epilogue.
enum class ResidualOrder : uint32_t {
    AfterEpilogue = 0,    // a following Add (e.g. YOLO's bottleneck)
    BeforeEpilogue = 1,   // an Add ahead of the activation (e.g. ResNet's block, pi0's time MLP)
};

enum class RopeType {
    Default,   // reads base
    Linear,    // ... and factor
    Llama3,    // ... and the frequency bounds
};

// RoPE frequency parameters. They describe a model, not a layer, so a model
// config carries one and each attention passes it with its own headDim.
struct RopeConfig {
    RopeType type;
    float    base;
    float    factor;
    float    lowFreqFactor;
    float    highFreqFactor;
    uint32_t originalMaxPos;
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

} // namespace vai

#endif   // VAI_NODES_NODE_COMMON_H
