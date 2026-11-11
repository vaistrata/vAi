#ifndef PYVAI_NODES_NODES_H
#define PYVAI_NODES_NODES_H

#include "python/bindings/common.h"
#include "vai/engine/neural-net.h"
#include "vai/nodes/rope.h"
#include <stdexcept>
#include <string>

namespace pyvai {

// A node binding checks its arguments before it constructs. The core states
// those contracts with VAI_ASSERT, which aborts unless NDEBUG is set and is
// compiled out when it is, so a binding throws a Python exception first.
// The two checks here guard what the core does not even assert.

// A Python net is built node by node, so its dtypes are set where each node is
// made, in its binding: the key a model net's constructor would hold.
struct DtypeKeyHolder : vai::NeuralNet
{
    using vai::NeuralNet::dtypeKey;
};

inline vai::DtypeKey dtypeKey()
{
    return DtypeKeyHolder::dtypeKey();
}

// Node() latches the thread-local current net; a null one segfaults later. The
// core never clears it, so a net that is gone is told apart by asking nanobind
// whether the pointer still names a Python object.
inline void requireCurrentNet(const char* node)
{
    vai::NeuralNet* net = vai::NeuralNet::current();
    if (!net)
        throw std::runtime_error(
            std::string(node) + ": no active NeuralNet on this thread — "
            "construct nodes inside a NeuralNet (its __init__ makes it current)");
    if (!nb::find(net).is_valid())
        throw std::runtime_error(
            std::string(node) + ": the NeuralNet that was current no longer exists — "
            "construct nodes inside a NeuralNet (its __init__ makes it current)");
}

// The core computes RoPE frequencies unchecked (NaN, inf or a silent no-op);
// `what` is the caller's name for the config ("rope", "config....layer_cfg.rope_cfg").
inline void checkRopeConfig(const vai::RopeConfig& cfg, const char* who, const char* what)
{
    auto fail = [&](const char* msg) {
        throw nb::value_error((std::string(who) + ": " + what + "." + msg).c_str());
    };
    if (!(cfg.base > 0.0f))
        fail("base must be > 0");
    if (cfg.type != vai::RopeType::Default && !(cfg.factor > 0.0f))
        fail("factor must be > 0");
    if (cfg.type == vai::RopeType::Llama3 && cfg.lowFreqFactor == cfg.highFreqFactor)
        fail("low_freq_factor and high_freq_factor must differ");
}

// One per vai/nodes/<node>.h; list order must match bindNodes()'s call order.

// Shape / plumbing
void bindIdentity(nb::module_&);
void bindGate(nb::module_&);
void bindReshape(nb::module_&);
void bindFlatten(nb::module_&);
void bindConcat(nb::module_&);
void bindPermute(nb::module_&);
void bindSlice(nb::module_&);
void bindRowSliceScatter(nb::module_&);
void bindIndexCopy(nb::module_&);
void bindTypeCast(nb::module_&);

// Element-wise
void bindAdd(nb::module_&);
void bindMul(nb::module_&);
void bindGatedMul(nb::module_&);
void bindEleAffine(nb::module_&);
void bindActivation(nb::module_&);

// Vision
void bindConvolution(nb::module_&);
void bindPool2d(nb::module_&);
void bindNearestUpsample(nb::module_&);

// Linear / normalization
void bindLinear(nb::module_&);
void bindLayerNorm(nb::module_&);
void bindGroupNorm(nb::module_&);
void bindRmsNorm(nb::module_&);

// Transformer
void bindTokenEmbedding(nb::module_&);
void bindTimeEmbedding(nb::module_&);
void bindCyclicTableReader(nb::module_&);
void bindRope(nb::module_&);
void bindAttention(nb::module_&);

// Output heads
void bindArgmax(nb::module_&);
void bindTopKSampling(nb::module_&);

}   // namespace pyvai

#endif // PYVAI_NODES_NODES_H
