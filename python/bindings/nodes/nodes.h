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

}   // namespace pyvai

#endif // PYVAI_NODES_NODES_H
