#include "python/bindings/nodes/bind-node.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/rope.h"
#include <cstdint>
#include <optional>

namespace pyvai {

namespace {

// The core struct has no defaults, so plain RoPE is spelled here; the scaling
// knobs sit at Llama 3's values, idle until a type reads them.
const vai::RopeConfig plainRope = {
    .type           = vai::RopeType::Default,
    .base           = 10000.0f,
    .factor         = 1.0f,
    .lowFreqFactor  = 1.0f,
    .highFreqFactor = 4.0f,
    .originalMaxPos = 8192,
};


// Bound before RopeNode, whose signature refers to them.
void bindRopeConfig(nb::module_& m)
{
    nb::enum_<vai::RopeType>(m, "RopeType",
        "How RopeConfig scales the frequencies: DEFAULT reads base only, LINEAR "
        "divides every frequency by factor, LLAMA3 divides only the long "
        "wavelengths and interpolates the band between the two frequency bounds.")
        .value("DEFAULT", vai::RopeType::Default)
        .value("LINEAR",  vai::RopeType::Linear)
        .value("LLAMA3",  vai::RopeType::Llama3);

    auto cls = nb::class_<vai::RopeConfig>(m, "RopeConfig")
        .def(
            "__init__",
            [](vai::RopeConfig* self, vai::RopeType type,
               float base, float factor, float low_freq_factor, float high_freq_factor,
               uint32_t original_max_pos)
            {
                new (self) vai::RopeConfig{
                    .type           = type,
                    .base           = base,
                    .factor         = factor,
                    .lowFreqFactor  = low_freq_factor,
                    .highFreqFactor = high_freq_factor,
                    .originalMaxPos = original_max_pos,
                };
            },
            nb::arg("type")             = plainRope.type,
            nb::arg("base")             = plainRope.base,
            nb::arg("factor")           = plainRope.factor,
            nb::arg("low_freq_factor")  = plainRope.lowFreqFactor,
            nb::arg("high_freq_factor") = plainRope.highFreqFactor,
            nb::arg("original_max_pos") = plainRope.originalMaxPos,
            "RoPE frequencies of a model. factor, the frequency bounds and "
            "original_max_pos are read by LINEAR / LLAMA3 only. Consumers require "
            "base > 0, factor > 0 where it is read, and low_freq_factor != "
            "high_freq_factor for LLAMA3."
        )
        .def_rw("type",             &vai::RopeConfig::type)
        .def_rw("base",             &vai::RopeConfig::base)
        .def_rw("factor",           &vai::RopeConfig::factor)
        .def_rw("low_freq_factor",  &vai::RopeConfig::lowFreqFactor)
        .def_rw("high_freq_factor", &vai::RopeConfig::highFreqFactor)
        .def_rw("original_max_pos", &vai::RopeConfig::originalMaxPos)
        .def("__eq__",
             [](const vai::RopeConfig& self, const vai::RopeConfig& other) {
                 return self.type == other.type && self.base == other.base
                     && self.factor == other.factor
                     && self.lowFreqFactor == other.lowFreqFactor
                     && self.highFreqFactor == other.highFreqFactor
                     && self.originalMaxPos == other.originalMaxPos;
             },
             nb::arg("other"), nb::is_operator())
        .def("__repr__", [](const vai::RopeConfig& c) {
            return nb::str("RopeConfig(type={}, base={}, factor={}, low_freq_factor={}, "
                           "high_freq_factor={}, original_max_pos={})")
                .format(c.type, c.base, c.factor, c.lowFreqFactor, c.highFreqFactor,
                        c.originalMaxPos);
        });
    // A mutable value type with __eq__ is unhashable, as Python's rule has it.
    cls.attr("__hash__") = nb::none();
}

}   // namespace


static void bindRope(nb::module_& m)
{
    bindRopeConfig(m);

    using C = vai::node::Rope::Config;
    bindNode<vai::node::Rope>(m, "RopeNode",
                              "Rotary position embedding of Q or K [..., H, Tq, head_dim], head_dim "
                              "even and >= 2. out0 has in0's shape.",
                              required("head_dim", &C::headDim)
                                  .check([](uint32_t d) { return d >= 2 && d % 2 == 0; }, "must be even and >= 2"),
                              // checkRopeConfig raises its own error, naming the field it rejects.
                              field("rope", &C::cfg, plainRope)
                                  .check(
                                      [](const vai::RopeConfig& rope) {
                                          checkRopeConfig(rope, "RopeNode", "rope");
                                          return true;
                                      },
                                      ""),
                              field("o_dtype", &C::oDtype))
        .def(
            "set_kv_cache_counter",
            [](vai::node::Rope& self, KVCacheCounter& counter) { self.setKvCacheCounter(counter.buf); },
            nb::arg("counter"),
            // The counter's dtor destroys its buffer — pin it to the node.
            nb::keep_alive<1, 2>(),
            "Bind the KV-cache token counter that supplies this forward's "
            "starting position. Required before the first forward.");
}
PYVAI_NODE_BINDING(bindRope);

}   // namespace pyvai
