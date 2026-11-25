#include "python/bindings/nodes/bind-node.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/rope.h"
#include <nanobind/stl/array.h>
#include <array>
#include <cstdint>
#include <optional>

namespace pyvai {

namespace {

// The core struct defaults everything but the base, so it is spelled here.
const vai::RopeConfig plainRope = {
    .base = 10000.0f,
};


// Bound before RopeNode, whose signature refers to them.
void bindRopeConfig(nb::module_& m)
{
    nb::enum_<vai::RopePosType>(
        m,
        "RopePosType",
        "Where each rotated pair gets its position: SEQUENTIAL, a start position plus the "
        "token's index; GRID2D, a ViT's raster grid, half the pairs turning by the row and "
        "half by the column; MROPE, (t, h, w) per token, the pairs alternating t, h, w.")
        .value("SEQUENTIAL", vai::RopePosType::Sequential)
        .value("GRID2D", vai::RopePosType::Grid2D)
        .value("MROPE", vai::RopePosType::MRoPE);

    auto cls =
        nb::class_<vai::RopeConfig>(m, "RopeConfig")
            .def(
                "__init__",
                [](vai::RopeConfig* self,
                   vai::RopePosType pos_type,
                   float base,
                   float factor,
                   uint32_t original_max_pos,
                   float low_freq_factor,
                   float high_freq_factor,
                   std::array<uint32_t, 3> mrope_section) {
                    new (self) vai::RopeConfig{
                        .posType = pos_type,
                        .base = base,
                        .factor = factor,
                        .originalMaxPos = original_max_pos,
                        .lowFreqFactor = low_freq_factor,
                        .highFreqFactor = high_freq_factor,
                        .mropeSection = mrope_section,
                    };
                },
                nb::arg("pos_type") = plainRope.posType,
                nb::arg("base") = plainRope.base,
                nb::arg("factor") = plainRope.factor,
                nb::arg("original_max_pos") = plainRope.originalMaxPos,
                nb::arg("low_freq_factor") = plainRope.lowFreqFactor,
                nb::arg("high_freq_factor") = plainRope.highFreqFactor,
                nb::arg("mrope_section") = plainRope.mropeSection,
                "RoPE parameters of a model, HF's rope_parameters. factor 1 leaves the "
                "frequencies alone, any other divides them; a nonzero original_max_pos makes "
                "it Llama 3's band, read with the two frequency bounds. mrope_section gives "
                "MROPE's pairs on (t, h, w). Consumers require base > 0, factor > 0, and "
                "low_freq_factor != high_freq_factor when original_max_pos is set.")
            .def_rw("pos_type", &vai::RopeConfig::posType)
            .def_rw("base", &vai::RopeConfig::base)
            .def_rw("factor", &vai::RopeConfig::factor)
            .def_rw("original_max_pos", &vai::RopeConfig::originalMaxPos)
            .def_rw("low_freq_factor", &vai::RopeConfig::lowFreqFactor)
            .def_rw("high_freq_factor", &vai::RopeConfig::highFreqFactor)
            .def_rw("mrope_section", &vai::RopeConfig::mropeSection)
            .def(
                "__eq__",
                [](const vai::RopeConfig& self, const vai::RopeConfig& other) {
                    return self.posType == other.posType && self.base == other.base && self.factor == other.factor &&
                           self.originalMaxPos == other.originalMaxPos && self.lowFreqFactor == other.lowFreqFactor &&
                           self.highFreqFactor == other.highFreqFactor && self.mropeSection == other.mropeSection;
                },
                nb::arg("other"),
                nb::is_operator())
            .def("__repr__", [](const vai::RopeConfig& c) {
                return nb::str("RopeConfig(pos_type={}, base={}, factor={}, original_max_pos={}, "
                               "low_freq_factor={}, high_freq_factor={}, mrope_section=[{}, {}, {}])")
                    .format(c.posType, c.base, c.factor, c.originalMaxPos, c.lowFreqFactor, c.highFreqFactor,
                            c.mropeSection[0], c.mropeSection[1], c.mropeSection[2]);
            });
    // A mutable value type with __eq__ is unhashable, as Python's rule has it.
    cls.attr("__hash__") = nb::none();
}

}   // namespace


static void bindRope(nb::module_& m)
{
    bindRopeConfig(m);

    using C = vai::node::Rope::Config;
    bindNode<vai::node::Rope>(m,
                              "RopeNode",
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
                                      ""))
        .def(
            "set_pos_param",
            [](vai::node::Rope& self, KVCacheCounter& pos_param) { self.setPosParam(pos_param.buf); },
            nb::arg("pos_param"),
            // The position source owns its buffer — pin it to the node.
            nb::keep_alive<1, 2>(),
            "Bind the uint the position type reads: the first input token's position, "
            "or the grid width under AXIAL2D. Required before the first forward.");
}
PYVAI_NODE_BINDING(bindRope);

}   // namespace pyvai
