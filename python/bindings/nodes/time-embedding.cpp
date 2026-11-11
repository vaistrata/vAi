#include "nodes.h"
#include "vai/nodes/time-embedding.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindTimeEmbedding(nb::module_& m)
{
    nb::class_<vai::node::TimeEmbedding, vai::Node>(m, "TimeEmbeddingNode")
        .def(
            "__init__",
            [](vai::node::TimeEmbedding* self, uint32_t embed_dim, float min_period, float max_period,
               uint32_t num_steps, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("TimeEmbeddingNode");
                if (embed_dim < 4 || embed_dim % 2 != 0)
                    throw nb::value_error("TimeEmbeddingNode: embed_dim must be even and >= 4");
                if (!(min_period > 0.0f) || !(max_period > min_period))
                    throw nb::value_error("TimeEmbeddingNode: need 0 < min_period < max_period");
                if (num_steps < 1)
                    throw nb::value_error("TimeEmbeddingNode: num_steps must be >= 1");
                new (self) vai::node::TimeEmbedding(vai::node::TimeEmbedding::Config{
                    .embedDim = embed_dim,
                    .minPeriod = min_period,
                    .maxPeriod = max_period,
                    .numSteps = num_steps,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("embed_dim"), nb::arg("min_period"), nb::arg("max_period"), nb::arg("num_steps"),
            nb::arg("o_dtype") = nb::none(),
            "Sinusoidal embedding of a timestep the node counts down itself: forward "
            "k after reset_time() embeds t = 1 - (k % num_steps) / num_steps. in0 "
            "[..., rows, *] is read for its shape alone (rank >= 2). out0 [..., rows, "
            "embed_dim] repeats one vector on every row: sin(2*pi*t/p_i) in the first "
            "half, cos in the second, p_i geometric from min_period to max_period.")
        .def("reset_time", &vai::node::TimeEmbedding::resetTime,
             "Start the count over, so the next forward embeds t = 1. Not "
             "synchronized: call net.sync() first if a forward may still be running.");
}

}   // namespace pyvai
