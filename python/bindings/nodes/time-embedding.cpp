#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/time-embedding.h"
#include <cstdint>

namespace pyvai {

static void bindTimeEmbedding(nb::module_& m)
{
    using C = vai::node::TimeEmbedding::Config;
    bindNode<vai::node::TimeEmbedding>(
        m,
        "TimeEmbeddingNode",
        "Sinusoidal embedding of a timestep the node counts down itself: forward "
        "k after reset_time() embeds t = 1 - (k % num_steps) / num_steps. in0 "
        "[..., rows, *] is read for its shape alone (rank >= 2). out0 [..., rows, "
        "embed_dim] repeats one vector on every row: sin(2*pi*t/p_i) in the first "
        "half, cos in the second, p_i geometric from min_period to max_period.",
        required("embed_dim", &C::embedDim)
            .check([](uint32_t d) { return d >= 4 && d % 2 == 0; }, "must be even and >= 4"),
        required("min_period", &C::minPeriod).check([](float p) { return p > 0.0f; }, "must be > 0"),
        required("max_period", &C::maxPeriod)
            .check([](float p, const C& c) { return p > c.minPeriod; }, "must be > min_period"),
        required("num_steps", &C::numSteps).check(atLeast(1u), "must be >= 1"))
        .def("reset_time",
             &vai::node::TimeEmbedding::resetTime,
             "Start the count over, so the next forward embeds t = 1. Not "
             "synchronized: call net.sync() first if a forward may still be running.");
}
PYVAI_NODE_BINDING(bindTimeEmbedding);

}   // namespace pyvai
