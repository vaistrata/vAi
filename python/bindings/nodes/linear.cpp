#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/linear.h"
#include <cstdint>

namespace pyvai {

static void bindLinear(nb::module_& m)
{
    nb::enum_<vai::ResidualOrder>(m, "ResidualOrder",
        "Where LinearNode adds its `residual` input, relative to the epilogue.")
        .value("AFTER_EPILOGUE",  vai::ResidualOrder::AfterEpilogue,
               "s * epilogue(x W^T + bias) + residual, a following add.")
        .value("BEFORE_EPILOGUE", vai::ResidualOrder::BeforeEpilogue,
               "s * epilogue(x W^T + bias + residual), a bias that changes per run.");

    using C = vai::node::Linear::Config;
    bindNode<vai::node::Linear>(
        m, "LinearNode",
        "Dense layer: out0 = scalar_scale * epilogue(in0 W^T + bias) + residual, "
        "or with the residual inside the epilogue per residual_order. has_bias "
        "opens the `bias` slot (w_dtype) and accept_residual the `residual` input "
        "(o_dtype); each is [..., out_dim] or rows that repeat over the output, "
        "such as [out_dim]. split_size is the K length per split-K span (0 = the "
        "kernel's heuristic, otherwise >= 32) and only matters past 8 input rows. "
        "acc_dtype is the matmul accumulator (None = the net's w_default_dtype); "
        "an fp16 one falls back to fp32 where the device cannot accumulate in fp16.",
        required("in_dim", &C::inDim), required("out_dim", &C::outDim), field("has_bias", &C::hasBias),
        field("accept_residual", &C::acceptResidual), field("residual_order", &C::residualOrder),
        field("epilogue", &C::epilogue), field("scalar_scale", &C::scalarScale), field("w_dtype", &C::wDtype),
        field("o_dtype", &C::oDtype),
        // The core asserts (aborts) on the tile path.
        field("split_size", &C::splitSize).check([](uint32_t s) { return s == 0 || s >= 32; }, "must be 0 or >= 32"),
        field("acc_dtype", &C::accDtype));
}
PYVAI_NODE_BINDING(bindLinear);

}   // namespace pyvai
