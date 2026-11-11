#include "nodes.h"
#include "vai/nodes/linear.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindLinear(nb::module_& m)
{
    nb::class_<vai::node::Linear, vai::Node>(m, "LinearNode")
        .def(
            "__init__",
            [](vai::node::Linear* self, uint32_t in_dim, uint32_t out_dim, bool accept_residual,
               vai::ResidualOrder residual_order, vai::Epilogue epilogue, float scalar_scale,
               std::optional<vai::Dtype> o_dtype, uint32_t split_size, std::optional<vai::Dtype> acc_dtype) {
                requireCurrentNet("LinearNode");
                if (split_size > 0 && split_size < 32)   // core asserts (aborts) on the tile path
                    throw nb::value_error("LinearNode: split_size must be 0 or >= 32");
                new (self) vai::node::Linear(vai::node::Linear::Config{
                    .inDim = in_dim,
                    .outDim = out_dim,
                    .acceptResidual = accept_residual,
                    .residualOrder = residual_order,
                    .epilogue = epilogue,
                    .scalarScale = scalar_scale,
                    .splitSize = split_size,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
                self->setAccDtype(acc_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("in_dim"), nb::arg("out_dim"), nb::arg("accept_residual") = false,
            nb::arg("residual_order") = vai::ResidualOrder::AfterEpilogue, nb::arg("epilogue") = vai::Epilogue::None,
            nb::arg("scalar_scale") = 1.0f, nb::arg("o_dtype") = nb::none(), nb::arg("split_size") = 0u,
            nb::arg("acc_dtype") = nb::none(),
            "Dense layer: out0 = scalar_scale * epilogue(in0 W^T + bias) + residual, "
            "or with the residual inside the epilogue per residual_order. The bias applies "
            "when the `bias` slot is filled; accept_residual opens the `residual` "
            "input (o_dtype). Each is [..., out_dim] or rows that repeat over the output, "
            "such as [out_dim]. split_size is the K length per split-K span (0 = the "
            "kernel's heuristic, otherwise >= 32) and only matters past 8 input rows. "
            "acc_dtype is the matmul accumulator (None = the weight's dtype); "
            "an fp16 one falls back to fp32 where the device cannot accumulate in fp16."
            " On the Intel path the first prepare packs \"weight\" in place to "
            "[ceil(N / 32), ceil(K / 2), 64].");
}

}   // namespace pyvai
