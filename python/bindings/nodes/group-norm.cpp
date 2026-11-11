#include "nodes.h"
#include "vai/nodes/group-norm.h"
#include <optional>

namespace pyvai {

void bindGroupNorm(nb::module_& m)
{
    nb::class_<vai::node::GroupNorm, vai::Node>(m, "GroupNormNode")
        .def(
            "__init__",
            [](vai::node::GroupNorm* self, uint32_t groups, float eps, bool coeffs_as_inputs, bool accept_residual,
               vai::ResidualOrder residual_order, vai::Epilogue epilogue, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("GroupNormNode");
                if (groups == 0)
                    throw nb::value_error("GroupNormNode: groups must be >= 1");
                new (self) vai::node::GroupNorm(vai::node::GroupNorm::Config{
                    .groups = groups,
                    .eps = eps,
                    .coeffsAsInputs = coeffs_as_inputs,
                    .acceptResidual = accept_residual,
                    .residualOrder = residual_order,
                    .epilogue = epilogue,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("groups"), nb::arg("eps") = 1e-5f, nb::arg("coeffs_as_inputs") = false,
            nb::arg("accept_residual") = false, nb::arg("residual_order") = vai::ResidualOrder::AfterEpilogue,
            nb::arg("epilogue") = vai::Epilogue::None, nb::arg("o_dtype") = nb::none(),
            "Normalize a [..., C, H, W] input per sample and per group of C / groups "
            "consecutive channels: (x - mean) * rsqrt(var + eps) * scale[c] + shift[c]. "
            "scale / shift [C] are parameters, or inputs computed upstream with "
            "coeffs_as_inputs=True; shift applies when filled or wired. "
            "accept_residual opens the `residual` "
            "input (input dtype), added after the epilogue or before it per residual_order. "
            "Group statistics accumulate in fp32.");
}

}   // namespace pyvai
