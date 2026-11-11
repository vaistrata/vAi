#include "nodes.h"
#include "vai/nodes/rms-norm.h"
#include <optional>

namespace pyvai {

void bindRmsNorm(nb::module_& m)
{
    nb::class_<vai::node::RmsNorm, vai::Node>(m, "RmsNormNode")
        .def(
            "__init__",
            [](vai::node::RmsNorm* self, float eps, bool coeffs_as_inputs, bool accept_residual,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("RmsNormNode");
                new (self) vai::node::RmsNorm(vai::node::RmsNorm::Config{
                    .eps = eps,
                    .coeffsAsInputs = coeffs_as_inputs,
                    .acceptResidual = accept_residual,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("eps") = 1e-5f, nb::arg("coeffs_as_inputs") = false, nb::arg("accept_residual") = false,
            nb::arg("o_dtype") = nb::none(),
            "RMS normalization of [..., feature] over the feature axis: x * "
            "rsqrt(mean(x^2) + eps) * scale + shift. The `scale` / `shift` slots "
            "[feature] are parameters, or inputs computed upstream with "
            "coeffs_as_inputs=True; each applies when filled or wired. accept_residual "
            "opens the `residual` input (in0's shape and dtype), added to the output. "
            "Row statistics accumulate in fp32 whatever the I/O dtypes.");
}

}   // namespace pyvai
