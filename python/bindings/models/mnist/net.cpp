#include "python/bindings/models/models.h"
#include "net.h"


namespace pyvai {

void bindMNIST(nb::module_& m)
{
    auto cls = nb::class_<mnist::Net, vai::NeuralNet>(m, "MNISTNet")
                   .def(
                       "__init__",
                       [](mnist::Net* self, vai::Dtype w_default_dtype, vai::Dtype o_default_dtype,
                          std::optional<eva::Device> device) {
                           // The core's convolution asserts on this pair.
                           if (w_default_dtype != vai::Dtype::Fp16 && o_default_dtype == vai::Dtype::Fp16)
                               throw nb::value_error("MNISTNet: fp32 weights cannot write an fp16 output");
                           new (self) mnist::Net(netConfig(w_default_dtype, o_default_dtype, device));
                       },
                       nb::arg("w_default_dtype") = vai::Dtype::Fp32, nb::arg("o_default_dtype") = vai::Dtype::Fp32,
                       nb::arg("device") = nb::none(),
                       "Build the MNIST classifier, with fp32 weights as examples/mnist/main.cpp. "
                       "in0 is [1, 28, 28], out0 the 10 class logits.");
    bindLoadWeights<mnist::Net>(cls, "MNISTNet", mnist::jroot::loadWeights,
                                "Load weights from a checkpoint folder (data/ckpts/mnist/jroot, which ships with "
                                "the repo): every *.safetensors in it is read.");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"in0"}, {"out0"});
}

}   // namespace pyvai
