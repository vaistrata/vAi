#include "python/bindings/module.h"
#include "forward.h"
#include "vai/engine/neural-net.h"
#include <optional>
#include <string>
#include <utility>


namespace pyvai {

void bindNeuralNet(nb::module_& m)
{
    // dynamic_attr: the boundary registry, input staging and default outputs
    // live on the net object (forward.h).
    nb::class_<vai::NeuralNet>(m, "NeuralNet", nb::dynamic_attr())
        .def(
            "__init__",
            [](vai::NeuralNet* self, vai::Dtype w_default_dtype, vai::Dtype o_default_dtype,
               size_t upload_chunk_size_mib, uint32_t buffer_pool_log_level, std::optional<eva::Device> device) {
                vai::NeuralNetConfig config = netConfig(w_default_dtype, o_default_dtype, device);
                config.uploadChunkSizeMiB = upload_chunk_size_mib;
                config.bufferPoolLogLevel = buffer_pool_log_level;
                new (self) vai::NeuralNet(config);
            },
            nb::arg("w_default_dtype") = vai::Dtype::Fp16, nb::arg("o_default_dtype") = vai::Dtype::Fp32,
            nb::arg("upload_chunk_size_mib") = vai::detail::kDefaultUploadChunkMiB,
            nb::arg("buffer_pool_log_level") = 0u, nb::arg("device") = nb::none(),
            "Build an empty net. `w_default_dtype` is the fallback for layer "
            "parameters (weights, KV cache) and `o_default_dtype` for layer "
            "outputs; a layer may override either, and takes its input dtype from "
            "whatever its producer wrote. `device` defaults to "
            "get_vai_device(). Nodes attach to the net constructed last on the "
            "thread, so finish building one net before constructing the next.")

        .def_prop_ro("w_default_dtype", [](const vai::NeuralNet& self) { return self.config().wDefaultDtype; })
        .def_prop_ro("o_default_dtype", [](const vai::NeuralNet& self) { return self.config().oDefaultDtype; })
        .def_prop_ro(
            "device", [](const vai::NeuralNet& self) { return self.device(); },
            "The Device this net lives on, for whatever must share it, e.g. "
            "KVCacheCounter(device=net.device).")

        // input()/output() wrappers carry `_pyvai_net`, a strong ref that also
        // anchors them for the DSL lifetime model (graph-flow.cpp).
        .def(
            "input",
            [](nb::handle_t<vai::NeuralNet> self, const std::string& name) -> nb::object {
                auto& net = nb::cast<vai::NeuralNet&>(self);
                nb::object o = nb::cast(net.input(name), nb::rv_policy::reference);
                o.attr("_pyvai_net") = self;
                recordBoundary(self, "_pyvai_input_names", name);
                return o;
            },
            nb::arg("name") = "in0", nb::sig("def input(self, name: str = 'in0') -> InputNode"),
            "The named input boundary node, created on first use. Names are the "
            "keys of __call__'s inputs.")
        .def(
            "output",
            [](nb::handle_t<vai::NeuralNet> self, const std::string& name) -> nb::object {
                auto& net = nb::cast<vai::NeuralNet&>(self);
                nb::object o = nb::cast(net.output(name), nb::rv_policy::reference);
                o.attr("_pyvai_net") = self;
                recordBoundary(self, "_pyvai_output_names", name);
                return o;
            },
            nb::arg("name") = "out0", nb::sig("def output(self, name: str = 'out0') -> OutputNode"),
            "The named output boundary node, created on first use. Names are the "
            "keys of __call__'s result.")

        .def_prop_ro(
            "input_names", [](nb::handle self) { return sortedNames(inputNamesOf(self)); },
            "Sorted input-boundary names — the keys __call__ accepts in `inputs`.")
        .def_prop_ro(
            "output_names", [](nb::handle self) { return sortedNames(outputNamesOf(self)); },
            "Sorted output-boundary names — the keys of __call__'s result.")

        .def("invalidate_command_buffer", &vai::NeuralNet::invalidateCommandBuffer)

        .def(
            "sync",
            [](vai::NeuralNet& self) {
                nb::gil_scoped_release rel;
                self.sync();
            },
            "Block until every forward submitted so far has completed.")

        .def(
            "__call__",
            [](nb::handle_t<vai::NeuralNet> self, nb::args args, nb::kwargs kwargs) -> nb::object {
                auto& net = nb::cast<vai::NeuralNet&>(self);
                return callForward(self, args, kwargs,
                    [&](vai::TensorDict in, vai::TensorRefDict refs, bool syncPrev) {
                        net(std::move(in), std::move(refs), syncPrev);
                    });
            },
            nb::sig(callSig),
            "Run one forward: net(x), net({name: x}) or net(name=x). Returns "
            "the output Tensor, or a name -> Tensor dict when there are several.\n\n"
            "Naming a subset of the inputs skips the branches only the omitted "
            "ones reach; such an output, like a gate-skipped one, comes back as an "
            "empty Tensor.\n\n"
            "Outputs are reused across calls and filled asynchronously (to_numpy() "
            "waits), so copy a result needed past the next call. `outputs=` "
            "supplies the receiving Tensors; keep the same objects and input set "
            "from call to call, or the commands re-record. `sync_prev=False` skips "
            "the wait for earlier forwards and is only safe after sync().")

        .def("__repr__", [](nb::handle self) -> std::string {
            std::string typeName = nb::cast<std::string>(self.type().attr("__name__"));
            return typeName + "()";
        });
}

}   // namespace pyvai
