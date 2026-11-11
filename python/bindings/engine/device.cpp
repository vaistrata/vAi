#include "python/bindings/module.h"
#include "vai/engine/neural-net.h"
#include "vai/nodes/node-common.h"   // gVaiDestSetPool
#include <cstdlib>
#include <optional>
#include <string>


namespace pyvai {

namespace {

// Windows: ExitProcess kills the driver's threads before the DLL detach that
// runs eva's static ~Runtime, whose vkDestroyDevice then crashes. Destroying
// the device at Py_AtExit makes ~Runtime a no-op. Harmless on Linux.
void shutdownEngineAtExit()
{
    if (!vai::gVaiDestSetPool)        // init probe — getVaiDevice() would lazy-init
        return;
    vai::gVaiDestSetPool.destroy();   // device child — destroy first
    vai::getVaiDevice().destroy();    // idempotent; waits the GPU idle inside
}

}   // namespace


void bindDevice(nb::module_& m)
{
    Py_AtExit(shutdownEngineAtExit);

    // An opaque handle: out of get_vai_device() or NeuralNet.device, into device=.
    nb::class_<eva::Device>(m, "Device")
        .def("__bool__",
             [](const eva::Device& self) { return bool(self); },
             "True when this handle is bound to a live device.")
        .def("__eq__",
             [](const eva::Device& self, nb::handle other) -> bool {
                 if (!nb::isinstance<eva::Device>(other))
                     return false;
                 return self == nb::cast<eva::Device>(other);
             },
             nb::arg("other"))
        .def("__repr__", [](const eva::Device& self) -> std::string {
            return self ? "Device(<bound>)" : "Device(<null>)";
        });

    m.def("init_vai_engine",
        []() { vai::initVAIEngine(); },
        "Initialize the vAi engine now instead of on the first NeuralNet: eva "
        "prints its GPU list here and asks on stdin when there are several. "
        "Idempotent.");

    // The built-in models draw their top-k sampling uniform from std::rand()
    // (models/production/*/net.h), which nothing seeds; this is the only knob.
    m.def("seed",
        [](unsigned int value) { std::srand(value); },
        nb::arg("value"),
        "Seed the C RNG the built-in LLMs sample from, so a generation repeats "
        "for the same prompt and device. Process-wide; a hand-built graph feeds "
        "TopKSamplingNode.set_uniform() and is unaffected.");

    m.def("get_vai_device",
        []() { return vai::getVaiDevice(); },
        "The process-wide default Device, used by nets built without device=. "
        "Initializes the engine on first use.");
}

}   // namespace pyvai
