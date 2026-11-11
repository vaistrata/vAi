#include "kvcache-counter.h"
#include "vai/groups/transformer.h"   // createKvCacheCounter
#include <cstdint>
#include <optional>
#include <string>

namespace pyvai {

void bindKVCacheCounter(nb::module_& m)
{
    nb::class_<KVCacheCounter>(m, "KVCacheCounter")
        .def(
            "__init__",
            [](KVCacheCounter* self, std::optional<eva::Device> device) {
                eva::Device dev = device.value_or(eva::Device{});
                if (!dev)
                    dev = vai::getVaiDevice();
                new (self) KVCacheCounter(vai::createKvCacheCounter(dev));   // starts at 0
            },
            nb::arg("device") = nb::none(),
            "Create the counter on `device`, by default get_vai_device(). Pass "
            "net.device when the consuming net was built with its own device."
        )
        .def("update", [](KVCacheCounter& self, uint32_t count) {
            self.count() = count;
        }, nb::arg("count"),
            "Set the count of tokens the KV cache holds. Not synchronized: call "
            "net.sync() first if a forward may still be reading the counter.")
        .def("reset", [](KVCacheCounter& self) {
            self.count() = 0;
        }, "Zero the count. Same synchronization rule as update().")
        .def_prop_ro("value", [](KVCacheCounter& self) -> uint32_t {
            return self.count();
        })
        .def("__repr__", [](KVCacheCounter& self) -> std::string {
            return "KVCacheCounter(value=" + std::to_string(self.count()) + ")";
        });
}

}   // namespace pyvai
