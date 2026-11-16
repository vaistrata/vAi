#include "python/bindings/module.h"
#include "python/bindings/checks.h"
#include "python/bindings/convert.h"
#include "vai/engine/safetensors-parser.h"
#include "vai/engine/tensor.h"
#include <optional>
#include <stdexcept>


namespace pyvai {

namespace {

void ensureKeyExists(const vai::SafetensorRef& ref, const std::string& key)
{
    try
    {
        (void) ref.getDataType();
    }
    catch (const std::out_of_range&)
    {
        throw nb::key_error(key.c_str());   // KeyError: '<key>', as a dict would
    }
}


// dtypeFromString asserts on anything else (Release mislabels the bytes as fp32).
void ensureLoadableDtype(const vai::SafetensorRef& ref, const std::string& key)
{
    const std::string dt = ref.getDataType();
    // clang-format off
    if (dt != "F32" && dt != "F16" && dt != "BF16")
        throw nb::type_error(
            ("SafetensorsParser: '" + key + "' has dtype " + dt
             + "; pyvai loads F32 / F16 / BF16 only (ref('" + key
             + "') still inspects it without loading)").c_str());
    // clang-format on
}

}   // namespace


void bindSafetensors(nb::module_& m)
{
    // Metadata-only handle; holds a raw parser pointer, so ref() uses keep_alive.
    nb::class_<vai::SafetensorRef>(m, "SafetensorRef")
        .def_prop_ro(
            "key",
            [](const vai::SafetensorRef& self) { return self.key; }
        )
        .def_prop_ro(
            "shape",
            [](const vai::SafetensorRef& self) { return shapeTuple(self.getShape()); },
            "The shape in the file header; no data is read."
        )
        .def_prop_ro(
            "dtype",
            [](const vai::SafetensorRef& self) { return self.getDataType(); },
            "The file's dtype tag ('F32', 'F16', 'BF16', ...): the on-disk format, "
            "not a Dtype."
        )
        .def(
            "__repr__",
            [](const vai::SafetensorRef& self) -> std::string {
                std::string s = "SafetensorRef('" + self.key + "', " +
                                self.getDataType() + ", [";
                auto shape = self.getShape();
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (i)
                        s += ", ";
                    s += std::to_string(shape[i]);
                }
                return s + "])";
            }
        );

    nb::class_<vai::SafetensorsParser>(m, "SafetensorsParser")
        .def(
            "__init__",
            [](vai::SafetensorsParser* self, const std::string& path) {
                ensureFileExists("SafetensorsParser", path);
                new (self) vai::SafetensorsParser(path.c_str());
            },
            nb::arg("path")
        )

        .def("tensor_names", &vai::SafetensorsParser::getTensorNames)

        .def(
            "__getitem__",
            [](const vai::SafetensorsParser& self, const std::string& key) -> vai::Tensor {
                auto ref = self[key];
                ensureKeyExists(ref, key);
                ensureLoadableDtype(ref, key);
                nb::gil_scoped_release rel;   // disk read + convert
                return vai::Tensor(ref);
            },
            nb::arg("key"),
            "Load `key` as a Tensor in the file's dtype."
        )

        .def(
            "get",
            [](const vai::SafetensorsParser& self,
               const std::string& key,
               std::optional<vai::Dtype> dtype) -> vai::Tensor {
                auto ref = self[key];
                ensureKeyExists(ref, key);
                // The source dtype must be loadable even when converting.
                ensureLoadableDtype(ref, key);
                nb::gil_scoped_release rel;
                return vai::Tensor(ref, dtype.value_or(vai::Dtype::None));
            },
            nb::arg("key"),
            nb::arg("dtype") = nb::none(),
            "Load `key` as a Tensor, converted to `dtype` when one is given."
        )

        .def(
            "ref",
            [](const vai::SafetensorsParser& self, const std::string& key) -> vai::SafetensorRef {
                auto ref = self[key];
                ensureKeyExists(ref, key);
                return ref;
            },
            nb::arg("key"),
            nb::keep_alive<0, 1>(),   // the ref keeps this parser alive
            "The metadata of `key` (.shape, .dtype) without loading its data. "
            "KeyError for an unknown key."
        )

        .def(
            "__repr__",
            [](const vai::SafetensorsParser& self) -> std::string {
                auto names = self.getTensorNames();
                return "SafetensorsParser(num_tensors=" +
                       std::to_string(names.size()) + ")";
            }
        );
}

}   // namespace pyvai
