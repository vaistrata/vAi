#include "python/bindings/module.h"
#include "python/bindings/checks.h"
#include "vai/utils/tokenizer.h"
#include <string>
#include <vector>


namespace pyvai {

using vai::utils::Tokenizer;

void bindTokenizer(nb::module_& m)
{
    nb::class_<Tokenizer>(m, "Tokenizer")
        .def(
            "__init__",
            [](Tokenizer* self, const std::string& tokenizer_json) {
                ensureFileExists("Tokenizer", tokenizer_json);
                new (self) Tokenizer(tokenizer_json);
            },
            nb::arg("tokenizer_json"),
            "Load a HuggingFace tokenizer.json."
        )

        .def("encode", &Tokenizer::encode,
             nb::arg("text"),
             nb::arg("add_special")   = false,
             nb::arg("parse_special") = true,
             nb::call_guard<nb::gil_scoped_release>())
        .def("decode",
             [](const Tokenizer& self, const std::vector<int>& ids, bool skip_special) -> nb::str {
                 std::string s;
                 {
                     nb::gil_scoped_release rel;
                     s = self.decode(ids, skip_special);
                 }
                 return nb::steal<nb::str>(
                     PyUnicode_DecodeUTF8(s.data(), (Py_ssize_t) s.size(), "replace"));
             },
             nb::arg("ids"),
             nb::arg("skip_special") = true,
             "The text of `ids`. A sequence cut mid-character comes out as U+FFFD; "
             "decode_bytes() gives the exact bytes.")
        .def("decode_bytes",
             [](const Tokenizer& self, const std::vector<int>& ids,
                bool skip_special) -> nb::bytes {
                 std::string s;
                 {
                     nb::gil_scoped_release rel;
                     s = self.decode(ids, skip_special);
                 }
                 return nb::bytes(s.data(), s.size());
             },
             nb::arg("ids"),
             nb::arg("skip_special") = true,
             "The raw UTF-8 bytes of `ids`. decode_bytes(a + b) == decode_bytes(a) + "
             "decode_bytes(b), so a stream decodes one token at a time.")

        .def("vocab_size", &Tokenizer::vocabSize)

        // End-of-turn ids are the checkpoint's, not the tokenizer's: this resolves them.
        .def("piece_to_id", &Tokenizer::pieceToId, nb::arg("piece"),
             "Returns -1 for an unknown piece.")
        .def("id_to_piece", &Tokenizer::idToPiece, nb::arg("id"),
             "Returns '' for an unknown id.")
        .def("is_special",  &Tokenizer::isSpecial, nb::arg("id"))

        .def("__repr__", [](const Tokenizer& self) -> std::string {
            return "Tokenizer(vocab_size=" + std::to_string(self.vocabSize()) + ")";
        });
}

}   // namespace pyvai
