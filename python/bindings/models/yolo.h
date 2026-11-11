#ifndef PYVAI_MODELS_YOLO_H
#define PYVAI_MODELS_YOLO_H

// What a YOLO family (YOLOX) binds on its tier enum: a tier's size letter, the
// one-letter suffix its checkpoints are named by on command lines.
#include "models.h"
#include <cstring>
#include <string>

namespace pyvai {

// from_letter() and letter on a tier enum. `letters` holds one letter per
// enumerator, in enumerator order.
template <typename Tier>
void bindTierLetters(nb::enum_<Tier>& tier, const char* pyName, const char* letters,
                     const char* fromLetterDoc, const char* letterDoc)
{
    tier.def_static(
            "from_letter",
            [pyName, letters](const std::string& letter) -> Tier {
                if (letter.size() == 1)
                    if (const char* p = std::strchr(letters, letter[0]); p && *p)
                        return static_cast<Tier>(p - letters);
                std::string known;
                for (const char* c = letters; *c; ++c)
                    known += (known.empty() ? "" : ", ") + std::string(1, *c);
                throw nb::value_error(
                    (std::string(pyName) + ".from_letter: unknown tier '" + letter
                     + "' (one of " + known + ")").c_str());
            },
            nb::arg("letter"),
            fromLetterDoc)
        .def_prop_ro(
            "letter",
            [letters](Tier tier) -> std::string {
                return std::string(1, letters[static_cast<int>(tier)]);
            },
            letterDoc);
}

}   // namespace pyvai

#endif // PYVAI_MODELS_YOLO_H
