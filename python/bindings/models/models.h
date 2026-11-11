#ifndef PYVAI_MODELS_MODELS_H
#define PYVAI_MODELS_MODELS_H

#include "python/bindings/common.h"
#include "python/bindings/checks.h"           // ensureDirExists (load_weights)
#include "python/bindings/engine/forward.h"   // declareBoundaries, callForward
#include "vai/engine/safetensors-parser.h"
#include "vai/engine/weights.h"               // vai::Checkpoint
#include <string>

namespace pyvai {

// load_weights(dir) as every C++-built model binds it.
// `loadDoc` says which checkpoint the folder holds.
template <typename Model, typename Cls>
void bindLoadWeights(Cls& cls, const char* pyName, void (*load)(Model&, const vai::Checkpoint&), const char* loadDoc)
{
    cls.def(
        "load_weights",
        [pyName, load](Model& self, const std::string& dir) {
            ensureDirExists((std::string(pyName) + ".load_weights").c_str(), dir);
            nb::gil_scoped_release rel;   // parse + upload can take seconds
            load(self, vai::Checkpoint(dir));
        },
        nb::arg("dir"), loadDoc);
}

void bindModelConfigs(nb::module_&);
void bindMNIST(nb::module_&);
void bindGPT2(nb::module_&);
void bindLlama3(nb::module_&);
void bindGemma3(nb::module_&);
void bindPaliGemma(nb::module_&);
void bindPi0(nb::module_&);
void bindPi05(nb::module_&);
void bindYOLOX(nb::module_&);

// A model's own nodes (models/<impl>/<model>/nodes/), bound into
// pyvai._pyvai.model_nodes.<model> by bindModels(): one per model that has them.
void bindYOLOXNodes(nb::module_&);

}   // namespace pyvai

#endif // PYVAI_MODELS_MODELS_H
