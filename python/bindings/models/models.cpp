#include "python/bindings/module.h"
#include "python/bindings/models/models.h"


namespace pyvai {

void bindModels(nb::module_& m)
{
    bindModelConfigs(m);   // the families' configs are built from these
    bindGPT2(m);
    bindLlama3(m);
    bindGemma3(m);
    bindPaliGemma(m);
    bindPi0(m);
    bindPi05(m);
    bindYOLOX(m);

    // Private: python/tests checks these nodes as it checks the core ones.
    nb::module_ nodes = m.def_submodule("model_nodes", "Nodes of models/<model>/nodes/, one submodule per model.");
    nb::module_ yoloxNodes = nodes.def_submodule("yolox", "models/yolox/nodes/");
    bindYOLOXNodes(yoloxNodes);
}

}   // namespace pyvai
