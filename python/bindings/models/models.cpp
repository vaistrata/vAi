#include "python/bindings/module.h"
#include "python/bindings/models/models.h"
#include "python/bindings/nodes/bind-node.h"
#include <set>
#include <string>


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

    // A model's own nodes (models/<model>/nodes/), one submodule per model that
    // registers any (PYVAI_MODEL_NODE_BINDING). Private: python/tests checks
    // them as it checks the core ones.
    nb::module_ nodes = m.def_submodule("model_nodes", "Nodes of models/<model>/nodes/, one submodule per model.");
    std::set<std::string> modelNames;
    for (const NodeBinding& b : nodeBindings())
        if (!b.modelName.empty())
            modelNames.insert(b.modelName);
    for (const std::string& modelName : modelNames)
    {
        nb::module_ sub = nodes.def_submodule(modelName.c_str(), ("models/" + modelName + "/nodes/").c_str());
        bindRegisteredNodes(sub, modelName);
    }
}

}   // namespace pyvai
