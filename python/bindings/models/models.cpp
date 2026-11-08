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
    std::set<std::string> models;
    for (const NodeBinding& b : nodeBindings())
        if (!b.model.empty())
            models.insert(b.model);
    for (const std::string& model : models)
    {
        nb::module_ sub = nodes.def_submodule(model.c_str(), ("models/" + model + "/nodes/").c_str());
        bindRegisteredNodes(sub, model);
    }
}

}   // namespace pyvai
