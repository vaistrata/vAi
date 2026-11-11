#include "python/bindings/module.h"
#include "python/bindings/nodes/bind-node.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/node-common.h"
#include <algorithm>

namespace pyvai {

std::vector<NodeBinding>& nodeBindings()
{
    static std::vector<NodeBinding> bindings;   // filled before the module init runs
    return bindings;
}

void bindRegisteredNodes(nb::module_& m, const std::string& modelName)
{
    std::vector<NodeBinding> picked;
    for (const NodeBinding& b : nodeBindings())
        if (b.modelName == modelName)
            picked.push_back(b);
    std::sort(picked.begin(), picked.end(),
              [](const auto& a, const auto& b) { return a.bindFunctionName < b.bindFunctionName; });
    for (const NodeBinding& b : picked)
        b.bindFunction(m);
}

void bindNodes(nb::module_& m)
{
    // Epilogue and KVCacheCounter are argument types of nodes below: register them first.
    nb::enum_<vai::Epilogue>(m, "Epilogue")
        .value("NONE",      vai::Epilogue::None)
        .value("RELU",      vai::Epilogue::Relu)
        .value("GELU",      vai::Epilogue::Gelu,     "Exact, through erf.")
        .value("GELU_TANH", vai::Epilogue::GeluTanh, "The tanh approximation (gelu_pytorch_tanh).")
        .value("SILU",      vai::Epilogue::Silu)
        .value("SIGMOID",   vai::Epilogue::Sigmoid);

    bindKVCacheCounter(m);

    // Every python/bindings/nodes/<node>.cpp that ends in PYVAI_NODE_BINDING.
    bindRegisteredNodes(m, "");
}

}   // namespace pyvai
