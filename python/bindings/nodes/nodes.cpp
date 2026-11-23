#include "python/bindings/module.h"
#include "python/bindings/nodes/bind-node.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/common.h"
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
    // vai::ActType, ResidualOrder and KVCacheCounter are argument types of nodes below:
    // register them first.
    nb::enum_<vai::ActType>(m, "ActType")
        .value("NONE", vai::ActType::None)
        .value("RELU", vai::ActType::Relu)
        .value("GELU", vai::ActType::Gelu, "Exact, through erf.")
        .value("GELU_TANH", vai::ActType::GeluTanh, "The tanh approximation (gelu_pytorch_tanh).")
        .value("SILU", vai::ActType::Silu)
        .value("SIGMOID", vai::ActType::Sigmoid);

    nb::enum_<vai::ResidualOrder>(
        m, "ResidualOrder", "Where a node adds its `residual` input, relative to the activation.")
        .value("AFTER_ACTIVATION", vai::ResidualOrder::AfterActivation, "activation(x) + residual, a following add.")
        .value("BEFORE_ACTIVATION",
               vai::ResidualOrder::BeforeActivation,
               "activation(x + residual), an add ahead of the activation.");

    bindKVCacheCounter(m);

    // Every python/bindings/nodes/<node>.cpp that ends in PYVAI_NODE_BINDING.
    bindRegisteredNodes(m, "");
}

}   // namespace pyvai
