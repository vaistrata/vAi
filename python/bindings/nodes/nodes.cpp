#include "python/bindings/module.h"
#include "nodes.h"
#include "kvcache-counter.h"
#include "vai/nodes/node-common.h"

namespace pyvai {

// Call order must match the list in nodes.h.
void bindNodes(nb::module_& m)
{
    // Epilogue, ResidualOrder and KVCacheCounter are argument types of nodes below:
    // register them first.
    nb::enum_<vai::Epilogue>(m, "Epilogue")
        .value("NONE",      vai::Epilogue::None)
        .value("RELU",      vai::Epilogue::Relu)
        .value("GELU",      vai::Epilogue::Gelu,     "Exact, through erf.")
        .value("GELU_TANH", vai::Epilogue::GeluTanh, "The tanh approximation (gelu_pytorch_tanh).")
        .value("SILU",      vai::Epilogue::Silu)
        .value("SIGMOID",   vai::Epilogue::Sigmoid);

    nb::enum_<vai::ResidualOrder>(m, "ResidualOrder",
                                  "Where LinearNode and ConvolutionNode add their `residual` input, relative to "
                                  "the epilogue.")
        .value("AFTER_EPILOGUE", vai::ResidualOrder::AfterEpilogue, "epilogue(x + bias) + residual, a following add.")
        .value("BEFORE_EPILOGUE", vai::ResidualOrder::BeforeEpilogue,
               "epilogue(x + bias + residual), an add ahead of the activation.");

    bindKVCacheCounter(m);

    bindIdentity(m);
    bindGate(m);
    bindReshape(m);
    bindFlatten(m);
    bindConcat(m);
    bindPermute(m);
    bindSlice(m);
    bindRowSliceScatter(m);
    bindIndexCopy(m);
    bindTypeCast(m);

    bindAdd(m);
    bindMul(m);
    bindGatedMul(m);
    bindEleAffine(m);
    bindActivation(m);

    bindConvolution(m);
    bindPool2d(m);
    bindNearestUpsample(m);

    bindLinear(m);
    bindLayerNorm(m);
    bindGroupNorm(m);
    bindRmsNorm(m);

    bindTokenEmbedding(m);
    bindTimeEmbedding(m);
    bindCyclicTableReader(m);
    bindRope(m);
    bindAttention(m);

    bindArgmax(m);
    bindTopKSampling(m);
}

}   // namespace pyvai
