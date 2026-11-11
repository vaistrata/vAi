#include "module.h"


NB_MODULE(_pyvai, m)
{
    m.doc() = "vAi engine — Python bindings.";

    pyvai::bindDevice(m);
    pyvai::bindTensor(m);
    pyvai::bindGraphFlow(m);
    pyvai::bindNodes(m);
    pyvai::bindNeuralNet(m);
    pyvai::bindSafetensors(m);
    pyvai::bindTokenizer(m);
    pyvai::bindImageProcessor(m);
    pyvai::bindModels(m);
}
