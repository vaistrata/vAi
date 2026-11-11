#ifndef PYVAI_MODULE_H
#define PYVAI_MODULE_H

// Module assembly: one bind* per area, called by module.cpp in this order (a
// type must be registered before a signature refers to it).
#include "common.h"

namespace pyvai {

void bindDevice(nanobind::module_&);
void bindTensor(nanobind::module_&);
void bindGraphFlow(nanobind::module_&);
void bindNodes(nanobind::module_&);
void bindNeuralNet(nanobind::module_&);
void bindSafetensors(nanobind::module_&);
void bindTokenizer(nanobind::module_&);
void bindImageProcessor(nanobind::module_&);
void bindModels(nanobind::module_&);

}   // namespace pyvai

#endif // PYVAI_MODULE_H
