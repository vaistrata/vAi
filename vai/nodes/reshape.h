#ifndef VAI_NODES_RESHAPE_H
#define VAI_NODES_RESHAPE_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  Reshape - Reshape tensor to specified dimensions
  Input: in0 [any shape]
  Output: out0 [dims] (shares buffer with input)

  A dim is a size, 0 for the input's dimension at that same position, or -1 for
  whatever the element count leaves over, at most once (= ONNX). 0 carries a size only
  the caller's data knows without the graph naming it. The resolved dims must multiply
  to the input's element count. The output aliases the input buffer.
=============================================================================*/
class Reshape : public Node
{
    std::vector<int64_t> dims_;

public:
    Reshape(std::vector<int64_t> dims);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override {}
};


inline Reshape::Reshape(std::vector<int64_t> dims)
: dims_(std::move(dims))
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void Reshape::prepare()
{
    const Tensor& inTensor = (*this)["in0"];
    VAI_ASSERT(inTensor.validShape());

    const std::vector<uint32_t>& shape = inTensor.shape();
    std::vector<uint32_t> dims(dims_.size());
    uint32_t product = 1;
    int inferred = -1;

    for (size_t i = 0; i < dims_.size(); ++i)
    {
        if (dims_[i] == -1)
        {
            VAI_ASSERT(inferred < 0);   // only one dim may be left over
            inferred = (int) i;
            continue;
        }
        if (dims_[i] == 0)
        {
            VAI_ASSERT(i < shape.size());
            dims[i] = shape[i];
        }
        else
        {
            VAI_ASSERT(dims_[i] > 0);
            dims[i] = (uint32_t) dims_[i];
        }
        product *= dims[i];
    }

    if (inferred >= 0)
    {
        VAI_ASSERT(product > 0 && inTensor.numElements() % product == 0);
        dims[inferred] = inTensor.numElements() / product;
    }
    else
        VAI_ASSERT(product == inTensor.numElements());

    (*this)["out0"] = inTensor.reshape(dims);
}

}   // namespace vai::node

#endif // VAI_NODES_RESHAPE_H
