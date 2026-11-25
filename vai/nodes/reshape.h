#ifndef VAI_NODES_RESHAPE_H
#define VAI_NODES_RESHAPE_H

#include "common.h"

namespace vai::node {

/*=============================================================================
  Reshape - Reshape tensor to specified dimensions
  Input: in0 [any shape]
  Output: out0 [leading.., dims] (shares buffer with input)

  The last trailingAxes axes of in0 become dims and the axes ahead of them pass
  through, so a leading batch axis may be there or not (= torch.unflatten /
  flatten). trailingAxes 0 means every axis. A dim is a size, or -1 for whatever
  the replaced axes leave over, at most once. The resolved dims must multiply to
  the replaced axes' element count. The output aliases the input buffer.
=============================================================================*/
class Reshape : public Node
{
    std::vector<int64_t> dims_;
    uint32_t trailingAxes_;

public:
    Reshape(std::vector<int64_t> dims, uint32_t trailingAxes = 0);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override {}
};


inline Reshape::Reshape(std::vector<int64_t> dims, uint32_t trailingAxes)
: dims_(std::move(dims))
, trailingAxes_(trailingAxes)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void Reshape::prepare()
{
    const Tensor& inTensor = (*this)["in0"];
    VAI_ASSERT(inTensor.validShape());

    const std::vector<uint32_t>& shape = inTensor.shape();
    const size_t replaced = trailingAxes_ ? trailingAxes_ : shape.size();
    VAI_ASSERT(replaced <= shape.size());
    const size_t lead = shape.size() - replaced;

    uint64_t count = 1;   // elements of the replaced axes
    for (size_t i = lead; i < shape.size(); ++i)
        count *= shape[i];

    std::vector<uint32_t> dims(shape.begin(), shape.begin() + lead);
    dims.resize(lead + dims_.size());
    uint64_t product = 1;
    int inferred = -1;

    for (size_t i = 0; i < dims_.size(); ++i)
    {
        if (dims_[i] == -1)
        {
            VAI_ASSERT(inferred < 0);   // only one dim may be left over
            inferred = (int) i;
            continue;
        }
        VAI_ASSERT(dims_[i] > 0);
        dims[lead + i] = (uint32_t) dims_[i];
        product *= dims[lead + i];
    }

    if (inferred >= 0)
    {
        VAI_ASSERT(product > 0 && count % product == 0);
        dims[lead + inferred] = uint32_t(count / product);
    }
    else
        VAI_ASSERT(product == count);

    (*this)["out0"] = inTensor.reshape(dims);
}

}   // namespace vai::node

#endif // VAI_NODES_RESHAPE_H
