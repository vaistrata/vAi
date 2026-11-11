#ifndef VAI_NODES_FLATTEN_H
#define VAI_NODES_FLATTEN_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  Flatten - Flatten tensor to 1D
  Input: in0 [any shape]
  Output: out0 [N] where N = total number of elements (shares buffer with input)

  elements is the input's total element count. The output aliases the input buffer.
=============================================================================*/
class Flatten : public Node
{
public:
    Flatten();
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override {}
};


inline Flatten::Flatten()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

inline void Flatten::prepare()
{
    VAI_ASSERT((*this)["in0"].validShape());
    const Tensor& inTen = (*this)["in0"];
    (*this)["out0"] = inTen.reshape({static_cast<uint32_t>(inTen.numElements())});
}

}   // namespace vai::node

#endif // VAI_NODES_FLATTEN_H
