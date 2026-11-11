#ifndef VAI_NODES_IDENTITY_H
#define VAI_NODES_IDENTITY_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  Identity - Pass-through node (no computation)
  Input: in0 [any shape]
  Output: out0 [same shape] (shares buffer with input)

  The output aliases the input buffer; no computation or copy occurs.
=============================================================================*/
class Identity : public Node
{
public:
    Identity() {
        addSlot("in0", NodeSlot::input);
        addSlot("out0", NodeSlot::output);
    }

    void prepare() override {
        (*this)["out0"] = (*this)["in0"];
    }

    void run(eva::CommandBuffer cmdBuff) override {}
};

}   // namespace vai::node

#endif // VAI_NODES_IDENTITY_H
