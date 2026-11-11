#ifndef VAI_NODES_GATE_H
#define VAI_NODES_GATE_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  Gate - conditional branch (no computation)
  Input : in0  [any shape]
  Output: then [same shape], else [same shape] — both alias in0's buffer

  One branch is recorded per command buffer. The engine drops every node that
  only the closed branch reaches, so those nodes neither prepare() nor run();
  a node both branches reach (the merge point and everything past it) is kept.
  Leaving "else" unwired makes a closed gate drop the whole downstream.

  The condition is host state, so flipping it re-records. Weights are uploaded
  by uploadModelParams(), which ignores gates — a branch closed on the first forward
  still gets its weights on the device.

  Two producers may share one input slot only where a gate's branches merge;
  sortNodes() rejects any other fan-in. The branches need not agree on shape or
  dtype: a flip re-records, so every recording derives the merge point and its
  downstream from whichever branch survived it.
=============================================================================*/
class Gate : public Node
{
    bool open_ = true;

public:
    Gate()
    {
        addSlot("in0",  NodeSlot::input);
        addSlot("then", NodeSlot::output);
        addSlot("else", NodeSlot::output);

        flags_ |= gate;   // the engine reads the two branches through this
    }

    bool isOpen() const { return open_; }

    // Call once the previous forward has completed, like any other host-side
    // per-forward state.
    void setOpen(bool open)
    {
        if (open_ == open)
            return;
        open_ = open;
        ownedNet_->invalidateCommandBuffer();
    }

    void prepare() override
    {
        (*this)["then"] = (*this)["in0"];
        (*this)["else"] = (*this)["in0"];
    }

    void run(eva::CommandBuffer) override {}
};

}   // namespace vai::node

#endif // VAI_NODES_GATE_H
