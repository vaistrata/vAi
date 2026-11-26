#ifndef VAI_NODES_UNFLATTEN_H
#define VAI_NODES_UNFLATTEN_H

#include "common.h"

namespace vai::node {

/*=============================================================================
  Unflatten - Split one axis into several, as torch.unflatten
  Input:  in0  [d0, .., d(r-1)]
  Output: out0 in0's shape with d(axis) replaced by sizes (shares buffer with input)

  A negative axis counts from the end. Each size is positive, or -1 for whatever
  d(axis) leaves over, at most once. The sizes must multiply to d(axis).
=============================================================================*/
class Unflatten : public Node
{
public:
    struct Config
    {
        int axis;                     // the axis to split; negative counts from the end
        std::vector<int64_t> sizes;   // what it splits into; one may be -1
    } config_;

    Unflatten(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override {}
};


inline Unflatten::Unflatten(Config config)
: config_(std::move(config))
{
    VAI_ASSERT(!config_.sizes.empty());
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}

inline void Unflatten::prepare()
{
    const Tensor& inTensor = (*this)["in0"];
    VAI_ASSERT(inTensor.validShape());

    const std::vector<uint32_t>& shape = inTensor.shape();
    int axis = config_.axis;
    if (axis < 0)
        axis += int(shape.size());
    VAI_ASSERT(axis >= 0 && axis < int(shape.size()));

    const uint32_t d = shape[axis];
    std::vector<uint32_t> sizes(config_.sizes.size());
    uint64_t product = 1;
    int inferred = -1;

    for (size_t i = 0; i < sizes.size(); ++i)
    {
        if (config_.sizes[i] == -1)
        {
            VAI_ASSERT(inferred < 0);   // only one size may be left over
            inferred = (int)i;
            continue;
        }
        VAI_ASSERT(config_.sizes[i] > 0);
        sizes[i] = (uint32_t)config_.sizes[i];
        product *= sizes[i];
    }

    if (inferred >= 0)
    {
        VAI_ASSERT(d % product == 0);
        sizes[inferred] = uint32_t(d / product);
    }
    else
        VAI_ASSERT(product == d);

    std::vector<uint32_t> dims(shape.begin(), shape.begin() + axis);
    dims.insert(dims.end(), sizes.begin(), sizes.end());
    dims.insert(dims.end(), shape.begin() + axis + 1, shape.end());
    (*this)["out0"] = inTensor.reshape(dims);
}

}   // namespace vai::node

#endif   // VAI_NODES_UNFLATTEN_H
