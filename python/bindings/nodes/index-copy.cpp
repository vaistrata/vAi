#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/index-copy.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace pyvai {

namespace {

// An index copy reads its index from a host-visible buffer its owner holds: a
// C++ model sizes one for its images, a node built from Python its own, held
// here and grown to fit.
struct PyIndexCopy : vai::node::IndexCopy
{
    eva::Buffer index;
    size_t capacity = 0;   // uint32 entries

    explicit PyIndexCopy(Config config) : vai::node::IndexCopy(config) {}
    ~PyIndexCopy() { if (index) index.destroy(); }

    void writeIndex(const std::vector<uint32_t>& entries)
    {
        // The GPU may still be reading the index. A node only holds its net const.
        auto* net = const_cast<vai::NeuralNet*>(ownedNet_);
        net->sync();
        if (entries.size() > capacity)
        {
            if (index)
                index.destroy();
            index = device().createBuffer({
                .size = entries.size() * sizeof(uint32_t),
                .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
                .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
            });
            capacity = entries.size();
            setIndex(index);
            net->invalidateCommandBuffer();   // the recording bound the old buffer
        }
        std::memcpy(index.map(), entries.data(), entries.size() * sizeof(uint32_t));
    }
};

}   // namespace


static void bindIndexCopy(nb::module_& m)
{
    using C = vai::node::IndexCopy::Config;
    bindNode<PyIndexCopy>(m, "IndexCopyNode",
                          "torch.index_copy along `axis`: out0 is in0 with slice index[n] taken from "
                          "slice n of in1, which is in0's shape with that axis replaced by len(index). "
                          "Both inputs share a dtype. out0 aliases in0, so no other node may read in0 "
                          "after this one; a forward without in1 passes in0 through. An index past "
                          "in0's axis leaves its slice unwritten.",
                          field("axis", &C::axis))
        .def(
            "set_index",
            [](PyIndexCopy& self, const std::vector<uint32_t>& index) {
                if (index.empty())
                    throw nb::value_error("IndexCopyNode.set_index: index must not be empty");
                self.writeIndex(index);
            },
            nb::arg("index"),
            "Where each slice of in1 lands along the axis; required before a forward "
            "that feeds in1. It waits for the GPU and is read at dispatch, so a rewrite "
            "no longer than any before it re-records nothing. A length other than in1's "
            "size along the axis is the next forward's error.");
}
PYVAI_NODE_BINDING(bindIndexCopy);

}   // namespace pyvai
