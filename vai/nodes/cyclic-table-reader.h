#ifndef VAI_NODES_CYCLIC_TABLE_READER_H
#define VAI_NODES_CYCLIC_TABLE_READER_H

#include "node-common.h"

namespace vai::node {

/*=============================================================================
  CyclicTableReader - read successive table rows, wrapping at the end
  Input:  in0    [*]                   only scheduling: the node runs when it is fed
  Params: table  [numRows, ...]       rows read in cyclic order, at the loader's dtype
  Output: out0   table's shape without the leading axis, at oDtype

  Run k after rewind() copies row k % numRows, so a loop whose per-step
  values depend on the step alone (e.g. pi0.5's adaRMS modulation) reads them
  with no host write, the way TimeEmbedding walks t. rewind() starts over.
=============================================================================*/
class CyclicTableReader : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::Buffer workgroupCounterBuffer; // row-copy workgroups counted since rewind()

public:
    struct PipeParams {
        bool aligned = false;   // a row is whole vec4s

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t numRows;   // number of rows in the table
    } config_;

    CyclicTableReader(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    void rewind() { *(uint32_t*)workgroupCounterBuffer.map() = 0; }

private:
    PipeParams pipeParams;
};


inline CyclicTableReader::CyclicTableReader(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("table", NodeSlot::persistent);

    VAI_ASSERT(config_.numRows >= 1);


    // Device-local, so the atomics stay in VRAM; host-visible, so rewind() can write it.
    workgroupCounterBuffer = device().createBuffer({
        .size = sizeof(uint32_t),
        .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::DEVICE_LOCAL | eva::MEMORY_PROPERTY::HOST_VISIBLE |
                       eva::MEMORY_PROPERTY::HOST_COHERENT,
    });
    rewind();
}

inline void CyclicTableReader::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& table = (*this)["table"];
    VAI_ASSERT(table.shape().size() >= 2 && table.shape()[0] == config_.numRows);

    std::vector<uint32_t> rowShape(table.shape().begin() + 1, table.shape().end());
    const PipeParams p = { .aligned = outerSize(rowShape, 0) % 4 == 0 };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request("cyclic-table-reader", {
            eva::constant_id<SPEC_ID_TYPE_O>(oDtype),       // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(table.dtype()),        // typeI (table)
            eva::constant_id<SPEC_ID_ALIGNED>(p.aligned ? 1u : 0u)  // aligned (vec4)
        });
        // clang-format on
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(std::move(rowShape), oDtype);
}

inline void CyclicTableReader::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();

    descSet.write({ outBuf, (*this)["table"].buffer(), workgroupCounterBuffer });

    struct {
        uint32_t elemsPerRow, numRows;
    } constants = { uint32_t((*this)["out0"].numElements()), config_.numRows };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(pipeParams.aligned ? constants.elemsPerRow / 4 : constants.elemsPerRow)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_CYCLIC_TABLE_READER_H
