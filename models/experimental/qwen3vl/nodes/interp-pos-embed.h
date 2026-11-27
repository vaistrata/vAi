#ifndef EXPERIMENTAL_QWEN3VL_NODES_INTERP_POS_EMBED_H
#define EXPERIMENTAL_QWEN3VL_NODES_INTERP_POS_EMBED_H

#include "vai/nodes/common.h"
#include <cmath>

namespace model::experimental::qwen3vl::node {

/////////////////////////////////////////////////////////////////////////////////////////
// InterpPosEmbed - add a learned position table resampled to the patch grid, in place
// Input: in0 [..., N, D], N = Gh * Gw patches in tileSize x tileSize tiles
// Output: out0 in0 with each patch's sample of table added
//
// table [S * S, D] is an S x S grid of learned rows. A patch at (row, col) of the
// Gh x Gw grid samples it bilinearly with corners aligned, so the image size may
// change from one forward to the next (HF's fast_pos_embed_interpolate).
//
// gridWidth is a host-visible uniform uint (Gw) the model owns and hands over with
// setGridWidth, the one Rope's Grid2D reads; Gh = N / Gw. out0 aliases in0, so no
// other node may read in0 after this one.
/////////////////////////////////////////////////////////////////////////////////////////
class InterpPosEmbed : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;
    eva::Buffer gridWidth;
    uint32_t side = 0;   // S

public:
    struct PipeParams
    {
        vai::Dtype oDtype = vai::Dtype::None;
        vai::Dtype wDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config
    {
        uint32_t tileSize = 1;   // patch order, as PatchEmbedding lays it; 1 = raster
    } config_;

    InterpPosEmbed();
    InterpPosEmbed(Config config);
    void setGridWidth(eva::Buffer buffer) { gridWidth = buffer; }
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline InterpPosEmbed::InterpPosEmbed()
: InterpPosEmbed(Config())
{
}

inline InterpPosEmbed::InterpPosEmbed(Config config)
: config_(config)
{
    VAI_ASSERT(config_.tileSize >= 1);
    addSlot("out0", vai::NodeSlot::output);
    addSlot("in0", vai::NodeSlot::input);
    addSlot("table", vai::NodeSlot::persistent);
}

inline void InterpPosEmbed::prepare()
{
    const vai::Tensor& inTen = (*this)["in0"];
    const vai::Tensor& table = (*this)["table"];
    VAI_ASSERT(inTen.validShape() && inTen.shape().size() >= 2);
    VAI_ASSERT(gridWidth && "InterpPosEmbed: setGridWidth() was never called");

    const auto& shape = inTen.shape();
    const uint32_t D = shape.back();
    const uint32_t N = shape[shape.size() - 2];
    VAI_ASSERT(table.shape().size() == 2 && table.shape()[1] == D);
    side = uint32_t(std::lround(std::sqrt(double(table.shape()[0]))));
    VAI_ASSERT(side * side == table.shape()[0]);

    const uint32_t T = config_.tileSize;
    const uint32_t gridW = *(uint32_t*)gridWidth.map();
    VAI_ASSERT(gridW > 0 && N % gridW == 0 && gridW % T == 0 && (N / gridW) % T == 0);

    (*this)["out0"] = inTen;

    const PipeParams p = { .oDtype = inTen.dtype(), .wDtype = table.dtype() };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        // clang-format off
        pipe = pipelinePool().request(EXPERIMENTAL_QWEN3VL_SHADER_OUTPUT_DIR, "interp-pos-embed", {
            eva::constant_id<0>(T),                        // tileSize
            eva::constant_id<SPEC_ID_TYPE_O>(p.oDtype),    // typeO
            eva::constant_id<SPEC_ID_TYPE_W>(p.wDtype)     // typeW
        });
        // clang-format on
        if (!descSet)
            descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
    }
}

inline void InterpPosEmbed::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto tableBuf = (*this)["table"].buffer();
    const auto& shape = (*this)["in0"].shape();

    struct
    {
        uint32_t D, N, rows, S;
    } constants = { shape.back(), shape[shape.size() - 2], uint32_t(vai::outerSize(shape)), side };

    descSet.write({ outBuf, tableBuf, gridWidth });
    cmdBuff.bindPipeline(pipe)
        .bindDescSets({ descSet })
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(constants.D, constants.rows)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace model::experimental::qwen3vl::node

#endif   // EXPERIMENTAL_QWEN3VL_NODES_INTERP_POS_EMBED_H
