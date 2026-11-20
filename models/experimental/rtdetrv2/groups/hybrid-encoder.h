#ifndef EXPERIMENTAL_RTDETRV2_GROUPS_HYBRID_ENCODER_H
#define EXPERIMENTAL_RTDETRV2_GROUPS_HYBRID_ENCODER_H

#include "vai/nodes/concat.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/nearest-upsample.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include "csp-layer.h"
#include "encoder-layer.h"

namespace model::experimental::rtdetrv2::group {

/////////////////////////////////////////////////////////////////////////////////////////
// HybridEncoderConfig - input channels and encoder/fusion settings
/////////////////////////////////////////////////////////////////////////////////////////
struct HybridEncoderConfig
{
    uint32_t inChannels[3];
    EncoderLayerConfig layerCfg;
    float expansion = 0.5f;
};

static constexpr uint32_t p5Shape[2] = { 20, 20 };

/////////////////////////////////////////////////////////////////////////////////////////
// positionEmbedding - Fixed 2D sine/cosine positions for the configured encoder map
// Returns [H * W, dim] in oDefaultDtype, with sin(x), cos(x), sin(y), cos(y) blocks.
/////////////////////////////////////////////////////////////////////////////////////////
inline vai::Tensor positionEmbedding(uint32_t dim)
{
    const uint32_t height = p5Shape[0];
    const uint32_t width = p5Shape[1];

    const uint32_t numTokens = height * width;
    std::vector<float> values(size_t(numTokens) * dim);
    const uint32_t pd = dim / 4;
    // Match the checkpoint reference's meshgrid(width,height,indexing="ij")
    // order.
    for (uint32_t i = 0; i < numTokens; ++i)
    {
        const uint32_t xIndex = i / height;
        for (uint32_t j = 0; j < pd; ++j)
        {
            const float freq = 1.0f / std::pow(10000.0f, float(j) / pd);
            const float x = float(xIndex) * freq;
            const float y = (i % height) * freq;
            values[i * dim + j] = std::sin(x);
            values[i * dim + pd + j] = std::cos(x);
            values[i * dim + 2 * pd + j] = std::sin(y);
            values[i * dim + 3 * pd + j] = std::cos(y);
        }
    }
    return vai::Tensor({ numTokens, dim }, vai::NeuralNet::current()->config().oDefaultDtype)
        .setFloats(std::move(values));
}

/////////////////////////////////////////////////////////////////////////////////////////
// HybridEncoder - Coarse-scale self attention and top-panConv/bottom-upsample feature fusion
// Input:  p3 [C3, 4H, 4W], p4 [C4, 2H, 2W], p5 [C5, H, W]
// Output: out_p3 [D, 4H, 4W], out_p4 [D, 2H, 2W], out_p5 [D, H, W]
// H = W = 20 for 640x640 inputs. D is layerCfg.embedDim.
// The encoder receives [1, H * W, D] tokens; its output is restored to [D, H, W].
/////////////////////////////////////////////////////////////////////////////////////////
struct HybridEncoder : public vai::NodeGroup
{
    vai::node::Convolution proj0, proj1, proj2;
    vai::node::Permute toHWC, toCHW;
    vai::node::Reshape mergeHW, splitHW;
    EncoderLayer encoder;
    vai::node::Convolution fpnConv0, fpnConv1;   // 1x1 stride-1
    vai::node::Convolution panConv0, panConv1;   // 3x3 stride-2
    CSPLayer fpn0, fpn1;
    CSPLayer pan0, pan1;
    vai::node::NearestUpsample upsample0, upsample1;
    vai::node::Concat fpnConcat0, fpnConcat1;
    vai::node::Concat panConcat0, panConcat1;

    HybridEncoder(HybridEncoderConfig config)
    : proj0(
          { // p3(stage1)
            .inChannels = config.inChannels[0],
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 1,
            .stride = 1 })
    , proj1(
          { // p4(stage2)
            .inChannels = config.inChannels[1],
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 1,
            .stride = 1 })
    , proj2(
          { // p5(stage3)
            .inChannels = config.inChannels[2],
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 1,
            .stride = 1 })
    , toHWC({ 1, 2, 0 })
    , mergeHW({ 1, -1, config.layerCfg.embedDim })
    , encoder(config.layerCfg)
    , splitHW({ p5Shape[0], p5Shape[1], config.layerCfg.embedDim })
    , toCHW({ 2, 0, 1 })
    , fpnConv0(
          { .inChannels = config.layerCfg.embedDim,
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 1,
            .stride = 1,
            .epilogue = vai::Epilogue::Silu })
    , fpnConv1(
          { .inChannels = config.layerCfg.embedDim,
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 1,
            .stride = 1,
            .epilogue = vai::Epilogue::Silu })
    , panConv0(
          { .inChannels = config.layerCfg.embedDim,
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 3,
            .stride = 2,
            .epilogue = vai::Epilogue::Silu })
    , panConv1(
          { .inChannels = config.layerCfg.embedDim,
            .outChannels = config.layerCfg.embedDim,
            .kernelSize = 3,
            .stride = 2,
            .epilogue = vai::Epilogue::Silu })
    , fpn0({ config.layerCfg.embedDim, config.expansion })
    , fpn1({ config.layerCfg.embedDim, config.expansion })
    , pan0({ config.layerCfg.embedDim, config.expansion })
    , pan1({ config.layerCfg.embedDim, config.expansion })
    , fpnConcat0({ .axis = 0, .inputCount = 2 })
    , fpnConcat1({ .axis = 0, .inputCount = 2 })
    , panConcat0({ .axis = 0, .inputCount = 2 })
    , panConcat1({ .axis = 0, .inputCount = 2 })
    {
        defineSlot("p3", proj0.slot("in0"));
        defineSlot("p4", proj1.slot("in0"));
        defineSlot("p5", proj2.slot("in0"));

        proj0 - "in1" / fpnConcat1;
        proj1 - "in1" / fpnConcat0;
        proj2 - toHWC - mergeHW - encoder - splitHW - toCHW - fpnConv0;

        fpnConv0 - "in1" / panConcat1;
        fpnConv0 - upsample0 - "in0" / fpnConcat0 - fpn0 - fpnConv1;

        fpnConv1 - "in1" / panConcat0;
        fpnConv1 - upsample1 - "in0" / fpnConcat1 - fpn1;

        fpn1 - panConv0 - "in0" / panConcat0 - pan0;

        pan0 - panConv1 - "in0" / panConcat1 - pan1;

        encoder.slot("position").getValueRef() =
            positionEmbedding(config.layerCfg.embedDim)
                .reshape({ 1, p5Shape[0] * p5Shape[1], config.layerCfg.embedDim });

        defineSlot("out_p3", fpn1.slot("out0"));
        defineSlot("out_p4", pan0.slot("out0"));
        defineSlot("out_p5", pan1.slot("out0"));
    }
};

}   // namespace model::experimental::rtdetrv2::group

#endif   // EXPERIMENTAL_RTDETRV2_GROUPS_HYBRID_ENCODER_H
