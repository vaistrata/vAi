/*=============================================================================
  Written by   phg9296@vaistrata.com
  Model        RT-DETRv2 (S / M / L / X)
=============================================================================*/

#ifndef EXPERIMENTAL_RTDETRV2_NET_H
#define EXPERIMENTAL_RTDETRV2_NET_H

#include "groups/decoder-layer.h"
#include "groups/decoder.h"
#include "groups/encoder-layer.h"
#include "groups/hybrid-encoder.h"
#include "groups/query-selection.h"
#include "groups/resnet-vd.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include <cstdint>
#include <iterator>

namespace model::experimental::rtdetrv2 {

/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - checkpoint dimensions shared by the backbone, encoder, and decoder
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;
    group::ResNetVdConfig backboneCfg;
    group::EncoderLayerConfig encoderLayerCfg{};
    group::DecoderLayerConfig decoderLayerCfg = {
        .ffnHiddenDim = 1024,
        .normEps = 1e-5f,
        .crossCfg = { .inDim = 256,
                      .outDim = 256,
                      .headDim = 32,
                      .numHeads = 8,
                      .featureShapes = { { 80, 80 }, { 40, 40 }, { 20, 20 } },
                      .numSamplingPoints = { 4, 4, 4 } }
    };
    uint32_t decoderLayers;
    uint32_t positionHiddenDim = 512;
    uint32_t numQueries = 300;
    uint32_t numClasses = 80;
    float expansion = 0.5f;

    static constexpr ModelConfig small()   // ResNet18 — PekingU/rtdetr_v2_r18vd
    {
        return {
            .variant = "small",
            .backboneCfg = { .stageDepths = { 2, 2, 2, 2 }, .bottleneck = false },
            .decoderLayers = 3,
        };
    }

    static constexpr ModelConfig medium()   // ResNet34 — PekingU/rtdetr_v2_r34vd
    {
        return {
            .variant = "medium",
            .backboneCfg = { .stageDepths = { 3, 4, 6, 3 }, .bottleneck = false },
            .decoderLayers = 4,
        };
    }

    static constexpr ModelConfig large()   // ResNet50 — PekingU/rtdetr_v2_r50vd
    {
        return {
            .variant = "large",
            .backboneCfg = { .stageDepths = { 3, 4, 6, 3 }, .bottleneck = true },
            .decoderLayers = 6,
            .expansion = 1.f,
        };
    }

    static constexpr ModelConfig xlarge()   // ResNet101 — PekingU/rtdetr_v2_r101vd
    {
        return {
            .variant = "xlarge",
            .backboneCfg = { .stageDepths = { 3, 4, 23, 3 }, .bottleneck = true },
            .encoderLayerCfg = { .embedDim = 384, .headDim = 48, .ffnHiddenDim = 2048 },
            .decoderLayers = 6,
            .expansion = 1.f,
        };
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// Net - RT-DETRv2 object detection
// Input:  in0 [3, 640, 640], contiguous CHW in oDefaultDtype (FP32 or FP16)
// Output: pred_logits [300, 80], pred_boxes [300, 4] (both FP32)
// Boxes use normalized center-x, center-y, width, height coordinates.
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    group::ResNetVd backbone;
    group::HybridEncoder encoder;
    vai::node::Convolution proj0, proj1, proj2;
    vai::node::Permute tokenPermute0, tokenPermute1, tokenPermute2;
    vai::node::Reshape tokenShape0, tokenShape1, tokenShape2;
    vai::node::Concat memory;
    group::QuerySelection querySelection;
    group::Decoder decoder;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg)
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , backbone(modelCfg.backboneCfg)
    , encoder(
          { .inChannels = { modelCfg.backboneCfg.outChannels(1),
                            modelCfg.backboneCfg.outChannels(2),
                            modelCfg.backboneCfg.outChannels(3) },
            .layerCfg = modelCfg.encoderLayerCfg,
            .expansion = modelCfg.expansion })
    , proj0({ .inChannels = modelCfg.encoderLayerCfg.embedDim,
              .outChannels = modelCfg.decoderLayerCfg.crossCfg.inDim,
              .kernelSize = 1,
              .stride = 1 })
    , proj1({ .inChannels = modelCfg.encoderLayerCfg.embedDim,
              .outChannels = modelCfg.decoderLayerCfg.crossCfg.inDim,
              .kernelSize = 1,
              .stride = 1 })
    , proj2({ .inChannels = modelCfg.encoderLayerCfg.embedDim,
              .outChannels = modelCfg.decoderLayerCfg.crossCfg.inDim,
              .kernelSize = 1,
              .stride = 1 })
    , tokenPermute0({ 1, 2, 0 })
    , tokenPermute1({ 1, 2, 0 })
    , tokenPermute2({ 1, 2, 0 })
    , tokenShape0({ -1, modelCfg.decoderLayerCfg.crossCfg.inDim })
    , tokenShape1({ -1, modelCfg.decoderLayerCfg.crossCfg.inDim })
    , tokenShape2({ -1, modelCfg.decoderLayerCfg.crossCfg.inDim })
    , memory({ .axis = 0, .inputCount = uint32_t(std::size(modelCfg.decoderLayerCfg.crossCfg.featureShapes)) })
    , querySelection(
          { .embedDim = modelCfg.decoderLayerCfg.crossCfg.inDim,
            .numQueries = modelCfg.numQueries,
            .numClasses = modelCfg.numClasses,
            .normEps = modelCfg.decoderLayerCfg.normEps,
            .featureShapes = { { modelCfg.decoderLayerCfg.crossCfg.featureShapes[0][0],
                                 modelCfg.decoderLayerCfg.crossCfg.featureShapes[0][1] },
                               { modelCfg.decoderLayerCfg.crossCfg.featureShapes[1][0],
                                 modelCfg.decoderLayerCfg.crossCfg.featureShapes[1][1] },
                               { modelCfg.decoderLayerCfg.crossCfg.featureShapes[2][0],
                                 modelCfg.decoderLayerCfg.crossCfg.featureShapes[2][1] } } })
    , decoder(
          { .numLayers = modelCfg.decoderLayers,
            .numClasses = modelCfg.numClasses,
            .positionHiddenDim = modelCfg.positionHiddenDim,
            .layerCfg = modelCfg.decoderLayerCfg })
    {
        input("in0") - backbone;
        // [D, H, W] -> [H, W, D] -> [H * W, D], then stack P3/P4/P5.
        backbone / "p3" - "p3" / encoder;
        encoder / "out_p3" - proj0 - tokenPermute0 - tokenShape0 - "in0" / memory;

        backbone / "p4" - "p4" / encoder;
        encoder / "out_p4" - proj1 - tokenPermute1 - tokenShape1 - "in1" / memory;

        backbone / "p5" - "p5" / encoder;
        encoder / "out_p5" - proj2 - tokenPermute2 - tokenShape2 - "in2" / memory;

        memory - "memory" / decoder;
        memory - "memory" / querySelection;
        querySelection / "queries" - "queries" / decoder;
        querySelection / "boxLogits" - "boxLogits" / decoder;
        decoder / "pred_logits" - output("pred_logits");
        decoder / "pred_boxes" - output("pred_boxes");

        const auto key = dtypeKey();
        querySelection.classHead.setODtype(vai::Dtype::Fp32, key);
        querySelection.bbox2.setODtype(vai::Dtype::Fp32, key);
        querySelection.anchorAdd.setODtype(vai::Dtype::Fp32, key);
        decoder.initBoxes.setODtype(vai::Dtype::Fp32, key);
        for (auto& block : decoder.blocks)
        {
            block.bbox2.setODtype(vai::Dtype::Fp32, key);
            block.layer.cross.offsets.setODtype(vai::Dtype::Fp32, key);
            block.layer.cross.logits.setODtype(vai::Dtype::Fp32, key);
            if (block.classHead)
                block.classHead->setODtype(vai::Dtype::Fp32, key);
        }
    }
};

}   // namespace model::experimental::rtdetrv2

#endif   // EXPERIMENTAL_RTDETRV2_NET_H
