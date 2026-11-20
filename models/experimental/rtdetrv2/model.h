/*=============================================================================
  Written by   phg9296@vaistrata.com
  Model        RT-DETRv2 (S / M / L / X)
=============================================================================*/

#ifndef EXPERIMENTAL_RTDETRV2_MODEL_H
#define EXPERIMENTAL_RTDETRV2_MODEL_H

#include "groups/resnet-vd.h"
#include "groups/decoder-layer.h"
#include "groups/decoder.h"
#include "groups/encoder-layer.h"
#include "groups/hybrid-encoder.h"
#include "groups/query-selection.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"

namespace model::experimental::rtdetrv2 {

/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - checkpoint dimensions shared by the backbone, encoder, and decoder
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* name;   // checkpoint folder under data/rtdetrv2/ckpts/
    group::ResNetvdConfig backboneCfg;
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

    static constexpr ModelConfig S()   // ResNet18 — PekingU/rtdetr_v2_r18vd
    {
        return {
            .name = "small",
            .backboneCfg = { .stageDepths = { 2, 2, 2, 2 }, .bottleneck = false },
            .decoderLayers = 3,
        };
    }

    static constexpr ModelConfig M()   // ResNet34 — PekingU/rtdetr_v2_r34vd
    {
        return {
            .name = "medium",
            .backboneCfg = { .stageDepths = { 3, 4, 6, 3 }, .bottleneck = false },
            .decoderLayers = 4,
        };
    }

    static constexpr ModelConfig L()   // ResNet50 — PekingU/rtdetr_v2_r50vd
    {
        return {
            .name = "large",
            .backboneCfg = { .stageDepths = { 3, 4, 6, 3 }, .bottleneck = true },
            .decoderLayers = 6,
            .expansion = 1.f,
        };
    }

    static constexpr ModelConfig X()   // ResNet101 — PekingU/rtdetr_v2_r101vd
    {
        return {
            .name = "xlarge",
            .backboneCfg = { .stageDepths = { 3, 4, 23, 3 }, .bottleneck = true },
            .encoderLayerCfg = { .embedDim = 384, .headDim = 48, .ffnHiddenDim = 2048 },
            .decoderLayers = 6,
            .expansion = 1.f,
        };
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
// Model - RT-DETRv2 object detection
// Input:  in0 [3, 640, 640], contiguous CHW in oDefaultDtype (FP32 or FP16)
// Output: pred_logits [300, 80], pred_boxes [300, 4] (both FP32)
// Boxes use normalized center-x, center-y, width, height coordinates.
/////////////////////////////////////////////////////////////////////////////////////////
struct Model : public vai::NeuralNet
{
    ModelConfig cfg;
    group::ResNetvd backbone;
    group::HybridEncoder encoder;
    vai::node::Convolution proj0, proj1, proj2;
    vai::node::Permute tokenPermute0, tokenPermute1, tokenPermute2;
    vai::node::Reshape tokenShape0, tokenShape1, tokenShape2;
    vai::node::Concat memory;
    group::QuerySelection querySelection;
    group::Decoder decoder;

    Model(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg)
    : NeuralNet(netCfg)
    , cfg(modelCfg)
    , backbone(cfg.backboneCfg)
    , encoder(
          { .inChannels = { cfg.backboneCfg.outChannels(1),
                            cfg.backboneCfg.outChannels(2),
                            cfg.backboneCfg.outChannels(3) },
            .layerCfg = cfg.encoderLayerCfg,
            .expansion = cfg.expansion })
    , proj0({ .inChannels = cfg.encoderLayerCfg.embedDim,
              .outChannels = cfg.decoderLayerCfg.crossCfg.inDim,
              .kernelSize = 1,
              .stride = 1 })
    , proj1({ .inChannels = cfg.encoderLayerCfg.embedDim,
              .outChannels = cfg.decoderLayerCfg.crossCfg.inDim,
              .kernelSize = 1,
              .stride = 1 })
    , proj2({ .inChannels = cfg.encoderLayerCfg.embedDim,
              .outChannels = cfg.decoderLayerCfg.crossCfg.inDim,
              .kernelSize = 1,
              .stride = 1 })
    , tokenPermute0({ 1, 2, 0 })
    , tokenPermute1({ 1, 2, 0 })
    , tokenPermute2({ 1, 2, 0 })
    , tokenShape0({ -1, cfg.decoderLayerCfg.crossCfg.inDim })
    , tokenShape1({ -1, cfg.decoderLayerCfg.crossCfg.inDim })
    , tokenShape2({ -1, cfg.decoderLayerCfg.crossCfg.inDim })
    , memory({ .axis = 0, .inputCount = uint32_t(std::size(cfg.decoderLayerCfg.crossCfg.featureShapes)) })
    , querySelection(
          { .embedDim = cfg.decoderLayerCfg.crossCfg.inDim,
            .numQueries = cfg.numQueries,
            .numClasses = cfg.numClasses,
            .normEps = cfg.decoderLayerCfg.normEps,
            .featureShapes = { { cfg.decoderLayerCfg.crossCfg.featureShapes[0][0],
                                 cfg.decoderLayerCfg.crossCfg.featureShapes[0][1] },
                               { cfg.decoderLayerCfg.crossCfg.featureShapes[1][0],
                                 cfg.decoderLayerCfg.crossCfg.featureShapes[1][1] },
                               { cfg.decoderLayerCfg.crossCfg.featureShapes[2][0],
                                 cfg.decoderLayerCfg.crossCfg.featureShapes[2][1] } } })
    , decoder(
          { .numLayers = cfg.decoderLayers,
            .numClasses = cfg.numClasses,
            .positionHiddenDim = cfg.positionHiddenDim,
            .layerCfg = cfg.decoderLayerCfg })
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

#endif   // EXPERIMENTAL_RTDETRV2_MODEL_H
