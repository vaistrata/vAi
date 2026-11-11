/*=============================================================================
  Written by   oneway@vaistrata.com
               yhc407-vaistrata.com
  Model        ACT (Action Chunking with Transformers, ResNet-18 and a DETR encoder-decoder)
=============================================================================*/

#ifndef PRODUCTION_ACT_NET_H
#define PRODUCTION_ACT_NET_H

#include "groups/decoder-block.h"
#include "groups/encoder-block.h"
#include "vai/groups/resnet.h"
#include "vai/nodes/concat.h"
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>


namespace model::production::act {


// What a checkpoint's config.json fixes, for ModelConfig::get().
struct CheckpointConfig
{
    const char* variant;                       // checkpoint folder under data/ckpts/act/, e.g. "lerobot/transfer-cube"
    std::vector<std::string> cameras;          // <name> of observation.images.<name>, in config.json's
                                               // order; {} is the one camera of observation.image
    std::pair<uint32_t, uint32_t> imageSize;   // {height, width}, every camera's
    uint32_t stateDim = 0;                     // observation.state
    uint32_t actionDim = 0;
    uint32_t actionChunk = 0;   // actions one forward predicts
};


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - ACT as LeRobot trains it: one ResNet-18 over every camera, a 4-block
// encoder and a 1-block decoder. get() takes the checkpoint's CheckpointConfig.
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // as CheckpointConfig
    std::vector<std::string> cameras;
    std::pair<uint32_t, uint32_t> imageSize;
    uint32_t stateDim = 0;
    uint32_t actionDim = 0;
    uint32_t actionChunk = 0;

    uint32_t embedDim = 512;
    uint32_t numHeads = 8;
    uint32_t ffnHiddenDim = 3200;
    uint32_t numEncoderBlocks = 4;
    float normEps = 1e-5f;

    uint32_t numCameras() const { return cameras.empty() ? 1u : uint32_t(cameras.size()); }

    // ResNet-18 strides the image by 32, rounding up as each stride-2 layer does.
    uint32_t imageGridHeight() const { return CEIL_DIV(imageSize.first, 32u); }
    uint32_t imageGridWidth() const { return CEIL_DIV(imageSize.second, 32u); }
    uint32_t imageTokens() const { return numCameras() * imageGridHeight() * imageGridWidth(); }

    // The CVAE latent's token, the state's, then every camera's.
    uint32_t encoderTokens() const { return 2 + imageTokens(); }

    static ModelConfig get(CheckpointConfig ckpt)
    {
        return {
            .variant = ckpt.variant,
            .cameras = std::move(ckpt.cameras),
            .imageSize = ckpt.imageSize,
            .stateDim = ckpt.stateDim,
            .actionDim = ckpt.actionDim,
            .actionChunk = ckpt.actionChunk,
        };
    }
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;

    vai::group::ResNet18 imageEncoder;   // [B, 3, H, W] -> [B, C, h, w], B cameras as a batch
    vai::node::Reshape flatten;          // [B, C, h*w]
    vai::node::Permute transpose;        // [B, h*w, C]
    vai::node::Linear imageProj;         // [B, h*w, D]
    vai::node::Reshape mergeCameras;     // [B*h*w, D], camera by camera

    vai::node::Linear latentStateProj;     // [1, stateDim] -> [1, 2*D]
    vai::node::Reshape splitLatentState;   // [2, D]
    vai::node::Concat catTokens;           // [2 + B*h*w, D]
    std::vector<group::EncoderBlock> encoder;

    group::DecoderBlock decoder;   // [actionChunk, D]
    vai::node::LayerNorm decoderNorm;
    vai::node::Linear actionHead;   // [actionChunk, actionDim]

    // Filled by the loader: the normalization statistics and the decoder's projected queries.
    std::map<std::string, std::array<float, 3>> imageMean;   // per camera name; "image" for observation.image
    std::map<std::string, std::array<float, 3>> imageStd;
    std::vector<float> stateMean;   // [stateDim]
    std::vector<float> stateStd;
    std::vector<float> actionMean;   // [actionDim]
    std::vector<float> actionStd;
    vai::Tensor decoderQuery;   // [H, actionChunk, Dh]

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg)
        : NeuralNet(netCfg), modelCfg(modelCfg), flatten({0, vai::group::ResNet18::outChannels, -1}),
          transpose({0, 2, 1}), imageProj({.inDim = vai::group::ResNet18::outChannels, .outDim = modelCfg.embedDim}),
          mergeCameras({-1, int64_t(modelCfg.embedDim)}),
          latentStateProj({.inDim = modelCfg.stateDim, .outDim = 2 * modelCfg.embedDim}), splitLatentState({2, -1}),
          decoder({.embedDim = modelCfg.embedDim,
                   .numHeads = modelCfg.numHeads,
                   .ffnHiddenDim = modelCfg.ffnHiddenDim,
                   .normEps = modelCfg.normEps}),
          decoderNorm({.eps = modelCfg.normEps}), actionHead({.inDim = modelCfg.embedDim, .outDim = modelCfg.actionDim})
    {
        encoder.reserve(modelCfg.numEncoderBlocks);
        for (uint32_t i = 0; i < modelCfg.numEncoderBlocks; ++i)
            encoder.emplace_back(group::EncoderBlock::Config{.embedDim = modelCfg.embedDim,
                                                             .numHeads = modelCfg.numHeads,
                                                             .ffnHiddenDim = modelCfg.ffnHiddenDim,
                                                             .normEps = modelCfg.normEps});

        input("inState") - latentStateProj - splitLatentState - "in0" / catTokens;
        input("inImages") - imageEncoder - flatten - transpose - imageProj - mergeCameras - "in1" / catTokens;

        catTokens - encoder[0];
        for (uint32_t i = 1; i < modelCfg.numEncoderBlocks; ++i)
            encoder[i - 1] - encoder[i];

        encoder.back() - "memory" / decoder;
        input("inDecoderQuery") - "query" / decoder;
        decoder - decoderNorm - actionHead - output("outActions");

        actionHead.setODtype(vai::Dtype::Fp32, dtypeKey());
    }

    // Input: images [B, 3, H, W], one per camera in any order, each normalized with its
    // camera's imageMean / imageStd, and state [1, stateDim] with stateMean / stateStd.
    // Output: actions [actionChunk, actionDim], normalized; denormalizeActions gives joint units.
    void predict(const vai::Tensor& images, const vai::Tensor& state, vai::Tensor* actions)
    {
        (*this)({{"inImages", images}, {"inState", state}, {"inDecoderQuery", decoderQuery}},
                {{"outActions", actions}});
    }

    // state [stateDim], normalized in place: (s - mean[i]) / std[i].
    void normalizeState(std::vector<float>& state) const
    {
        for (size_t i = 0; i < state.size(); ++i)
            state[i] = (state[i] - stateMean[i]) / stateStd[i];
    }

    // actions [actionChunk, actionDim], un-normalized in place: a * std[d] + mean[d].
    void denormalizeActions(std::vector<float>& actions) const
    {
        const size_t dims = actionMean.size();
        for (size_t i = 0; i < actions.size(); ++i)
            actions[i] = actions[i] * actionStd[i % dims] + actionMean[i % dims];
    }
};


}   // namespace model::production::act

#endif   // PRODUCTION_ACT_NET_H
