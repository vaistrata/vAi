/*=============================================================================
  Written by   phg9296@vaistrata.com
  Checkpoint   Each tier reads data/yolox/ckpts/<tierName>/model.safetensors.
               Megvii ships PyTorch checkpoints in the official YOLOX release:
                 download   github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
                              -> data/yolox/ckpts/<tierName>/model.pth
                 convert    megvii-weights-convertor.py
                              -> data/yolox/ckpts/<tierName>/model.safetensors
=============================================================================*/

#ifndef YOLOX_MEGVII_WEIGHTS_H
#define YOLOX_MEGVII_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <string>
#include <cmath>


namespace yolox::megvii {

/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - map the official YOLOX state dict to the vAi graph
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    // Match the checkpoint's BN epsilon and the previous weight-dtype rounding.
    auto loadBatchNorm = [](auto& conv, const vai::Tensor& gamma,
                            const vai::Tensor& beta, const vai::Tensor& mean,
                            const vai::Tensor& variance)
    {
        constexpr float eps = 1e-3f;
        const uint32_t channels = conv.config_.outChannels;
        const vai::Dtype wt = conv.config_.wDtype;
        std::vector<float> gammaData(channels), betaData(channels);
        std::vector<float> meanData(channels), varianceData(channels);
        for (const vai::Tensor* tensor : {&gamma, &beta, &mean, &variance})
        {
            VAI_ASSERT(tensor->hasHostData() && tensor->isShapeOf(channels));
            VAI_ASSERT(tensor->dtype() == wt);
        }
        vai::convertDtype(gamma.hostData(), wt, gammaData.data(), vai::Dtype::Fp32, channels);
        vai::convertDtype(beta.hostData(), wt, betaData.data(), vai::Dtype::Fp32, channels);
        vai::convertDtype(mean.hostData(), wt, meanData.data(), vai::Dtype::Fp32, channels);
        vai::convertDtype(variance.hostData(), wt, varianceData.data(), vai::Dtype::Fp32, channels);

        std::vector<float> scale(channels), bias(channels);
        for (uint32_t c = 0; c < channels; ++c)
        {
            scale[c] = gammaData[c] / std::sqrt(varianceData[c] + eps);
            bias[c] = betaData[c] - meanData[c] * scale[c];
        }
        conv["scale"] = vai::Tensor(std::vector<uint32_t>{channels}, wt).setFloats(std::move(scale));
        conv["bias"] = vai::Tensor(std::vector<uint32_t>{channels}, wt).setFloats(std::move(bias));
    };

    auto loadBaseConv = [&](auto& conv, const std::string& path)
    {
        const vai::Dtype wt = conv.config_.wDtype;
        conv["weight"]   = vai::Tensor(ckpt[path + ".conv.weight"], wt);
        loadBatchNorm(conv,
            vai::Tensor(ckpt[path + ".bn.weight"], wt),
            vai::Tensor(ckpt[path + ".bn.bias"], wt),
            vai::Tensor(ckpt[path + ".bn.running_mean"], wt),
            vai::Tensor(ckpt[path + ".bn.running_var"], wt));
    };

    auto loadPrediction = [&](vai::node::Convolution& conv, const std::string& path)
    {
        const vai::Dtype wt = conv.config_.wDtype;
        conv["weight"] = vai::Tensor(ckpt[path + ".weight"], wt);
        conv["bias"] = vai::Tensor(ckpt[path + ".bias"], wt);
    };

    auto loadBottleneck = [&](Bottleneck& bottleneck, const std::string& path)
    {
        loadBaseConv(bottleneck.conv1, path + ".conv1");
        loadBaseConv(bottleneck.conv2, path + ".conv2");
    };

    auto loadCsp = [&](CSPLayer& csp, const std::string& path)
    {
        loadBaseConv(csp.conv1, path + ".conv1");
        loadBaseConv(csp.conv2, path + ".conv2");
        loadBaseConv(csp.conv3, path + ".conv3");
        for (uint32_t i = 0; i < csp.bottlenecks.size(); ++i)
            loadBottleneck(csp.bottlenecks[i], path + ".m." + std::to_string(i));
    };

    auto loadDownStage = [&]
    (
        Backbone::DownStage& stage,
        const std::string& path
    )
    {
        loadBaseConv(stage.conv, path + ".0");
        loadCsp(stage.csp, path + ".1");
    };

    auto& backbone = model.backbone;
    const std::string backbonePath = "backbone.backbone";
    loadBaseConv(backbone.stem.conv, backbonePath + ".stem.conv");
    loadDownStage(backbone.dark2, backbonePath + ".dark2");
    loadDownStage(backbone.dark3, backbonePath + ".dark3");
    loadDownStage(backbone.dark4, backbonePath + ".dark4");
    loadBaseConv(backbone.dark5.conv, backbonePath + ".dark5.0");
    loadBaseConv(backbone.dark5.spp.conv1, backbonePath + ".dark5.1.conv1");
    loadBaseConv(backbone.dark5.spp.conv2, backbonePath + ".dark5.1.conv2");
    loadCsp(backbone.dark5.csp, backbonePath + ".dark5.2");

    auto& neck = model.neck;
    const std::string neckPath = "backbone";
    loadBaseConv(neck.lateralConv0, neckPath + ".lateral_conv0");
    loadCsp(neck.c3P4, neckPath + ".C3_p4");
    loadBaseConv(neck.reduceConv1, neckPath + ".reduce_conv1");
    loadCsp(neck.c3P3, neckPath + ".C3_p3");
    loadBaseConv(neck.bottomUpConv2, neckPath + ".bu_conv2");
    loadCsp(neck.c3N3, neckPath + ".C3_n3");
    loadBaseConv(neck.bottomUpConv1, neckPath + ".bu_conv1");
    loadCsp(neck.c3N4, neckPath + ".C3_n4");

    auto loadHeadBranch = [&]
    (
        Head::PredictionBranch& branch,
        uint32_t index
    )
    {
        const std::string i = std::to_string(index);
        loadBaseConv(branch.stem, "head.stems." + i);
        loadBaseConv(branch.clsConv1, "head.cls_convs." + i + ".0");
        loadBaseConv(branch.clsConv2, "head.cls_convs." + i + ".1");
        loadBaseConv(branch.regConv1, "head.reg_convs." + i + ".0");
        loadBaseConv(branch.regConv2, "head.reg_convs." + i + ".1");
        loadPrediction(branch.clsPred, "head.cls_preds." + i);

        const vai::Dtype wt = branch.regObjPred.config_.wDtype;
        const std::string regPath = "head.reg_preds." + i;
        const std::string objPath = "head.obj_preds." + i;
        branch.regObjPred["weight"] =
            vai::concat({ckpt[regPath + ".weight"], ckpt[objPath + ".weight"]}, wt);
        branch.regObjPred["bias"] =
            vai::concat({ckpt[regPath + ".bias"], ckpt[objPath + ".bias"]}, wt);
    };

    loadHeadBranch(model.head.p3, 0u);
    loadHeadBranch(model.head.p4, 1u);
    loadHeadBranch(model.head.p5, 2u);
}

} // namespace yolox::megvii


template<>
inline void loadWeights<"megvii", yolox::Model>(yolox::Model& model)
{
    const std::string dir = std::string(VAI_DATA_DIR "/yolox/ckpts/") + yolox::tierName(model.cfg.tier);
    yolox::megvii::loadWeights(model, vai::Checkpoint(dir));
}

#endif // YOLOX_MEGVII_WEIGHTS_H
