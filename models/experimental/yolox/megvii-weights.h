/*=============================================================================
  Written by   phg9296@vaistrata.com
  Checkpoint   Each tier reads data/ckpts/yolox/megvii/<tierName>/model.safetensors.
               Megvii ships PyTorch checkpoints in the official YOLOX release:
                 download   github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
                              -> data/ckpts/yolox/megvii/<tierName>/model.pth
                 convert    megvii-weights-convertor.py
                              -> data/ckpts/yolox/megvii/<tierName>/model.safetensors
=============================================================================*/

#ifndef EXPERIMENTAL_YOLOX_MEGVII_WEIGHTS_H
#define EXPERIMENTAL_YOLOX_MEGVII_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <string>
#include <cmath>


namespace model::experimental::yolox::megvii {

/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - map the official YOLOX state dict to the vAi graph
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype.
    const vai::Dtype wDt = net.config().wDefaultDtype;

    // Match the checkpoint's BN epsilon and the previous weight-dtype rounding.
    auto loadBatchNorm =
        [&](auto& conv,
            const vai::Tensor& gamma,
            const vai::Tensor& beta,
            const vai::Tensor& mean,
            const vai::Tensor& variance) {
            constexpr float eps = 1e-3f;
            const uint32_t channels = conv.config_.outChannels;
            std::vector<float> gammaData(channels), betaData(channels);
            std::vector<float> meanData(channels), varianceData(channels);
            for (const vai::Tensor* tensor : { &gamma, &beta, &mean, &variance })
            {
                VAI_ASSERT(tensor->hasHostData() && tensor->isShapeOf(channels));
                VAI_ASSERT(tensor->dtype() == wDt);
            }
            vai::convertDtype(gamma.hostData(), wDt, gammaData.data(), vai::Dtype::Fp32, channels);
            vai::convertDtype(beta.hostData(), wDt, betaData.data(), vai::Dtype::Fp32, channels);
            vai::convertDtype(mean.hostData(), wDt, meanData.data(), vai::Dtype::Fp32, channels);
            vai::convertDtype(variance.hostData(), wDt, varianceData.data(), vai::Dtype::Fp32, channels);

            std::vector<float> scale(channels), bias(channels);
            for (uint32_t c = 0; c < channels; ++c)
            {
                scale[c] = gammaData[c] / std::sqrt(varianceData[c] + eps);
                bias[c] = betaData[c] - meanData[c] * scale[c];
            }
            conv["scale"] = vai::Tensor(std::vector<uint32_t>{ channels }, wDt).setFloats(std::move(scale));
            conv["bias"] = vai::Tensor(std::vector<uint32_t>{ channels }, wDt).setFloats(std::move(bias));
        };

    auto loadBaseConv = [&](auto& conv, const std::string& path) {
        conv["weight"] = vai::Tensor(ckpt[path + ".conv.weight"], wDt);
        loadBatchNorm(conv,
                      vai::Tensor(ckpt[path + ".bn.weight"], wDt),
                      vai::Tensor(ckpt[path + ".bn.bias"], wDt),
                      vai::Tensor(ckpt[path + ".bn.running_mean"], wDt),
                      vai::Tensor(ckpt[path + ".bn.running_var"], wDt));
    };

    auto loadPrediction = [&](vai::node::Convolution& conv, const std::string& path) {
        conv["weight"] = vai::Tensor(ckpt[path + ".weight"], wDt);
        conv["bias"] = vai::Tensor(ckpt[path + ".bias"], wDt);
    };

    auto loadBottleneck = [&](group::Bottleneck& bottleneck, const std::string& path) {
        loadBaseConv(bottleneck.conv1, path + ".conv1");
        loadBaseConv(bottleneck.conv2, path + ".conv2");
    };

    auto loadCsp = [&](group::CSPLayer& csp, const std::string& path) {
        const vai::Dtype wt = wDt;
        auto loadConvs = [&](const std::string& suffix) {
            vai::Tensor first(ckpt[path + ".conv1" + suffix], wt);
            vai::Tensor second(ckpt[path + ".conv2" + suffix], wt);
            return first.concat(second);
        };
        csp.conv12["weight"] = loadConvs(".conv.weight");
        loadBatchNorm(
            csp.conv12,
            loadConvs(".bn.weight"),
            loadConvs(".bn.bias"),
            loadConvs(".bn.running_mean"),
            loadConvs(".bn.running_var"));
        loadBaseConv(csp.conv3, path + ".conv3");
        for (uint32_t i = 0; i < csp.bottlenecks.size(); ++i)
            loadBottleneck(csp.bottlenecks[i], path + ".m." + std::to_string(i));
    };

    auto loadDownStage = [&](group::Backbone::Stage& stage, const std::string& path) {
        loadBaseConv(stage.conv, path + ".0");
        loadCsp(stage.csp, path + ".1");
    };

    auto& backbone = net.backbone;
    const std::string backbonePath = "backbone.backbone";
    loadBaseConv(backbone.stem, backbonePath + ".stem.conv");
    loadDownStage(backbone.stage1, backbonePath + ".dark2");
    loadDownStage(backbone.stage2, backbonePath + ".dark3");
    loadDownStage(backbone.stage3, backbonePath + ".dark4");
    loadBaseConv(backbone.stage4.conv, backbonePath + ".dark5.0");
    if (!backbone.stage4.spp.has_value())
        throw std::bad_optional_access();
    auto& spp = backbone.stage4.spp.value();
    loadBaseConv(spp.conv1, backbonePath + ".dark5.1.conv1");
    loadBaseConv(spp.conv2, backbonePath + ".dark5.1.conv2");
    loadCsp(backbone.stage4.csp, backbonePath + ".dark5.2");

    auto& neck = net.neck;
    const std::string neckPath = "backbone";
    loadBaseConv(neck.fpnConv0, neckPath + ".lateral_conv0");
    loadCsp(neck.fpn0, neckPath + ".C3_p4");
    loadBaseConv(neck.fpnConv1, neckPath + ".reduce_conv1");
    loadCsp(neck.fpn1, neckPath + ".C3_p3");
    loadBaseConv(neck.panConv0, neckPath + ".bu_conv2");
    loadCsp(neck.pan0, neckPath + ".C3_n3");
    loadBaseConv(neck.panConv1, neckPath + ".bu_conv1");
    loadCsp(neck.pan1, neckPath + ".C3_n4");

    auto& head = net.head;
    for (uint32_t index = 0; index < 3u; ++index)
    {
        const std::string i = std::to_string(index);
        loadBaseConv(head.stem[index], "head.stems." + i);
        auto& conv = head.fusedConv[index];
        const vai::Dtype fusedDtype = wDt;
        auto loadConvs = [&](const std::string& suffix) {
            vai::Tensor cls(ckpt[std::string("head.cls_convs.").append(i).append(".0").append(suffix)], fusedDtype);
            vai::Tensor reg(ckpt[std::string("head.reg_convs.").append(i).append(".0").append(suffix)], fusedDtype);
            return cls.concat(reg);
        };
        conv["weight"] = loadConvs(".conv.weight");
        loadBatchNorm(
            conv,
            loadConvs(".bn.weight"),
            loadConvs(".bn.bias"),
            loadConvs(".bn.running_mean"),
            loadConvs(".bn.running_var"));
        loadBaseConv(head.clsConv2[index], "head.cls_convs." + i + ".1");
        loadBaseConv(head.regConv2[index], "head.reg_convs." + i + ".1");
        loadPrediction(head.clsPred[index], "head.cls_preds." + i);
        const vai::Dtype wt = wDt;
        const std::string regPath = "head.reg_preds." + i;
        const std::string objPath = "head.obj_preds." + i;
        head.regObjPred[index]["weight"] = vai::concat({ ckpt[regPath + ".weight"], ckpt[objPath + ".weight"] }, wt);
        head.regObjPred[index]["bias"] = vai::concat({ ckpt[regPath + ".bias"], ckpt[objPath + ".bias"] }, wt);
    }
}

}   // namespace model::experimental::yolox::megvii

template <>
inline void loadWeights<"megvii", model::experimental::yolox::Net>(model::experimental::yolox::Net& net)
{
    const std::string dir =
        std::string(VAI_DATA_DIR "/ckpts/yolox/megvii/") + model::experimental::yolox::tierName(net.cfg.tier);
    model::experimental::yolox::megvii::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // EXPERIMENTAL_YOLOX_MEGVII_WEIGHTS_H
