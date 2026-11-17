/*=============================================================================
  Written by   qkrtmddn0507@vaistrata.com
  Checkpoint   Each policy reads every *.safetensors file in
               data/ckpts/diffusion-policy/lerobot/<ModelConfig::variant>/, taken from
               the matching LeRobot policy on HuggingFace:
                 pusht  lerobot/diffusion_pusht

  Supported    One shared RGB encoder over every camera (use_separate_rgb_encoder_per_camera
               false), a GroupNorm or BatchNorm ResNet18 backbone
               Images MEAN_STD, state and action MIN_MAX (pushT's normalization_mapping),
               in model.safetensors's buffers or the step 3 normalizer file
  Unsupported  observation.environment_state
               A separate RGB encoder per camera
               An architecture other than ModelConfig's (not checked)
               A preprocessing step ahead of the normalizer
=============================================================================*/

#ifndef EXPERIMENTAL_DIFFUSION_POLICY_LEROBOT_WEIGHTS_H
#define EXPERIMENTAL_DIFFUSION_POLICY_LEROBOT_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>


namespace model::experimental::diffusion_policy::lerobot {


// LeRobot's normalizer divides by std + 1e-8, and by max − min + 1e-8.
constexpr float kStatEps = 1e-8f;

// torchvision's BatchNorm2d, for a BatchNorm backbone.
constexpr float kBatchNormEps = 1e-5f;


/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps LeRobot's DiffusionPolicy state dict to the vAi graph
//
// diffusion.rgb_encoder  backbone.{0: conv1, 1: bn1, 4..7: layer1..4}, pool.nets, out
// diffusion.unet         diffusion_step_encoder.{1, 3}, down_modules / mid_modules /
//                        up_modules (residual blocks, then a resample), final_conv
//
// A Conv1dBlock is "block": block.0 the Conv1d, block.1 the GroupNorm.
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32. The
    // Conv1d's coopmat path takes fp16 weights whatever the default is.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;
    const vai::Dtype conv1dDt = vai::Dtype::Fp16;

    const ModelConfig& cfg = net.cfg;

    auto floats = [&](const std::string& key) {
        vai::Tensor t(ckpt[key], vai::Dtype::Fp32);
        const float* p = static_cast<const float*>(t.hostData());
        return std::vector<float>(p, p + t.numElements());
    };
    auto tensor = [](std::vector<uint32_t> shape, vai::Dtype dt, std::vector<float> data) {
        return vai::Tensor(std::move(shape), dt).setFloats(std::move(data));
    };

    auto loadLin = [&](vai::node::Linear& lin, const std::string& prefix) {
        lin["weight"] = vai::Tensor(ckpt[prefix + ".weight"], wDt);
        lin["bias"] = vai::Tensor(ckpt[prefix + ".bias"], wDt);
    };
    auto loadNorm = [&](auto& norm, const std::string& prefix) {
        norm["scale"] = vai::Tensor(ckpt[prefix + ".weight"], normDt);
        norm["shift"] = vai::Tensor(ckpt[prefix + ".bias"], normDt);
    };

    // Observation encoder
    {
        const std::string root = "diffusion.rgb_encoder.";
        group::RgbEncoder& enc = net.rgbEncoder;

        // Backbone - torchvision's convs carry no bias. GroupNorm is a node of its
        // own; a BatchNorm folds into the conv ahead of it.
        auto loadConvNorm = [&](vai::group::ConvNorm& cn, const std::string& convKey, const std::string& normKey) {
            cn.conv["weight"] = vai::Tensor(ckpt[convKey + ".weight"], wDt);
            if (cn.norm)
            {
                loadNorm(*cn.norm, normKey);
                return;
            }
            const uint32_t channels = cn.conv.config_.outChannels;
            const auto gamma = floats(normKey + ".weight");
            const auto beta = floats(normKey + ".bias");
            const auto mean = floats(normKey + ".running_mean");
            const auto var = floats(normKey + ".running_var");
            std::vector<float> scale(channels), bias(channels);
            for (uint32_t c = 0; c < channels; ++c)
            {
                scale[c] = gamma[c] / std::sqrt(var[c] + kBatchNormEps);
                bias[c] = beta[c] - mean[c] * scale[c];
            }
            cn.conv["scale"] = tensor({ channels }, wDt, std::move(scale));
            cn.conv["bias"] = tensor({ channels }, wDt, std::move(bias));
        };

        // nn.Sequential(*resnet.children()[:-2]): 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4..7 layer1..4.
        const std::string bb = root + "backbone.";
        auto& backbone = enc.backbone;
        loadConvNorm(backbone.conv1, bb + "0", bb + "1");
        size_t index = 0;
        for (uint32_t l = 0; l < 4; ++l)
        {
            for (uint32_t b = 0; b < backbone.depths[l]; ++b)
            {
                const std::string path = bb + std::to_string(4 + l) + "." + std::to_string(b) + ".";
                vai::group::BasicBlock& block = backbone.blocks[index++];
                loadConvNorm(block.conv1, path + "conv1", path + "bn1");
                loadConvNorm(block.conv2, path + "conv2", path + "bn2");
                if (block.shortcut)
                    loadConvNorm(*block.shortcut, path + "downsample.0", path + "downsample.1");
            }
        }

        // SpatialSoftmax: its num_kp projection, and the grid when the checkpoint
        // carries it as a buffer (pos_grid [H*W, 2], the Linear's weight transposed).
        enc.keypointProj["weight"] = vai::Tensor(ckpt[root + "pool.nets.weight"], wDt);
        enc.keypointProj["bias"] = vai::Tensor(ckpt[root + "pool.nets.bias"], wDt);
        if (ckpt.contains(root + "pool.pos_grid"))
        {
            const auto grid = floats(root + "pool.pos_grid");
            const uint32_t HW = enc.spatialSoftmax.HW;
            VAI_ASSERT(grid.size() == size_t(HW) * 2u);
            std::vector<float> gridT(2u * HW);
            for (uint32_t j = 0; j < HW; ++j)
            {
                gridT[j] = grid[2u * j];
                gridT[HW + j] = grid[2u * j + 1u];
            }
            enc.spatialSoftmax.expectedXy["weight"] = tensor({ 2u, HW }, vai::Dtype::Fp32, std::move(gridT));
        }

        loadLin(enc.out, root + "out");
    }

    // Action denoiser
    {
        const std::string root = "diffusion.unet.";
        group::ConditionalUnet1d& unet = net.unet;

        auto loadConv1d = [&](node::Convolution1d& conv, const std::string& prefix) {
            conv["weight"] = vai::Tensor(ckpt[prefix + ".weight"], conv1dDt);
            conv["bias"] = vai::Tensor(ckpt[prefix + ".bias"], wDt);
        };
        auto loadConv1dBlock = [&](group::Conv1dBlock& block, const std::string& prefix) {
            loadConv1d(block.conv, prefix + ".block.0");
            loadNorm(block.norm, prefix + ".block.1");
        };
        auto loadResBlock = [&](group::ConditionalResidualBlock1d& block, const std::string& prefix) {
            loadConv1dBlock(block.conv1, prefix + "conv1");
            loadLin(block.condEncoder, prefix + "cond_encoder.1");   // .0 is the Mish, run once by the UNet
            loadConv1dBlock(block.conv2, prefix + "conv2");
            if (block.residualConv)
                loadConv1d(*block.residualConv, prefix + "residual_conv");
        };

        loadLin(unet.stepEncoder.fc1, root + "diffusion_step_encoder.1");
        loadLin(unet.stepEncoder.fc2, root + "diffusion_step_encoder.3");

        const uint32_t L = uint32_t(cfg.downDims.size());
        for (uint32_t l = 0; l < L; ++l)
        {
            const std::string path = root + "down_modules." + std::to_string(l) + ".";
            loadResBlock(unet.downBlocks[2 * l], path + "0.");
            loadResBlock(unet.downBlocks[2 * l + 1], path + "1.");
            if (unet.downsamples[l])
                loadConv1d(*unet.downsamples[l], path + "2");
        }
        loadResBlock(unet.midBlocks[0], root + "mid_modules.0.");
        loadResBlock(unet.midBlocks[1], root + "mid_modules.1.");
        for (uint32_t u = 0; u + 1 < L; ++u)
        {
            const std::string path = root + "up_modules." + std::to_string(u) + ".";
            loadResBlock(unet.upBlocks[2 * u], path + "0.");
            loadResBlock(unet.upBlocks[2 * u + 1], path + "1.");
            unet.upsamples[u]["weight"] = vai::Tensor(ckpt[path + "2.weight"], wDt);   // [Ci, Co, K]
            unet.upsamples[u]["bias"] = vai::Tensor(ckpt[path + "2.bias"], wDt);
        }

        loadConv1dBlock(unet.finalBlock, root + "final_conv.0");
        loadConv1d(unet.finalConv, root + "final_conv.1");
    }

    // The statistics the caller normalizes with, from model.safetensors's buffers in
    // an older export, else from the processor file.
    const bool bufferStats = ckpt.contains("normalize_inputs.buffer_observation_state.min");
    auto stat = [&](const std::string& feature, const char* which) {
        if (!bufferStats)
            return floats(feature + "." + which);
        std::string buffer = feature;
        std::replace(buffer.begin(), buffer.end(), '.', '_');
        return floats((feature == "action" ? "unnormalize_outputs.buffer_" : "normalize_inputs.buffer_") + buffer +
                      "." + which);
    };
    {
        const auto mean = stat("observation.image", "mean");   // [3, 1, 1]
        const auto std = stat("observation.image", "std");
        VAI_ASSERT(mean.size() == 3u && std.size() == 3u);
        for (uint32_t c = 0; c < 3; ++c)
        {
            net.imageMean[c] = mean[c];
            net.imageStd[c] = std[c] + kStatEps;
        }

        net.stateMin = stat("observation.state", "min");
        net.stateMax = stat("observation.state", "max");
        net.actionMin = stat("action", "min");
        net.actionMax = stat("action", "max");
        VAI_ASSERT(net.stateMin.size() == cfg.stateDim && net.stateMax.size() == cfg.stateDim);
        VAI_ASSERT(net.actionMin.size() == cfg.actionDim && net.actionMax.size() == cfg.actionDim);
    }
}

}   // namespace model::experimental::diffusion_policy::lerobot


// loadWeights<"lerobot">(net) - composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"lerobot", model::experimental::diffusion_policy::Net>(
    model::experimental::diffusion_policy::Net& net)
{
    const std::string dir = std::string(VAI_DATA_DIR "/ckpts/diffusion-policy/lerobot/") + net.cfg.variant;

    // LeRobot's processor pipeline normalizes in its step 3. A step ahead of it moves
    // the file to another step, which this loader does not run, and a stale file of
    // another step would shadow step 3's keys.
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        const std::string name = entry.path().filename().string();
        if (name.starts_with("policy_preprocessor_") && name.ends_with("normalizer_processor.safetensors") &&
            name != "policy_preprocessor_step_3_normalizer_processor.safetensors")
            throw std::runtime_error("unsupported LeRobot preprocessing: " + dir + "/" + name);
    }

    model::experimental::diffusion_policy::lerobot::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_LEROBOT_WEIGHTS_H
