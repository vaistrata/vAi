/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each policy reads every *.safetensors file in
               data/ckpts/act/<ModelConfig::variant>/, taken from the matching
               LeRobot policy on HuggingFace:
                 lerobot/transfer-cube       lerobot/act_aloha_sim_transfer_cube_human
                 mair-robosyn/item-assembly  mair-robosyn/ACT_sim_item_assembly_official
                 ...                         or any other LeRobot ACT policy within Supported
                 
  Supported    Any number of cameras, all one size
               Any state width, action width and chunk size
               MEAN_STD statistics, in model.safetensors or the step 3 normalizer file
  Unsupported  Cameras of different sizes
               An architecture other than ModelConfig's (not checked)
               observation.environment_state
               MIN_MAX or IDENTITY normalization
               A preprocessing step ahead of the normalizer, e.g. relative actions
=============================================================================*/

#ifndef PRODUCTION_ACT_LEROBOT_WEIGHTS_H
#define PRODUCTION_ACT_LEROBOT_WEIGHTS_H

#include "net.h"
#include "vai/engine/weights.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>


namespace model::production::act::lerobot {


// LeRobot's normalizer divides by std + 1e-8.
constexpr float kStdEps = 1e-8f;

// torchvision's FrozenBatchNorm2d.
constexpr float kBatchNormEps = 1e-5f;


// Camera i's name, which keys imageMean / imageStd, and its input feature, which keys
// its statistics in the checkpoint.
inline std::string cameraName(const ModelConfig& cfg, uint32_t i)
{
    return cfg.cameras.empty() ? "image" : cfg.cameras[i];
}
inline std::string cameraFeature(const ModelConfig& cfg, uint32_t i)
{
    return cfg.cameras.empty() ? "observation.image" : "observation.images." + cfg.cameras[i];
}


// The image tokens' positions: DETR's normalized 2D sine embedding, the y half
// then the x half, each alternating sin and cos. Token t sits at row t / w,
// column t % w.
inline std::vector<float> imagePositions(uint32_t h, uint32_t w, uint32_t dim)
{
    const uint32_t half = dim / 2;
    const double twoPi = 2.0 * std::numbers::pi;
    std::vector<float> pos(size_t(h) * w * dim);
    for (uint32_t i = 0; i < h; ++i)
    {
        for (uint32_t j = 0; j < w; ++j)
        {
            const double y = (i + 1) / (h + 1e-6) * twoPi;
            const double x = (j + 1) / (w + 1e-6) * twoPi;
            float* row = pos.data() + (size_t(i) * w + j) * dim;
            for (uint32_t k = 0; k < half; ++k)
            {
                const double freq = std::pow(10000.0, 2.0 * (k / 2) / half);
                row[k] = float(k % 2 == 0 ? std::sin(y / freq) : std::cos(y / freq));
                row[half + k] = float(k % 2 == 0 ? std::sin(x / freq) : std::cos(x / freq));
            }
        }
    }
    return pos;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps LeRobot's ACTPolicy state dict to the vAi graph
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Net& net, const vai::Checkpoint& ckpt)
{
    // Weights load at the net's wDefaultDtype, norm coefficients at Fp32.
    const vai::Dtype wDt = net.config().wDefaultDtype;
    const vai::Dtype normDt = vai::Dtype::Fp32;

    const ModelConfig& cfg = net.modelCfg;
    const uint32_t D = cfg.embedDim;

    auto floats = [&](const std::string& key) {
        vai::Tensor t(ckpt[key], vai::Dtype::Fp32);
        const float* p = static_cast<const float*>(t.hostData());
        return std::vector<float>(p, p + t.numElements());
    };
    // rows [first, first + count) of a [rows, cols] matrix, or of a vector when cols is 1
    auto rows = [](const std::vector<float>& m, uint32_t cols, uint32_t first, uint32_t count) {
        return std::vector<float>(m.begin() + size_t(first) * cols, m.begin() + size_t(first + count) * cols);
    };
    auto tensor = [](std::vector<uint32_t> shape, vai::Dtype dt, std::vector<float> data) {
        return vai::Tensor(std::move(shape), dt).setFloats(std::move(data));
    };

    auto loadLin = [&](vai::node::Linear& lin, const std::string& prefix) {
        lin["weight"] = vai::Tensor(ckpt[prefix + ".weight"], wDt);
        lin["bias"] = vai::Tensor(ckpt[prefix + ".bias"], wDt);
    };
    auto loadLayerNorm = [&](vai::node::LayerNorm& ln, const std::string& prefix) {
        ln["scale"] = vai::Tensor(ckpt[prefix + ".weight"], normDt);
        ln["shift"] = vai::Tensor(ckpt[prefix + ".bias"], normDt);
    };

    // Backbone - every BatchNorm folds into the conv ahead of it.
    auto loadBnConv = [&](vai::node::Convolution& conv, const std::string& convKey, const std::string& bnKey) {
        const uint32_t channels = conv.config_.outChannels;
        conv["weight"] = vai::Tensor(ckpt[convKey + ".weight"], wDt);

        const auto gamma = floats(bnKey + ".weight");
        const auto beta = floats(bnKey + ".bias");
        const auto mean = floats(bnKey + ".running_mean");
        const auto var = floats(bnKey + ".running_var");
        std::vector<float> scale(channels), bias(channels);
        for (uint32_t c = 0; c < channels; ++c)
        {
            scale[c] = gamma[c] / std::sqrt(var[c] + kBatchNormEps);
            bias[c] = beta[c] - mean[c] * scale[c];
        }
        conv["scale"] = tensor({ channels }, wDt, std::move(scale));
        conv["bias"] = tensor({ channels }, wDt, std::move(bias));
    };
    {
        const std::string root = "model.backbone.";
        auto& backbone = net.imageEncoder;
        loadBnConv(backbone.conv1.conv, root + "conv1", root + "bn1");

        size_t index = 0;
        for (uint32_t l = 0; l < 4; ++l)
        {
            for (uint32_t b = 0; b < backbone.depths[l]; ++b)
            {
                const std::string path = root + "layer" + std::to_string(l + 1) + "." + std::to_string(b) + ".";
                vai::group::BasicBlock& block = backbone.blocks[index++];
                loadBnConv(block.conv1.conv, path + "conv1", path + "bn1");
                loadBnConv(block.conv2.conv, path + "conv2", path + "bn2");
                if (block.shortcut)
                    loadBnConv(block.shortcut->conv, path + "downsample.0", path + "downsample.1");
            }
        }
    }

    {
        // The checkpoint's 1x1 conv kernel [D, C, 1, 1] is the Linear's [D, C].
        net.imageProj["weight"] = vai::Tensor(ckpt["model.encoder_img_feat_input_proj.weight"], wDt)
                                      .reshape({ net.imageProj.config_.outDim, net.imageProj.config_.inDim });
        net.imageProj["bias"] = vai::Tensor(ckpt["model.encoder_img_feat_input_proj.bias"], wDt);
    }

    // The statistics the caller normalizes with: a feature's mean or std, from
    // model.safetensors's buffers in an older export, else from the processor file.
    const bool bufferStats = ckpt.contains("normalize_inputs.buffer_observation_state.mean");
    auto stat = [&](const std::string& feature, const char* which) {
        if (!bufferStats)
            return floats(feature + "." + which);
        std::string buffer = feature;
        std::replace(buffer.begin(), buffer.end(), '.', '_');
        return floats((feature == "action" ? "unnormalize_outputs.buffer_" : "normalize_inputs.buffer_") + buffer +
                      "." + which);
    };
    {
        for (uint32_t i = 0; i < cfg.numCameras(); ++i)
        {
            const auto mean = stat(cameraFeature(cfg, i), "mean");
            const auto std = stat(cameraFeature(cfg, i), "std");
            auto& imageMean = net.imageMean[cameraName(cfg, i)];
            auto& imageStd = net.imageStd[cameraName(cfg, i)];
            for (uint32_t c = 0; c < 3; ++c)
            {
                imageMean[c] = mean[c];
                imageStd[c] = std[c] + kStdEps;
            }
        }

        net.stateMean = stat("observation.state", "mean");
        net.stateStd = stat("observation.state", "std");
        for (float& s : net.stateStd)
            s += kStdEps;
        net.actionMean = stat("action", "mean");
        net.actionStd = stat("action", "std");
        VAI_ASSERT(net.stateMean.size() == cfg.stateDim && net.actionMean.size() == cfg.actionDim);
    }

    // Latent and state tokens - one projection of the normalized state. The latent
    // is zero at inference, so its token is the latent projection's bias.
    {
        const auto latentBias = floats("model.encoder_latent_input_proj.bias");
        const auto stateW = floats("model.encoder_robot_state_input_proj.weight");   // [D, S]
        const auto stateB = floats("model.encoder_robot_state_input_proj.bias");

        const uint32_t S = cfg.stateDim;
        VAI_ASSERT(stateW.size() == size_t(D) * S);
        std::vector<float> weight(size_t(2) * D * S, 0.0f), bias(size_t(2) * D);
        for (uint32_t r = 0; r < D; ++r)
        {
            bias[r] = latentBias[r];
            bias[D + r] = stateB[r];
            for (uint32_t s = 0; s < S; ++s)
                weight[size_t(D + r) * S + s] = stateW[size_t(r) * S + s];
        }
        net.latentStateProj["weight"] = tensor({ 2 * D, S }, wDt, std::move(weight));
        net.latentStateProj["bias"] = tensor({ 2 * D }, wDt, std::move(bias));
    }

    // Encoder positions [2 + B*h*w, D], row for row the tokens catTokens stacks:
    //   rows 0, 1  the latent and the state token, learned (encoder_1d_feature_pos_embed [2, D];
    //              a third row, the environment state's, is not supported)
    //   rows 2 ..  the B cameras' tokens, each camera DETR's 2D sine on its own h x w
    //              feature map (imageGridHeight x imageGridWidth, 15 x 20 at 480 x 640),
    //              row-major like flatten + transpose lay the map out: camera n's cell
    //              (i, j) is token 2 + (n*h + i)*w + j
    // The encoder's queries and keys and the decoder's keys read x + pos, which reaches
    // them as bias rows (projBias). pos is kept for one camera, [2 + h*w, D]; the cameras
    // share their rows, and so do the bias rows, which forCameras copies per camera.
    auto pos = floats("model.encoder_1d_feature_pos_embed.weight");
    VAI_ASSERT(pos.size() == size_t(2) * D);
    {
        const auto image = imagePositions(cfg.imageGridHeight(), cfg.imageGridWidth(), D);
        pos.insert(pos.end(), image.begin(), image.end());
    }
    auto forCameras = [&](std::vector<float> m, uint32_t cols) {
        const std::vector<float> image(m.begin() + size_t(2) * cols, m.end());
        for (uint32_t n = 1; n < cfg.numCameras(); ++n)
            m.insert(m.end(), image.begin(), image.end());
        VAI_ASSERT(m.size() == size_t(cfg.encoderTokens()) * cols);
        return m;
    };

    // b + pos · Wᵀ on the first posCols columns and b alone on the rest, one row per
    // position. W's first posCols rows are transposed so the inner loop runs along a
    // contiguous row of both.
    auto projBias = [&](const std::vector<float>& weight, const std::vector<float>& bias, uint32_t posCols) {
        const uint32_t cols = uint32_t(bias.size());
        std::vector<float> weightT(size_t(D) * posCols);
        for (uint32_t r = 0; r < posCols; ++r)
            for (uint32_t c = 0; c < D; ++c)
                weightT[size_t(c) * posCols + r] = weight[size_t(r) * D + c];

        const size_t positions = pos.size() / D;
        std::vector<float> out(positions * cols);
        for (size_t t = 0; t < positions; ++t)
        {
            float* row = out.data() + t * cols;
            std::copy(bias.begin(), bias.end(), row);
            for (uint32_t c = 0; c < D; ++c)
            {
                const float p = pos[t * D + c];
                const float* w = weightT.data() + size_t(c) * posCols;
                for (uint32_t r = 0; r < posCols; ++r)
                    row[r] += p * w[r];
            }
        }
        return forCameras(std::move(out), cols);
    };

    // nn.MultiheadAttention stacks Q, K and V into one in_proj [3D, D]. A fused projection
    // takes count of them from first, in that order; the leading withPos of them read
    // x + pos, which reaches them as bias rows, so the projection reads x alone.
    auto loadInProj = [&](vai::node::Linear& lin,
                          const std::vector<float>& inW,
                          const std::vector<float>& inB,
                          uint32_t first,
                          uint32_t count,
                          uint32_t withPos) {
        auto weight = rows(inW, D, first * D, count * D);
        const auto bias = rows(inB, 1, first * D, count * D);
        lin["bias"] = tensor({ cfg.encoderTokens(), count * D }, wDt, projBias(weight, bias, withPos * D));
        lin["weight"] = tensor({ count * D, D }, wDt, std::move(weight));
    };

    for (uint32_t i = 0; i < cfg.numEncoderBlocks; ++i)
    {
        const std::string path = "model.encoder.layers." + std::to_string(i) + ".";
        auto& block = net.encoder[i];
        const auto inW = floats(path + "self_attn.in_proj_weight");
        const auto inB = floats(path + "self_attn.in_proj_bias");
        loadInProj(block.mhAttn.projQKV, inW, inB, 0, 3, 2);   // Q, K with pos; V
        loadLin(block.mhAttn.projOut, path + "self_attn.out_proj");
        loadLayerNorm(block.norm1, path + "norm1");
        loadLin(block.ffn.fc1, path + "linear1");
        loadLin(block.ffn.fc2, path + "linear2");
        loadLayerNorm(block.norm2, path + "norm2");
    }

    // Decoder - its input is zeros, so the self-attention, norm1 and the queries are
    // computed here. V of zeros is V's bias on every row, every attention row sums
    // to one, so the self-attention leaves the same row everywhere.
    {
        const std::string path = "model.decoder.layers.0.";
        auto& decoder = net.decoder;

        const auto selfBias = floats(path + "self_attn.in_proj_bias");
        const auto selfOutW = floats(path + "self_attn.out_proj.weight");
        const auto selfOutB = floats(path + "self_attn.out_proj.bias");
        const auto gamma1 = floats(path + "norm1.weight");
        const auto beta1 = floats(path + "norm1.bias");

        std::vector<float> x1(D);
        for (uint32_t r = 0; r < D; ++r)
        {
            double acc = selfOutB[r];
            for (uint32_t c = 0; c < D; ++c)
                acc += double(selfOutW[size_t(r) * D + c]) * selfBias[2 * D + c];
            x1[r] = float(acc);
        }
        {
            double mean = 0.0, var = 0.0;
            for (float v : x1)
                mean += v;
            mean /= D;
            for (float v : x1)
                var += (v - mean) * (v - mean);
            var /= D;
            const double inv = 1.0 / std::sqrt(var + cfg.normEps);
            for (uint32_t r = 0; r < D; ++r)
                x1[r] = float((x1[r] - mean) * inv * gamma1[r] + beta1[r]);
        }

        // query = (x1 + its position) W_q + b_q, written per head as [H, actionChunk, Dh].
        const uint32_t Dh = D / cfg.numHeads;
        const auto crossW = floats(path + "multihead_attn.in_proj_weight");
        const auto crossB = floats(path + "multihead_attn.in_proj_bias");
        const auto queryPos = floats("model.decoder_pos_embed.weight");   // [actionChunk, D]
        VAI_ASSERT(queryPos.size() == size_t(cfg.actionChunk) * D);
        std::vector<float> query(size_t(cfg.actionChunk) * D);
        std::vector<float> in(D);
        for (uint32_t t = 0; t < cfg.actionChunk; ++t)
        {
            for (uint32_t c = 0; c < D; ++c)
                in[c] = x1[c] + queryPos[size_t(t) * D + c];
            for (uint32_t r = 0; r < D; ++r)
            {
                double acc = crossB[r];
                for (uint32_t c = 0; c < D; ++c)
                    acc += double(crossW[size_t(r) * D + c]) * in[c];
                query[(size_t(r / Dh) * cfg.actionChunk + t) * Dh + r % Dh] = float(acc);
            }
        }
        net.decoderQuery = tensor({ cfg.numHeads, cfg.actionChunk, Dh }, net.config().oDefaultDtype, std::move(query));

        loadInProj(decoder.projKV, crossW, crossB, 1, 2, 1);   // K with pos; V

        // x1 is the cross-attention's residual; it rides in the output projection's bias.
        {
            auto bias = floats(path + "multihead_attn.out_proj.bias");
            for (uint32_t r = 0; r < D; ++r)
                bias[r] += x1[r];
            decoder.projOut["weight"] = vai::Tensor(ckpt[path + "multihead_attn.out_proj.weight"], wDt);
            decoder.projOut["bias"] = tensor({ D }, wDt, std::move(bias));
        }
        loadLayerNorm(decoder.norm1, path + "norm2");
        loadLin(decoder.ffn.fc1, path + "linear1");
        loadLin(decoder.ffn.fc2, path + "linear2");
        loadLayerNorm(decoder.norm2, path + "norm3");
    }
    loadLayerNorm(net.decoderNorm, "model.decoder.norm");

    // Action head, in the normalized units.
    {
        auto weight = floats("model.action_head.weight");   // [A, D]
        auto bias = floats("model.action_head.bias");
        const uint32_t A = cfg.actionDim;
        VAI_ASSERT(weight.size() == size_t(A) * D);
        net.actionHead["weight"] = tensor({ A, D }, wDt, std::move(weight));
        net.actionHead["bias"] = tensor({ A }, wDt, std::move(bias));
    }
}

}   // namespace model::production::act::lerobot


// loadWeights<"lerobot">(net) - composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"lerobot", model::production::act::Net>(model::production::act::Net& net)
{
    const std::string dir = std::string(VAI_DATA_DIR "/ckpts/act/") + net.modelCfg.variant;

    // LeRobot's processor pipeline normalizes in its step 3. A step ahead of it (e.g.
    // relative actions) moves the file to another step, which this loader does not
    // run, and a stale file of another step would shadow step 3's keys.
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        const std::string name = entry.path().filename().string();
        if (name.starts_with("policy_preprocessor_") && name.ends_with("normalizer_processor.safetensors") &&
            name != "policy_preprocessor_step_3_normalizer_processor.safetensors")
            throw std::runtime_error("unsupported LeRobot preprocessing: " + dir + "/" + name);
    }

    model::production::act::lerobot::loadWeights(net, vai::Checkpoint(dir));
}

#endif   // PRODUCTION_ACT_LEROBOT_WEIGHTS_H
