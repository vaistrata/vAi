/*=============================================================================
  Written by   phg9296@vaistrata.com
  Checkpoint   Each tier reads every *.safetensors file in
               data/rtdetrv2/ckpts/<ModelConfig::name>/.
               Download model.safetensors from the matching Hugging Face repo:
                 small     PekingU/rtdetr_v2_r18vd
                 medium    PekingU/rtdetr_v2_r34vd
                 large     PekingU/rtdetr_v2_r50vd
                 xlarge    PekingU/rtdetr_v2_r101vd
               Keep the upstream filenames in the checkpoint folder.
               Tensors use each node's weight dtype; BatchNorm scale and bias
               are computed in FP32 before conversion.
               RepVGG branches are folded into one 3x3 kernel and bias in FP32.
=============================================================================*/

#ifndef EXPERIMENTAL_RTDETRV2_PEKINGU_WEIGHTS_H
#define EXPERIMENTAL_RTDETRV2_PEKINGU_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

namespace model::experimental::rtdetrv2::pekingu {

/////////////////////////////////////////////////////////////////////////////////////////
// loadRepVGG - Convert the two bias-free Conv+BN checkpoint branches to one Conv
// W = scale3 * W3 + center_pad(scale1 * W1), b = bias3 + bias1.
// Fold and sum in FP32, then convert once to the node's weight dtype.
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadRepVGG(vai::node::Convolution& node, const vai::Checkpoint& ckpt, const std::string& key)
{
    const auto& config = node.config_;
    const uint32_t ci = config.inChannels, co = config.outChannels;
    VAI_ASSERT(config.kernelSize == 3 && config.stride == 1 && !node.hasScale());
    std::vector<float> weight(size_t(co) * ci * 9, 0.f), bias(co, 0.f);

    auto fuseBranch = [&](const std::string& branch, uint32_t k) {
        const vai::Tensor kernel(ckpt[branch + ".conv.weight"], vai::Dtype::Fp32);
        const vai::Tensor gammaTensor(ckpt[branch + ".norm.weight"], vai::Dtype::Fp32);
        const vai::Tensor betaTensor(ckpt[branch + ".norm.bias"], vai::Dtype::Fp32);
        const vai::Tensor meanTensor(ckpt[branch + ".norm.running_mean"], vai::Dtype::Fp32);
        const vai::Tensor varianceTensor(ckpt[branch + ".norm.running_var"], vai::Dtype::Fp32);
        VAI_ASSERT(kernel.isShapeOf(co, ci, k, k));
        VAI_ASSERT(
            gammaTensor.isShapeOf(co) && betaTensor.isShapeOf(co) && meanTensor.isShapeOf(co) &&
            varianceTensor.isShapeOf(co));
        const auto* w = static_cast<const float*>(kernel.hostData());
        const auto* gamma = static_cast<const float*>(gammaTensor.hostData());
        const auto* beta = static_cast<const float*>(betaTensor.hostData());
        const auto* mean = static_cast<const float*>(meanTensor.hostData());
        const auto* variance = static_cast<const float*>(varianceTensor.hostData());
        for (uint32_t o = 0; o < co; ++o)
        {
            const float scale = gamma[o] / std::sqrt(variance[o] + 1e-5f);
            bias[o] += beta[o] - mean[o] * scale;
            for (uint32_t i = 0; i < ci; ++i)
            {
                const size_t channelPair = size_t(o) * ci + i;
                for (uint32_t p = 0; p < k * k; ++p)
                {
                    // OIHW layout: the center of a 3x3 kernel is flat offset 4.
                    const uint32_t spatial = k == 1 ? 4 : p;
                    weight[channelPair * 9 + spatial] += w[channelPair * k * k + p] * scale;
                }
            }
        }
    };
    fuseBranch(key + ".conv1", 3);
    fuseBranch(key + ".conv2", 1);
    node["weight"] = vai::Tensor({ co, ci, 3, 3 }, node.wDefaultDtype()).setFloats(std::move(weight));
    node["bias"] = vai::Tensor(std::vector<uint32_t>{ co }, node.wDefaultDtype()).setFloats(std::move(bias));
}

/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps Hugging Face RTDetrV2ForObjectDetection keys to model members
// Only the final decoder classification head is needed for inference.
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    // Keep convolution weights unchanged and apply BatchNorm through scale/bias slots.
    auto loadConv = [&](vai::node::Convolution& node, const std::string& weightKey, const std::string& normKey) {
        const auto& config = node.config_;
        const uint32_t channels = config.outChannels;
        const auto dtype = node.wDefaultDtype();
        node["weight"] = vai::Tensor(ckpt[weightKey + ".weight"], dtype);

        const vai::Tensor gammaTensor(ckpt[normKey + ".weight"], vai::Dtype::Fp32);
        const vai::Tensor betaTensor(ckpt[normKey + ".bias"], vai::Dtype::Fp32);
        const vai::Tensor meanTensor(ckpt[normKey + ".running_mean"], vai::Dtype::Fp32);
        const vai::Tensor varianceTensor(ckpt[normKey + ".running_var"], vai::Dtype::Fp32);
        VAI_ASSERT(
            gammaTensor.isShapeOf(channels) && betaTensor.isShapeOf(channels) && meanTensor.isShapeOf(channels) &&
            varianceTensor.isShapeOf(channels));
        const auto* gamma = reinterpret_cast<const float*>(gammaTensor.hostData());
        const auto* beta = reinterpret_cast<const float*>(betaTensor.hostData());
        const auto* mean = reinterpret_cast<const float*>(meanTensor.hostData());
        const auto* variance = reinterpret_cast<const float*>(varianceTensor.hostData());
        std::vector<float> scale(channels), bias(channels);
        for (uint32_t c = 0; c < channels; ++c)
        {
            scale[c] = gamma[c] / std::sqrt(variance[c] + 1e-5f);
            bias[c] = beta[c] - mean[c] * scale[c];
        }
        node["scale"] = vai::Tensor(std::vector<uint32_t>{ channels }, dtype).setFloats(std::move(scale));
        node["bias"] = vai::Tensor(std::vector<uint32_t>{ channels }, dtype).setFloats(std::move(bias));
    };

    auto loadNorm = [&](vai::node::LayerNorm& node, const std::string& key) {
        const auto dtype = node.wDefaultDtype();
        node["scale"] = vai::Tensor(ckpt[key + ".weight"], dtype);
        node["shift"] = vai::Tensor(ckpt[key + ".bias"], dtype);
    };

    auto loadLinear = [&](vai::node::Linear& node, const std::string& key) {
        const auto dtype = node.wDefaultDtype();
        node["weight"] = vai::Tensor(ckpt[key + ".weight"], dtype);
        node["bias"] = vai::Tensor(ckpt[key + ".bias"], dtype);
    };

    auto loadAttention = [&](group::MultiHeadAttention<QkvLayout::Separate>& layer, const std::string& key) {
        const auto dtype = layer.toQK.wDefaultDtype();
        layer.toQK["weight"] = vai::concat({ ckpt[key + ".q_proj.weight"], ckpt[key + ".k_proj.weight"] }, dtype);
        if (ckpt.contains(key + ".q_proj.bias"))
            layer.toQK["bias"] = vai::concat({ ckpt[key + ".q_proj.bias"], ckpt[key + ".k_proj.bias"] }, dtype);
        loadLinear(layer.toV, key + ".v_proj");
        loadLinear(layer.toOut, key + ".out_proj");
    };

    auto loadCSP = [&](group::CSPLayer& layer, const std::string& key) {
        loadConv(layer.conv1, key + ".conv1.conv", key + ".conv1.norm");
        loadConv(layer.conv2, key + ".conv2.conv", key + ".conv2.norm");
        loadRepVGG(layer.repVgg0, ckpt, key + ".bottlenecks.0");
        loadRepVGG(layer.repVgg1, ckpt, key + ".bottlenecks.1");
        loadRepVGG(layer.repVgg2, ckpt, key + ".bottlenecks.2");
        if (layer.optionConv)
            loadConv(*layer.optionConv, key + ".conv3.conv", key + ".conv3.norm");
    };

    // ResNet-vd stem, residual branches, and projected shortcuts.
    auto& backbone = model.backbone;
    const std::string root = "model.backbone.model";
    loadConv(backbone.stem0, root + ".embedder.embedder.0.convolution", root + ".embedder.embedder.0.normalization");
    loadConv(backbone.stem1, root + ".embedder.embedder.1.convolution", root + ".embedder.embedder.1.normalization");
    loadConv(backbone.stem2, root + ".embedder.embedder.2.convolution", root + ".embedder.embedder.2.normalization");
    size_t blockIndex = 0;
    for (uint32_t stage = 0; stage < std::size(model.cfg.backboneCfg.stageDepths); ++stage)
    {
        for (uint32_t i = 0; i < model.cfg.backboneCfg.stageDepths[stage]; ++i)
        {
            auto key = root + ".encoder.stages." + std::to_string(stage) + ".layers." + std::to_string(i);
            auto& block = backbone.blocks[blockIndex++];
            for (size_t j = 0; j < block.main.size(); ++j)
            {
                auto prefix = key + ".layer." + std::to_string(j);
                loadConv(block.main[j], prefix + ".convolution", prefix + ".normalization");
            }
            if (block.shortcut)
            {
                auto prefix = key + ".shortcut" + (block.avgPool ? ".1" : "");
                loadConv(*block.shortcut, prefix + ".convolution", prefix + ".normalization");
            }
        }
    }

    // Hybrid encoder input projections, coarse-scale attention, and FPN/PAN blocks.
    auto& encoder = model.encoder;
    loadConv(encoder.proj0, "model.encoder_input_proj.0.0", "model.encoder_input_proj.0.1");
    loadConv(encoder.proj1, "model.encoder_input_proj.1.0", "model.encoder_input_proj.1.1");
    loadConv(encoder.proj2, "model.encoder_input_proj.2.0", "model.encoder_input_proj.2.1");
    {
        auto& layer = encoder.encoder;
        const std::string key = "model.encoder.encoder.0.layers.0";
        loadAttention(layer.attention, key + ".self_attn");
        loadLinear(layer.fc1, key + ".fc1");
        loadLinear(layer.fc2, key + ".fc2");
        loadNorm(layer.norm1, key + ".self_attn_layer_norm");
        loadNorm(layer.norm2, key + ".final_layer_norm");
    }
    loadConv(encoder.fpnConv0, "model.encoder.lateral_convs.0.conv", "model.encoder.lateral_convs.0.norm");
    loadConv(encoder.panConv0, "model.encoder.downsample_convs.0.conv", "model.encoder.downsample_convs.0.norm");
    loadCSP(encoder.fpn0, "model.encoder.fpn_blocks.0");
    loadCSP(encoder.pan0, "model.encoder.pan_blocks.0");
    loadConv(encoder.fpnConv1, "model.encoder.lateral_convs.1.conv", "model.encoder.lateral_convs.1.norm");
    loadConv(encoder.panConv1, "model.encoder.downsample_convs.1.conv", "model.encoder.downsample_convs.1.norm");
    loadCSP(encoder.fpn1, "model.encoder.fpn_blocks.1");
    loadCSP(encoder.pan1, "model.encoder.pan_blocks.1");

    // Model-level pyramid projections adapt the hybrid encoder outputs for the decoder.
    loadConv(model.proj0, "model.decoder_input_proj.0.0", "model.decoder_input_proj.0.1");
    loadConv(model.proj1, "model.decoder_input_proj.1.0", "model.decoder_input_proj.1.1");
    loadConv(model.proj2, "model.decoder_input_proj.2.0", "model.decoder_input_proj.2.1");
    // Encoder heads used for initial query selection.
    auto& querySelection = model.querySelection;
    loadLinear(querySelection.proj, "model.enc_output.0");
    loadNorm(querySelection.norm, "model.enc_output.1");
    loadLinear(querySelection.classHead, "model.enc_score_head");
    loadLinear(querySelection.bbox0, "model.enc_bbox_head.layers.0");
    loadLinear(querySelection.bbox1, "model.enc_bbox_head.layers.1");
    loadLinear(querySelection.bbox2, "model.enc_bbox_head.layers.2");
    // Position heads use the same checkpoint keys; box heads are specific to each stage.
    auto& decoder = model.decoder;
    auto& values = decoder.valueProjection;
    loadLinear(values, "model.decoder.layers.0.encoder_attn.value_proj");
    for (size_t i = 1; i < decoder.blocks.size(); ++i)
    {
        const auto key = "model.decoder.layers." + std::to_string(i) + ".encoder_attn.value_proj";
        values["weight"] = values["weight"].concat(vai::Tensor(ckpt[key + ".weight"], values.wDefaultDtype()));
        if (values.hasBias())
            values["bias"] = values["bias"].concat(vai::Tensor(ckpt[key + ".bias"], values.wDefaultDtype()));
    }
    loadLinear(decoder.blocks.front().pos0, "model.decoder.query_pos_head.layers.0");
    loadLinear(decoder.blocks.front().pos1, "model.decoder.query_pos_head.layers.1");
    for (size_t i = 0; i < decoder.blocks.size(); ++i)
    {
        auto& block = decoder.blocks[i];
        auto& layer = block.layer;
        auto key = "model.decoder.layers." + std::to_string(i);
        // The checkpoint has one shared position head for every decoder layer.
        for (const char* parameter : { "weight", "bias" })
        {
            block.pos0[parameter] = decoder.blocks.front().pos0[parameter];
            block.pos1[parameter] = decoder.blocks.front().pos1[parameter];
        }
        loadAttention(layer.self, key + ".self_attn");
        loadLinear(layer.cross.offsets, key + ".encoder_attn.sampling_offsets");
        loadLinear(layer.cross.logits, key + ".encoder_attn.attention_weights");
        loadLinear(layer.cross.out, key + ".encoder_attn.output_proj");
        loadLinear(layer.fc1, key + ".fc1");
        loadLinear(layer.fc2, key + ".fc2");
        auto bboxKey = "model.decoder.bbox_embed." + std::to_string(i);
        loadLinear(block.bbox0, bboxKey + ".layers.0");
        loadLinear(block.bbox1, bboxKey + ".layers.1");
        loadLinear(block.bbox2, bboxKey + ".layers.2");
        loadNorm(layer.norm1, key + ".self_attn_layer_norm");
        loadNorm(layer.norm2, key + ".encoder_attn_layer_norm");
        loadNorm(layer.norm3, key + ".final_layer_norm");
        if (block.classHead)
            loadLinear(*block.classHead, "model.decoder.class_embed." + std::to_string(i));
    }
}

}   // namespace model::experimental::rtdetrv2::pekingu

// loadWeights<"pekingu">(model) - composes the path in Checkpoint and loads it.
template <>
inline void loadWeights<"pekingu", model::experimental::rtdetrv2::Model>(model::experimental::rtdetrv2::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/rtdetrv2/ckpts/") + model.cfg.name;
    model::experimental::rtdetrv2::pekingu::loadWeights(model, vai::Checkpoint(dir));
}

#endif   // EXPERIMENTAL_RTDETRV2_PEKINGU_WEIGHTS_H
