/*=============================================================================
  Written by   phg9296@vaistrata.com
  Task         RT-DETRv2 — detects objects in one image and reports timings.
  Assets       data/ckpts/rtdetrv2/pekingu/small/model.safetensors
                 from  huggingface.co/PekingU/rtdetr_v2_r18vd
               data/images/dog.jpg
                 ships with the repo
=============================================================================*/

#include "net.h"
#include "pekingu-weights.h"
#include "processing.h"
#include "vai/utils/image-processor.h"
#include <cstdio>
#include <string>

namespace rtdetrv2 = model::experimental::rtdetrv2;

namespace {

// Group matching operations by stage and input shape, as in the YOLO demos.
void setProfileTags(rtdetrv2::Net& net)
{
    auto tagConv = [](vai::node::Convolution& node, const std::string& stage, uint32_t spatial) {
        const auto& cfg = node.config_;
        node.setProfileTag(
            stage + ".Conv" + std::to_string(cfg.kernelSize) + "x" + std::to_string(cfg.kernelSize) + "s" +
            std::to_string(cfg.stride) + ".[" + std::to_string(cfg.inChannels) + "x" + std::to_string(cfg.outChannels) +
            "@" + std::to_string(spatial) + "]");
    };
    auto tagLinear = [](vai::node::Linear& node, const std::string& stage, uint32_t tokens) {
        node.setProfileTag(
            stage + ".Linear.[" + std::to_string(tokens) + "x" + std::to_string(node.config_.inDim) + "x" +
            std::to_string(node.config_.outDim) + "]");
    };
    auto tagAttention =
        [&](rtdetrv2::group::MultiHeadAttention<rtdetrv2::QkvLayout::Separate>& attention,
            const std::string& stage,
            uint32_t tokens) {
            tagLinear(attention.toQK, stage, tokens);
            tagLinear(attention.toV, stage, tokens);
            tagLinear(attention.toOut, stage, tokens);
            attention.attn.setProfileTag(
                stage + ".Attention.[" + std::to_string(tokens) + "x" + std::to_string(attention.D) + "]");
            if (attention.posAdd)
                attention.posAdd->setProfileTag(stage + ".PositionAdd");
        };
    auto tagCsp = [&](rtdetrv2::group::CSPLayer& csp, uint32_t spatial) {
        tagConv(csp.conv1, "Encoder", spatial);
        tagConv(csp.conv2, "Encoder", spatial);
        tagConv(csp.repVgg0, "Encoder", spatial);
        tagConv(csp.repVgg1, "Encoder", spatial);
        tagConv(csp.repVgg2, "Encoder", spatial);
        if (csp.optionConv)
            tagConv(*csp.optionConv, "Encoder", spatial);
    };

    auto& backbone = net.backbone;
    uint32_t spatial = 640;
    tagConv(backbone.stem0, "Backbone", spatial);
    spatial /= backbone.stem0.config_.stride;
    tagConv(backbone.stem1, "Backbone", spatial);
    spatial /= backbone.stem1.config_.stride;
    tagConv(backbone.stem2, "Backbone", spatial);
    spatial /= backbone.stem2.config_.stride;
    backbone.maxPool.setProfileTag("Backbone.MaxPool");
    spatial /= 2;
    for (auto& block : backbone.blocks)
    {
        uint32_t shortcutSpatial = spatial;
        for (auto& conv : block.main)
        {
            tagConv(conv, "Backbone", spatial);
            spatial /= conv.config_.stride;
        }
        if (auto& avgPool = block.avgPool; avgPool.has_value())
        {
            avgPool->setProfileTag("Backbone.AveragePooling");
            shortcutSpatial /= 2;
        }
        if (auto& shortcut = block.shortcut; shortcut.has_value())
            tagConv(*shortcut, "Backbone", shortcutSpatial);
    }

    auto& encoder = net.encoder;
    tagConv(encoder.proj0, "Encoder", 80);
    tagConv(encoder.proj1, "Encoder", 40);
    tagConv(encoder.proj2, "Encoder", 20);
    encoder.toHWC.setProfileTag("Encoder.Permute");
    encoder.toCHW.setProfileTag("Encoder.Permute");
    auto& layer = encoder.encoder;
    tagAttention(layer.attention, "Encoder", 400);
    tagLinear(layer.fc1, "Encoder", 400);
    tagLinear(layer.fc2, "Encoder", 400);
    layer.norm1.setProfileTag("Encoder.LayerNorm");
    layer.norm2.setProfileTag("Encoder.LayerNorm");
    layer.sum1.setProfileTag("Encoder.Add");
    layer.sum2.setProfileTag("Encoder.Add");
    layer.gelu.setProfileTag("Encoder.GELU");
    tagConv(encoder.fpnConv0, "Encoder", 20);
    tagConv(encoder.panConv0, "Encoder", 80);
    tagCsp(encoder.fpn0, 40);
    tagCsp(encoder.pan0, 40);
    encoder.upsample0.setProfileTag("Encoder.Upsample");
    encoder.fpnConcat0.setProfileTag("Encoder.Concat");
    encoder.panConcat0.setProfileTag("Encoder.Concat");
    tagConv(encoder.fpnConv1, "Encoder", 40);
    tagConv(encoder.panConv1, "Encoder", 40);
    tagCsp(encoder.fpn1, 80);
    tagCsp(encoder.pan1, 20);
    encoder.upsample1.setProfileTag("Encoder.Upsample");
    encoder.fpnConcat1.setProfileTag("Encoder.Concat");
    encoder.panConcat1.setProfileTag("Encoder.Concat");

    tagConv(net.proj0, "DecoderInput", 80);
    net.tokenPermute0.setProfileTag("DecoderInput.Permute");
    tagConv(net.proj1, "DecoderInput", 40);
    net.tokenPermute1.setProfileTag("DecoderInput.Permute");
    tagConv(net.proj2, "DecoderInput", 20);
    net.tokenPermute2.setProfileTag("DecoderInput.Permute");
    net.memory.setProfileTag("DecoderInput.Concat");

    auto& querySelection = net.querySelection;
    tagLinear(querySelection.proj, "QuerySelection", 8400);
    tagLinear(querySelection.classHead, "QuerySelection", 8400);
    querySelection.norm.setProfileTag("QuerySelection.LayerNorm");
    tagLinear(querySelection.bbox0, "QuerySelection", 300);
    tagLinear(querySelection.bbox1, "QuerySelection", 300);
    tagLinear(querySelection.bbox2, "QuerySelection", 300);
    querySelection.anchorAdd.setProfileTag("QuerySelection.AnchorAdd");
    querySelection.queryScore.setProfileTag("QuerySelection.Score");
    querySelection.topk.setProfileTag("QuerySelection.TopK");
    querySelection.querySelect.setProfileTag("QuerySelection.GatherMemory");
    querySelection.anchorSelect.setProfileTag("QuerySelection.GatherAnchors");

    auto& decoder = net.decoder;
    tagLinear(decoder.valueProjection, "Decoder.CrossAttention.Values", 8400);
    decoder.initBoxes.setProfileTag("Decoder.InitBoxes.Sigmoid");
    for (auto& block : decoder.blocks)
    {
        auto& layer = block.layer;
        tagLinear(block.pos0, "Decoder.Position", 300);
        tagLinear(block.pos1, "Decoder.Position", 300);
        tagAttention(layer.self, "Decoder.SelfAttention", 300);
        auto& cross = layer.cross;
        cross.queryPosAdd.setProfileTag("Decoder.CrossAttention.PositionAdd");
        tagLinear(cross.offsets, "Decoder.CrossAttention", 300);
        tagLinear(cross.logits, "Decoder.CrossAttention", 300);
        tagLinear(cross.out, "Decoder.CrossAttention", 300);
        cross.attnCore.setProfileTag("Decoder.CrossAttention.DeformableAttention");
        layer.sum1.setProfileTag("Decoder.Add");
        layer.sum2.setProfileTag("Decoder.Add");
        layer.sum3.setProfileTag("Decoder.Add");
        layer.norm1.setProfileTag("Decoder.LayerNorm");
        layer.norm2.setProfileTag("Decoder.LayerNorm");
        layer.norm3.setProfileTag("Decoder.LayerNorm");
        tagLinear(layer.fc1, "Decoder.FFN", 300);
        tagLinear(layer.fc2, "Decoder.FFN", 300);
        tagLinear(block.bbox0, "Decoder.Box", 300);
        tagLinear(block.bbox1, "Decoder.Box", 300);
        tagLinear(block.bbox2, "Decoder.Box", 300);
        block.refine.setProfileTag("Decoder.Box.Refine");
        if (block.classHead)
            tagLinear(*block.classHead, "Decoder.ClassHead", 300);
    }
}

}   // namespace

int main()
{
    constexpr auto modelCfg = rtdetrv2::ModelConfig::small();
    constexpr vai::Dtype dtype = vai::Dtype::Fp32;
    constexpr float threshold = 0.5f;
    const std::string imagePath = VAI_DATA_DIR "/images/dog.jpg";

    auto device = vai::getVaiDevice();
    rtdetrv2::Net net({ .device = device, .wDefaultDtype = dtype, .oDefaultDtype = dtype }, modelCfg);
    loadWeights<"pekingu">(net);
    setProfileTags(net);

    // Pillow's antialias squashes the image to 640x640 in [0, 1].
    vai::utils::ImageProcessor image(imagePath);
    vai::Tensor input({ 3, 640, 640 }, dtype);
    input.setFloats(image.resize(640, vai::utils::Resize::BilinearAntialias).normalize(0.0f, 1.0f).chw());

    vai::Tensor logits, boxes;
    for (uint32_t i = 0; i < 5u; ++i)
        net({ { "in0", input } }, { { "pred_logits", &logits }, { "pred_boxes", &boxes } });
    net.resetProfileRecords();

    for (uint32_t i = 0; i < 10u; ++i)
        net({ { "in0", input } }, { { "pred_logits", &logits }, { "pred_boxes", &boxes } });
    net.printProfile();

    auto detections = rtdetrv2::postprocess(
        logits.getData<float>(),
        boxes.getData<float>(),
        image.geometry().sourceWidth,
        image.geometry().sourceHeight,
        threshold);

    std::printf("\n=== Detections (%zu) ===\n", detections.size());
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto& detection = detections[i];
        std::printf(
            "[%zu] %s (%.2f%%): [%.1f, %.1f, %.1f, %.1f]\n",
            i,
            rtdetrv2::cocoClasses[detection.label],
            detection.score * 100.0f,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2);
    }
    return 0;
}
