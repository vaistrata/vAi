/*=============================================================================
  Written by   phg9296@vaistrata.com
  Task         YOLOX — detects objects in a batch of images and reports timings.
  Assets       data/images/dog.jpg
                 from  github.com/Megvii-BaseDetection/YOLOX/tree/main/assets
=============================================================================*/

#include "megvii-weights.h"
#include "net.h"
#include "processing.h"
#include "vai/utils/image-processor.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;
namespace yolox = model::experimental::yolox;

namespace {

void setProfileTags(yolox::Net& net)
{
    auto tagConv = [](vai::node::Convolution& conv, uint32_t spatial) {
        const auto& config = conv.config_;
        conv.setProfileTag("Conv" + std::to_string(config.kernelSize) + "x" + std::to_string(config.kernelSize) + "s" +
                           std::to_string(config.stride) + ".[" + std::to_string(config.inChannels) + "x" +
                           std::to_string(config.outChannels) + "@" + std::to_string(spatial) + "]");
    };
    auto tagBottleneck = [&](yolox::group::Bottleneck& block, uint32_t spatial) {
        tagConv(block.conv1, spatial);
        tagConv(block.conv2, spatial);
    };
    auto tagCsp = [&](yolox::group::CSPLayer& csp, uint32_t spatial) {
        tagConv(csp.conv12, spatial);
        csp.conv1.setProfileTag("conv1.RowSlice");
        csp.conv2.setProfileTag("conv2.RowSlice");
        for (auto& block : csp.bottlenecks)
            tagBottleneck(block, spatial);
        tagConv(csp.conv3, spatial);
    };

    const uint32_t h = net.runtimeCfg.imageSize;
    auto& backbone = net.backbone;
    backbone.stem.setProfileTag(
        "Focus.[3x" + std::to_string(backbone.stem.config_.outChannels) + "@" + std::to_string(h / 2u) + "]");
    tagConv(backbone.stage1.conv, h / 2u);
    tagCsp(backbone.stage1.csp, h / 4u);
    tagConv(backbone.stage2.conv, h / 4u);
    tagCsp(backbone.stage2.csp, h / 8u);
    tagConv(backbone.stage3.conv, h / 8u);
    tagCsp(backbone.stage3.csp, h / 16u);
    tagConv(backbone.stage4.conv, h / 16u);
    if (!backbone.stage4.spp.has_value())
        throw std::bad_optional_access();
    auto& spp = backbone.stage4.spp.value();
    tagConv(spp.conv1, h / 32u);
    spp.pool5.setProfileTag("SPP.Pool5x5");
    spp.pool9.setProfileTag("SPP.Pool9x9");
    spp.pool13.setProfileTag("SPP.Pool13x13");
    spp.concat.setProfileTag("SPP.Concat");
    tagConv(spp.conv2, h / 32u);
    tagCsp(backbone.stage4.csp, h / 32u);

    auto& neck = net.neck;
    tagConv(neck.fpnConv0, h / 32u);
    tagCsp(neck.fpn0, h / 16u);
    tagConv(neck.fpnConv1, h / 16u);
    tagCsp(neck.fpn1, h / 8u);
    tagConv(neck.panConv0, h / 8u);
    tagCsp(neck.pan0, h / 16u);
    tagConv(neck.panConv1, h / 16u);
    tagCsp(neck.pan1, h / 32u);

    auto& head = net.head;
    for (uint32_t i = 0; i < 3u; ++i)
    {
        const uint32_t spatial = h / (8u << i);
        tagConv(head.stem[i], spatial);
        tagConv(head.fusedConv[i], spatial);
        head.clsSlice[i].setProfileTag("clsConv1.RowSlice");
        head.regSlice[i].setProfileTag("regConv1.RowSlice");
        tagConv(head.clsConv2[i], spatial);
        tagConv(head.clsPred[i], spatial);
        tagConv(head.regConv2[i], spatial);
        tagConv(head.regObjPred[i], spatial);
    }
    net.detector.setProfileTag("Decode");
}

}   // namespace

int main(int argc, char** argv)
{
    constexpr yolox::Tier tier = yolox::Tier::LARGE;
    std::vector<std::string> paths(argv + 1, argv + argc);
    if (paths.empty())
        paths.emplace_back(VAI_DATA_DIR "/images/dog.jpg");

    const yolox::ModelConfig cfg = yolox::getConfigCOCO(tier);
    const yolox::RuntimeConfig runtimeCfg{
        .imageSize = 640u,
        .confThreshold = 0.25f,
        .iouThreshold = 0.45f,
    };
    yolox::Net net(
        {
            .wDefaultDtype = Dtype::Fp16,
        },
        cfg,
        runtimeCfg);

    std::printf("Loading YOLOX-%s weights...\n", yolox::tierName(tier));
    loadWeights<"megvii">(net);
    setProfileTags(net);

    const uint32_t sz = runtimeCfg.imageSize, batch = static_cast<uint32_t>(paths.size());
    std::vector<float> pixels;
    pixels.reserve(size_t(batch) * 3 * sz * sz);
    std::vector<utils::ImageGeometry> geometry;
    for (const auto& path : paths)
    {
        utils::ImageProcessor image(path);
        image.letterbox(sz, utils::Align::TopLeft, 114.0f).bgr().chw();
        const std::vector<float>& data = image;
        pixels.insert(pixels.end(), data.begin(), data.end());
        geometry.push_back(image.geometry());
    }
    Tensor input = Tensor({ batch, 3u, sz, sz }).setFloats(std::move(pixels));
    Tensor output;

    for (uint32_t i = 0; i < 5u; ++i)
        net({ { "in0", input } }, { { "out0", &output } });
    net.resetProfileRecords();

    for (uint32_t i = 0; i < 10u; ++i)
        net({ { "in0", input } }, { { "out0", &output } });
    net.printProfile();
    std::vector<float> raw(output.numElements());
    std::memcpy(raw.data(), output.getData(), raw.size() * sizeof(float));
    const auto detections =
        yolox::postprocessBatch(raw, geometry, runtimeCfg.confThreshold, runtimeCfg.iouThreshold, cfg.numClasses);
    for (uint32_t b = 0; b < batch; ++b)
    {
        std::printf("\n%s (%zu detections)\n", paths[b].c_str(), detections[b].size());
        for (const auto& d : detections[b])
            std::printf(
                "%s (%.2f%%): [%.1f, %.1f, %.1f, %.1f]\n",
                d.className.c_str(),
                d.confidence * 100.0f,
                d.x1,
                d.y1,
                d.x2,
                d.y2);
    }
    return 0;
}
