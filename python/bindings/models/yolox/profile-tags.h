#ifndef PYVAI_MODELS_YOLOX_PROFILE_TAGS_H
#define PYVAI_MODELS_YOLOX_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from models/experimental/yolox/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/experimental/yolox/net.h"
#include <cstdint>
#include <string>

namespace pyvai {
namespace yolox = ::model::experimental::yolox;   // the bindings' short name for it
}

namespace pyvai::examples {
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
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_YOLOX_PROFILE_TAGS_H
