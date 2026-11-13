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
        tagConv(csp.conv1, spatial);
        tagConv(csp.conv2, spatial);
        for (auto& block : csp.bottlenecks)
            tagBottleneck(block, spatial);
        tagConv(csp.conv3, spatial);
    };
    auto tagBranch = [&](yolox::group::Head::PredictionBranch& branch, uint32_t spatial) {
        tagConv(branch.stem, spatial);
        tagConv(branch.clsConv1, spatial);
        tagConv(branch.clsConv2, spatial);
        tagConv(branch.clsPred, spatial);
        tagConv(branch.regConv1, spatial);
        tagConv(branch.regConv2, spatial);
        tagConv(branch.regObjPred, spatial);
    };

    const uint32_t h = net.runtimeCfg.imageSize;
    auto& backbone = net.backbone;
    backbone.stem.conv.setProfileTag("FocusConv3x3s1.[3x" + std::to_string(backbone.stem.conv.config_.outChannels) +
                                     "@" + std::to_string(h / 2u) + "]");
    tagConv(backbone.dark2.conv, h / 2u);
    tagCsp(backbone.dark2.csp, h / 4u);
    tagConv(backbone.dark3.conv, h / 4u);
    tagCsp(backbone.dark3.csp, h / 8u);
    tagConv(backbone.dark4.conv, h / 8u);
    tagCsp(backbone.dark4.csp, h / 16u);
    tagConv(backbone.dark5.conv, h / 16u);
    tagConv(backbone.dark5.spp.conv1, h / 32u);
    backbone.dark5.spp.sppf.setProfileTag("SPPF");
    tagConv(backbone.dark5.spp.conv2, h / 32u);
    tagCsp(backbone.dark5.csp, h / 32u);

    auto& neck = net.neck;
    tagConv(neck.lateralConv0, h / 32u);
    tagCsp(neck.c3P4, h / 16u);
    tagConv(neck.reduceConv1, h / 16u);
    tagCsp(neck.c3P3, h / 8u);
    tagConv(neck.bottomUpConv2, h / 8u);
    tagCsp(neck.c3N3, h / 16u);
    tagConv(neck.bottomUpConv1, h / 16u);
    tagCsp(neck.c3N4, h / 32u);

    tagBranch(net.head.p3, h / 8u);
    tagBranch(net.head.p4, h / 16u);
    tagBranch(net.head.p5, h / 32u);
    net.detector.decode.setProfileTag("Decode");
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_YOLOX_PROFILE_TAGS_H
