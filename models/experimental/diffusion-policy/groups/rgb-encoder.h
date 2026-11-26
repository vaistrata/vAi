#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_RGB_ENCODER_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_RGB_ENCODER_H

#include "vai/groups/resnet.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/reshape.h"
#include "center-crop.h"
#include "spatial-softmax.h"
#include <cstdint>
#include <optional>
#include <vector>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// RgbEncoder - lerobot's DiffusionRgbEncoder: one image to one feature vector
// Input:  in0  [3, S, S],  or [B, 3, S, S] with allowBatch
// Output: out0 [2 * K],    or [B, 2 * K]    with allowBatch; K = numKeypoints
//
// out0 = relu(out(flatten(spatialSoftmax(keypointProj(backbone(crop(in0)))))))
//
// backbone is torchvision's ResNet18 without avgpool / fc, its norm as ResNetConfig
// says: lerobot's use_group_norm swaps every BatchNorm2d for GroupNorm(C / 16, C),
// which is ResNetNorm::GroupNorm with groupNormWidth 16. keypointProj is the
// SpatialSoftmax's num_kp projection (pool.nets), a 1x1 conv 512 -> K.
//
// cropSize > 0 adds crop, lerobot's crop_shape: its eval-time center crop (training's
// random crop has no place in inference); 0 feeds the S x S image as it is.
// allowBatch adds a leading B, e.g. n_obs_steps x cameras folded together as lerobot
// does, each image encoded on its own.
/////////////////////////////////////////////////////////////////////////////////////////
struct RgbEncoder : public vai::NodeGroup
{
    uint32_t encodedSize;   // the side the backbone sees: cropSize, or S without a crop
    uint32_t featureSize;   // side of the backbone's last feature map
    uint32_t featureDim;    // 2 * K

    vai::node::Identity input;           // [B, 3, S, S]
    std::optional<CenterCrop> crop;      //              -> [B, 3, c, c]     (crop_shape)
    vai::group::ResNet18 backbone;       //              -> [B, 512, f, f]
    vai::node::Convolution keypointProj; //              -> [B, K, f, f]     (pool.nets)
    SpatialSoftmax spatialSoftmax;       //              -> [B, K, 2]        (pool)
    vai::node::Reshape flatten;          //              -> [B, 2 * K]
    vai::node::Linear out;               //              -> [B, 2 * K], + relu (out, relu)

    struct Config {
        vai::ResNetConfig backbone;
        uint32_t imageSize;      // S, the input side
        uint32_t cropSize = 0;   // the center crop's side; 0: no crop
        uint32_t numKeypoints;   // K
        bool allowBatch = false;   // true: a leading batch axis, kept the whole way
    };

    // The side of ResNet's last feature map: conv1, the max pool and layer2 .. layer4
    // each halve it, rounding up (k7 / k3 at stride 2, half padding), e.g. 84 -> 3.
    static constexpr uint32_t resnetFeatureSize(uint32_t imageSize)
    {
        uint32_t s = imageSize;
        for (uint32_t i = 0; i < 5; ++i)
            s = (s + 1u) / 2u;
        return s;
    }

    RgbEncoder(Config config)
    : encodedSize(config.cropSize > 0 ? config.cropSize : config.imageSize)
    , featureSize(resnetFeatureSize(encodedSize))
    , featureDim(2 * config.numKeypoints)
    , backbone(config.backbone)
    , keypointProj({
          .inChannels = vai::group::ResNet18::outChannels,
          .outChannels = config.numKeypoints,
          .kernelSize = 1,
          .stride = 1,
      })
    , spatialSoftmax({
          .channels = config.numKeypoints,
          .height = featureSize,
          .width = featureSize,
          .allowBatch = config.allowBatch,
      })
    , flatten(config.allowBatch ? std::vector<int64_t>{ 0, featureDim } : std::vector<int64_t>{ featureDim })
    , out({ .inDim = featureDim, .outDim = featureDim, .actType = vai::ActType::Relu })
    {
        VAI_ASSERT(encodedSize >= 32u && config.numKeypoints >= 1u);

        if (config.cropSize > 0)
            crop.emplace(CenterCrop::Config{
                .imageSize = config.imageSize,
                .cropSize = config.cropSize,
                .allowBatch = config.allowBatch,
            });

        // clang-format off
        input - crop - backbone - keypointProj - spatialSoftmax - flatten - out;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", out.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_RGB_ENCODER_H
