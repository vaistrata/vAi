/*=============================================================================
  Written by   qkrtmddn0507@vaistrata.com
  Task         Diffusion Policy (pushT) — turns two observation steps (camera image and
               agent position) into the next actions, profiles the encode and the
               denoise.
  Assets       data/images/cam_high.jpg, a stand-in for the pushT camera
                 from  huggingface.co/datasets/suz22/RoboTwin_clean
=============================================================================*/

#include "net.h"
#include "lerobot-weights.h"
#include "processing.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/time-checker.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
namespace dp = model::experimental::diffusion_policy;


/////////////////////////////////////////////////////////////////////////////////////////
// Tag nodes for profiling — group by kernel shape, so identical-shape ops across
// the encoder and the UNet collapse to one row per distinct kernel.
/////////////////////////////////////////////////////////////////////////////////////////
void setProfileTags(dp::Net& net)
{
    const auto& cfg = net.cfg;
    auto shape = [](uint32_t in, uint32_t out, uint32_t length) {
        return ".[" + std::to_string(in) + "x" + std::to_string(out) + "@" + std::to_string(length) + "]";
    };

    // Observation encoder — ResNet18, its GroupNorms, the keypoint head.
    {
        auto& enc = net.rgbEncoder;
        auto tagConvNorm = [&](vai::group::ConvNorm& cn, uint32_t spatial) {
            const auto& c = cn.conv.config_;
            cn.conv.setProfileTag(
                "Encoder.Conv" + std::to_string(c.kernelSize) + "s" + std::to_string(c.stride) +
                shape(c.inChannels, c.outChannels, spatial));
            if (cn.norm)
                cn.norm->setProfileTag("Encoder.GroupNorm");
        };

        uint32_t spatial = (enc.encodedSize + 1u) / 2u;   // after conv1
        tagConvNorm(enc.backbone.conv1, spatial);
        enc.backbone.maxPool.setProfileTag("Encoder.MaxPool");
        spatial = (spatial + 1u) / 2u;
        size_t index = 0;
        for (uint32_t l = 0; l < 4; ++l)
        {
            if (l > 0)
                spatial = (spatial + 1u) / 2u;
            for (uint32_t b = 0; b < enc.backbone.depths[l]; ++b)
            {
                auto& block = enc.backbone.blocks[index++];
                tagConvNorm(block.conv1, spatial);
                tagConvNorm(block.conv2, spatial);
                if (block.shortcut)
                    tagConvNorm(*block.shortcut, spatial);
            }
        }
        if (enc.crop)
        {
            enc.crop->cropRows.setProfileTag("Encoder.Crop");
            enc.crop->cropCols.setProfileTag("Encoder.Crop");
        }
        enc.keypointProj.setProfileTag("Encoder.KeypointProj");
        enc.spatialSoftmax.softmax.setProfileTag("Encoder.SpatialSoftmax");
        enc.spatialSoftmax.expectedXy.setProfileTag("Encoder.SpatialSoftmax");
        enc.out.setProfileTag("Encoder.Out");
        net.catFeatures.setProfileTag("Encoder.Concat");
    }

    // UNet — a residual block's shape is its channels and the horizon it runs at.
    {
        auto& unet = net.unet;
        const uint32_t condDim = cfg.stepEmbedDim + cfg.globalCondDim();
        auto tagConv1d = [&](dp::node::Convolution1d& conv, uint32_t length) {
            const auto& c = conv.config_;
            conv.setProfileTag(
                "UNet.Conv1d.k" + std::to_string(c.kernelSize) + "s" + std::to_string(c.stride) +
                shape(c.inChannels, c.outChannels, length));
        };
        auto tagBlock = [&](dp::group::Conv1dBlock& block, uint32_t length) {
            tagConv1d(block.conv, length);
            block.norm.setProfileTag(
                "UNet.GroupNormMish.[" + std::to_string(block.conv.config_.outChannels) + "@" + std::to_string(length) +
                "]");
        };
        auto tagRes = [&](dp::group::ConditionalResidualBlock1d& res, uint32_t length) {
            tagBlock(res.conv1, length);
            tagBlock(res.conv2, length);
            res.condEncoder.setProfileTag(
                "UNet.CondEncoder.Linear.[" + std::to_string(condDim) + "x" +
                std::to_string(res.condEncoder.config_.outDim) + "]");
            const std::string channels =
                "[" + std::to_string(res.conv2.conv.config_.outChannels) + "@" + std::to_string(length) + "]";
            res.film.affine.setProfileTag("UNet.Film." + channels);
            if (res.film.scale)
            {
                res.film.scale->setProfileTag("UNet.Film.Slice");
                res.film.shift->setProfileTag("UNet.Film.Slice");
            }
            res.residual.setProfileTag("UNet.Residual.Add." + channels);
            if (res.residualConv)
                tagConv1d(*res.residualConv, length);
        };

        const uint32_t L = uint32_t(cfg.downDims.size());
        for (uint32_t l = 0; l < L; ++l)
        {
            const uint32_t length = cfg.horizon >> l;
            tagRes(unet.downBlocks[2 * size_t(l)], length);
            tagRes(unet.downBlocks[2 * size_t(l) + 1], length);
            if (auto& downsample = unet.downsamples[l]; downsample.has_value())
                tagConv1d(*downsample, length);
        }
        for (auto& res : unet.midBlocks)
            tagRes(res, cfg.horizon >> (L - 1));
        for (uint32_t u = 0; u + 1 < L; ++u)
        {
            const uint32_t length = cfg.horizon >> (L - 1 - u);
            unet.upCats[u].setProfileTag("UNet.Concat");
            tagRes(unet.upBlocks[2 * size_t(u)], length);
            tagRes(unet.upBlocks[2 * size_t(u) + 1], length);
            const auto& c = unet.upsamples[u].config_;
            unet.upsamples[u].setProfileTag("UNet.ConvTranspose1d" + shape(c.inChannels, c.outChannels, length));
        }
        tagBlock(unet.finalBlock, cfg.horizon);
        tagConv1d(unet.finalConv, cfg.horizon);

        unet.stepEncoder.posEmb.setProfileTag("UNet.StepEncoder");
        unet.stepEncoder.fc1.setProfileTag("UNet.StepEncoder");
        unet.stepEncoder.mish.setProfileTag("UNet.StepEncoder");
        unet.stepEncoder.fc2.setProfileTag("UNet.StepEncoder");
        unet.catCond.setProfileTag("UNet.Cond");
        unet.condMish.setProfileTag("UNet.Cond");
        unet.toChannels.setProfileTag("UNet.Permute");
        unet.toTime.setProfileTag("UNet.Permute");
    }

    net.timestepTable.setProfileTag("Denoise.CyclicTableReader");
    net.scheduleTable.setProfileTag("Denoise.CyclicTableReader");
    net.ddpmStep.setProfileTag("Denoise.DdpmStep");
}


int main()
{
    // Construct model
    const dp::ModelConfig modelCfg = dp::ModelConfig::_0_3B();
    printf("Loading diffusion-policy-%s model...\n", modelCfg.variant);
    // clang-format off
    dp::Net net({
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg);
    // clang-format on

    // Configure the per-node GPU execution-time profiler
    setProfileTags(net);

    // Load weights
    printf("Loading weights...\n");
    {
        utils::TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"lerobot">(net);
    }

    // One frame per observation step, oldest first. The demo has one frame, so both
    // steps see it; a robot loop passes its last numObsSteps frames.
    const std::vector<std::string> imagePaths(modelCfg.numObsSteps, VAI_DATA_DIR "/images/cam_high.jpg");
    const uint32_t B = modelCfg.numObsSteps * modelCfg.numCameras;
    const uint32_t S = modelCfg.imageSize;

    // Preprocess the images and stack them into [B, 3, S, S]: [0, 1], then MEAN_STD.
    // The encoder center-crops them itself.
    std::vector<float> pixels;
    pixels.reserve(size_t(B) * 3 * S * S);
    for (uint32_t k = 0; k < B; ++k)
    {
        printf("Loading image %u: %s\n", k, imagePaths[k].c_str());
        // clang-format off
        std::vector<float> chw = utils::ImageProcessor(imagePaths[k])
            .resize(S)                                      // squashed to the policy's S x S
            .normalize(0.0f, 1.0f)
            .standardize(net.imageMean, net.imageStd)
            .chw();
        // clang-format on
        pixels.insert(pixels.end(), chw.begin(), chw.end());
    }

    // The state is the agent's position in the 512x512 pushT field, one per step.
    const std::vector<float> rawState = { 222.0f, 97.0f, 223.5f, 98.0f };
    const std::vector<float> state = dp::normalizeMinMax(rawState, net.stateMin, net.stateMax);

    // Initialize the input tensors
    Tensor inImages = Tensor({ B, 3, S, S }).setFloats(std::move(pixels));
    Tensor inState = Tensor({ modelCfg.numObsSteps, modelCfg.stateDim }).setFloats(state);

    // The diffusion noise, drawn here from a seed: the starting sample x_T and every
    // denoising step's z. decode() overwrites the sample, so each run starts from a
    // fresh copy of x_T.
    const uint64_t seed = 0;
    const uint32_t steps = net.runtimeCfg.denoiseSteps;
    const std::vector<float> initNoise = dp::sampleNoise(size_t(modelCfg.horizon) * modelCfg.actionDim, seed);
    Tensor stepNoise = Tensor({ steps, modelCfg.horizon, modelCfg.actionDim }, Dtype::Fp32);
    stepNoise.setFloats(dp::sampleNoise(stepNoise.numElements(), seed + 1));
    auto initSample = [&] {
        return Tensor({ modelCfg.horizon, modelCfg.actionDim }, Dtype::Fp32).setFloats(initNoise);
    };
    Tensor sample;

    // Warm-up.
    {
        net.encode(inImages, inState);
        sample = initSample();
        net.decode(stepNoise, &sample);
    }

    net.resetProfileRecords();

    // Encode — every observation step's image and state into the global condition.
    {
        utils::TimeChecker totalTimer(
            "\n"
            "────────────────────────────────────────\n"
            "▶ Encode {} images + {} states in {}ms \n"
            "────────────────────────────────────────",
            B,
            modelCfg.numObsSteps);

        net.encode(inImages, inState);

        net.sync();   // this sync is not necessary, but it is here to measure the encode time.
    }

    net.printProfile();

    // Denoise — DDPM from noise at the last timestep down to the action sample at 0.
    sample = initSample();
    {
        utils::TimeChecker totalTimer(
            "\n"
            "────────────────────────────────────────\n"
            "▶ Denoise {} steps over a {}-step sample in {}ms \n"
            "────────────────────────────────────────",
            steps,
            modelCfg.horizon);

        net.decode(stepNoise, &sample);

        net.sync();   // this sync is not necessary, but it is here to measure the denoise time.
    }

    net.printProfile();

    // The executed actions — the window from the current step, in pushT's pixel units.
    const std::vector<float> actions = dp::unnormalizeMinMax(
        dp::actionWindow(sample.getData<float>(), modelCfg.actionDim, modelCfg.actionStart(), modelCfg.numActionSteps),
        net.actionMin,
        net.actionMax);

    printf("\n=== Actions (%u steps x %u dims) ===\n", modelCfg.numActionSteps, modelCfg.actionDim);
    for (uint32_t s = 0; s < modelCfg.numActionSteps; ++s)
    {
        printf("  t%-2u ", s);
        for (uint32_t d = 0; d < modelCfg.actionDim; ++d)
            printf("% .4f ", actions[size_t(s) * modelCfg.actionDim + d]);
        printf("\n");
    }

    printf("\n--- Actions complete ---\n");

    return 0;
}
