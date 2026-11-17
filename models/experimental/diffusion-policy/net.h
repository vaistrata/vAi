/*=============================================================================
  Written by   qkrtmddn0507@vaistrata.com
  Model        diffusion policy (pushT)
=============================================================================*/

#ifndef EXPERIMENTAL_DIFFUSION_POLICY_NET_H
#define EXPERIMENTAL_DIFFUSION_POLICY_NET_H

#include "ddpm-scheduler.h"
#include "processing.h"
#include "groups/conditional-unet-1d.h"
#include "groups/rgb-encoder.h"
#include "nodes/ddpm-step.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/cyclic-table-reader.h"
#include "vai/nodes/reshape.h"
#include <array>
#include <vector>


namespace model::experimental::diffusion_policy {

struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/ckpts/diffusion-policy/lerobot/

    // Observation encoder
    vai::ResNetConfig backbone;
    uint32_t imageSize;      // the square camera image
    uint32_t cropSize;       // lerobot crop_shape, center-cropped; 0: no crop
    uint32_t numKeypoints;   // spatial_softmax_num_keypoints
    uint32_t numCameras;     // image features, one shared encoder over all of them
    uint32_t stateDim;       // observation.state
    uint32_t numObsSteps;    // n_obs_steps

    // Action denoiser
    uint32_t actionDim;               // action
    uint32_t horizon;                 // actions one sample carries
    uint32_t numActionSteps;          // n_action_steps: the ones executed
    std::vector<uint32_t> downDims;   // the UNet's channels per level
    uint32_t kernelSize;
    uint32_t numGroups;               // n_groups
    uint32_t stepEmbedDim;            // diffusion_step_embed_dim
    bool filmScale;                   // use_film_scale_modulation
    DdpmConfig scheduler;

    // One image's feature: 2 coordinates per keypoint.
    uint32_t imageFeatureDim() const { return 2 * numKeypoints; }

    // One observation step's features: its state, then every camera's image feature.
    uint32_t stepFeatureDim() const { return stateDim + numCameras * imageFeatureDim(); }

    // global_cond: every observation step's features, step by step.
    uint32_t globalCondDim() const { return numObsSteps * stepFeatureDim(); }

    // The executed actions start at the current observation's step in the sample.
    uint32_t actionStart() const { return numObsSteps - 1; }
};

struct RuntimeConfig
{
    uint32_t denoiseSteps = 100;   // num_inference_steps; lerobot's None is the training count
};

// lerobot/diffusion_pusht: GroupNorm ResNet18 on 84x84 crops of 96x96 images, 32 keypoints,
// one camera, a 2-D agent position, two observation steps; a 3-level UNet denoising a
// 16-step 2-D action sample with 100-step squaredcos DDPM, 8 actions executed.
inline ModelConfig getConfigPushT()
{
    return {
        .variant = "pusht",
        .backbone = { .norm = vai::ResNetNorm::GroupNorm, .groupNormWidth = 16 },
        .imageSize = 96u,
        .cropSize = 84u,
        .numKeypoints = 32u,
        .numCameras = 1u,
        .stateDim = 2u,
        .numObsSteps = 2u,
        .actionDim = 2u,
        .horizon = 16u,
        .numActionSteps = 8u,
        .downDims = { 512u, 1024u, 2048u },
        .kernelSize = 5u,
        .numGroups = 8u,
        .stepEmbedDim = 128u,
        .filmScale = true,
        .scheduler = {},
    };
}


/////////////////////////////////////////////////////////////////////////////////////////
// Net - Diffusion Policy: observation encoder, then a DDPM loop over a conditional UNet
//
// encode() - lerobot's _prepare_global_conditioning
//   Input : inImages [numObsSteps * numCameras, 3, imageSize, imageSize], step-major,
//                    normalized; the encoder center-crops them to cropSize
//           inState  [numObsSteps, stateDim] Fp32, normalized
//   Output: outGlobalCond [1, globalCondDim], e.g. [1, 132] for pushT = 2 * (2 + 64)
//
// decode() - lerobot's conditional_sample, one forward per denoising step
//   Input : inActions    [horizon, actionDim] Fp32, x_t (noise before the first step)
//           inGlobalCond [1, globalCondDim], encode()'s output, kept on the GPU
//           inStepNoise  [denoiseSteps, horizon, actionDim] Fp32, each step's z
//   Output: outActions   [horizon, actionDim] Fp32, x_{t-1}, into the same tensor
//
// As pi0.5 runs its flow, the loop re-runs one recording: the timestep and the DDPM
// coefficients a step needs come from two CyclicTableReaders, one row per forward,
// rewound by decode(); the sample goes back in as the next step's input. DDPM's
// update is not a residual Linear (x0 is clipped, noise is added), so DdpmStep
// applies it after the UNet. Its noise is drawn on the host per inference
// (drawNoise(seed)), so a run is reproducible from a seed, though not torch's stream.
/////////////////////////////////////////////////////////////////////////////////////////
struct Net : public vai::NeuralNet
{
    ModelConfig cfg;
    RuntimeConfig runtimeCfg;

    // Observation encoder
    group::RgbEncoder rgbEncoder;      // [S*N, 3, H, W] -> [S*N, F], F = imageFeatureDim
    vai::node::Reshape perStep;        //                -> [S, N*F]
    vai::node::Concat catFeatures;     // [S, stateDim] + [S, N*F] -> [S, stepFeatureDim]
    vai::node::Reshape flattenCond;    //                -> [1, globalCondDim]

    // Action denoiser
    vai::node::CyclicTableReader timestepTable;   // [steps, 1] -> [1], the step's timestep
    vai::node::CyclicTableReader scheduleTable;   // [steps, 6] -> [6], the step's DDPM row
    group::ConditionalUnet1d unet;                // [horizon, A] -> [horizon, A], ε
    node::DdpmStep ddpmStep;                      // x_t, ε -> x_{t-1}

    vai::Tensor globalCond;   // encode()'s output, decode()'s condition
    vai::Tensor stepNoise;    // [steps, horizon, A], drawNoise() refills it

    // Filled by the loader: the statistics the caller normalizes with. pushT maps the
    // images MEAN_STD, the state and the action MIN_MAX (to [-1, 1]).
    std::array<float, 3> imageMean{};   // per channel
    std::array<float, 3> imageStd{};    // + 1e-8, as lerobot divides by it
    std::vector<float> stateMin;        // [stateDim]
    std::vector<float> stateMax;
    std::vector<float> actionMin;       // [actionDim]
    std::vector<float> actionMax;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& cfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , cfg(cfg)
    , runtimeCfg(runtimeCfg)
    , rgbEncoder({
          .backbone = cfg.backbone,
          .imageSize = cfg.imageSize,
          .cropSize = cfg.cropSize,
          .numKeypoints = cfg.numKeypoints,
          .allowBatch = true,
      })
    , perStep({ int64_t(cfg.numObsSteps), int64_t(cfg.numCameras * cfg.imageFeatureDim()) })
    , catFeatures({ .axis = -1, .inputCount = 2 })
    , flattenCond({ 1, int64_t(cfg.globalCondDim()) })
    , timestepTable({ .numRows = runtimeCfg.denoiseSteps })
    , scheduleTable({ .numRows = runtimeCfg.denoiseSteps })
    , unet({
          .actionDim = cfg.actionDim,
          .globalCondDim = cfg.globalCondDim(),
          .downDims = cfg.downDims,
          .kernelSize = cfg.kernelSize,
          .groups = cfg.numGroups,
          .stepEmbedDim = cfg.stepEmbedDim,
          .filmScale = cfg.filmScale,
      })
    , ddpmStep({
          .predictionType = cfg.scheduler.predictionType,
          .clipSample = cfg.scheduler.clipSample,
          .clipSampleRange = cfg.scheduler.clipSampleRange,
      })
    , stepNoise({ runtimeCfg.denoiseSteps, cfg.horizon, cfg.actionDim }, vai::Dtype::Fp32)
    {
        VAI_ASSERT(cfg.numCameras >= 1u && cfg.stateDim >= 1u && cfg.numObsSteps >= 1u);
        VAI_ASSERT(cfg.actionStart() + cfg.numActionSteps <= cfg.horizon);
        // The UNet halves the horizon once per level below the top, and doubles it back.
        VAI_ASSERT(cfg.horizon % (1u << (cfg.downDims.size() - 1u)) == 0u);

        // clang-format off
        input("inState") -                                   "in0"/catFeatures;
        input("inImages") - rgbEncoder - perStep -           "in1"/catFeatures;
        catFeatures - flattenCond - output("outGlobalCond");

        input("inActions") -                      "in0"/unet;
        input("inGlobalCond") -           "globalCond"/unet;
        input("inActions") - timestepTable - "timestep"/unet;
        input("inActions") -                  "in0"/ddpmStep;
        unet -                        "modelOutput"/ddpmStep;
        input("inActions") - scheduleTable - "schedule"/ddpmStep;
        input("inStepNoise") -              "noise"/ddpmStep;
        ddpmStep - output("outActions");
        // clang-format on

        const DdpmSchedule schedule = DdpmSchedule::make(cfg.scheduler, runtimeCfg.denoiseSteps);
        timestepTable["table"] = vai::Tensor({ runtimeCfg.denoiseSteps, 1u }, vai::Dtype::Fp32)
                                     .setFloats(schedule.timesteps);
        scheduleTable["table"] = vai::Tensor({ runtimeCfg.denoiseSteps, DdpmSchedule::rowWidth }, vai::Dtype::Fp32)
                                     .setFloats(schedule.rows);

        // Concat joins one dtype: the image features come out in the state's Fp32,
        // the step embedding in the global condition's.
        rgbEncoder.out.setODtype(vai::Dtype::Fp32, dtypeKey());
        unet.stepEncoder.fc2.setODtype(vai::Dtype::Fp32, dtypeKey());
        // The schedule and the sample stay Fp32 whatever the activations are.
        timestepTable.setODtype(vai::Dtype::Fp32, dtypeKey());
        scheduleTable.setODtype(vai::Dtype::Fp32, dtypeKey());
        ddpmStep.setODtype(vai::Dtype::Fp32, dtypeKey());
    }

    // Input: images [numObsSteps * numCameras, 3, S, S], state [numObsSteps, stateDim].
    void encode(const vai::Tensor& images, const vai::Tensor& state)
    {
        (*this)({ { "inImages", images }, { "inState", state } }, { { "outGlobalCond", &globalCond } }, false);
    }

    // Draws the loop's noise from seed: the starting sample x_T into action
    // [horizon, actionDim], and every step's z into stepNoise. Call it ahead of decode().
    void drawNoise(uint64_t seed, vai::Tensor* action)
    {
        VAI_ASSERT(action);
        const size_t sampleSize = size_t(cfg.horizon) * cfg.actionDim;

        sync();   // The GPU may still be reading stepNoise.
        *action = vai::Tensor({ cfg.horizon, cfg.actionDim }, vai::Dtype::Fp32).setFloats(sampleNoise(sampleSize, seed));
        stepNoise.setFloats(sampleNoise(stepNoise.numElements(), seed + 1));
    }

    // Input: action [horizon, actionDim], x_T (drawNoise()'s); encode() ran first.
    // Output: the denoised sample x_0 in the same GPU tensor. The executed actions are
    // its rows actionStart() .. actionStart() + numActionSteps, still normalized.
    void decode(vai::Tensor* action)
    {
        VAI_ASSERT(action);

        sync();   // The GPU may still be reading the readers' counters.
        timestepTable.rewind();
        scheduleTable.rewind();

        for (uint32_t step = 0; step < runtimeCfg.denoiseSteps; ++step)
            (*this)({ { "inActions", *action }, { "inGlobalCond", globalCond }, { "inStepNoise", stepNoise } },
                    { { "outActions", action } },
                    false);
    }
};

}   // namespace model::experimental::diffusion_policy

#endif //EXPERIMENTAL_DIFFUSION_POLICY_NET_H
