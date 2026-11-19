#include "net.h"
#include "lerobot-weights.h"
#include "../common/vla-config.h"
#include "eva-runtime.h"
#include "bench-utils.h"
#include <cstdio>
#include <cstring>

using namespace vai;
namespace pi05 = model::experimental::pi05;
namespace common = bench::pi05;


// The checkpoints vAi loads, found by the configuration's dataset: their variant is its folder.
constexpr pi05::ModelConfig (*modelConfigs[])() = {
    &pi05::ModelConfig::_3B,
    &pi05::ModelConfig::_LIBERO,
};


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [--comparison NAME] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  --comparison   the configuration of the framework compared with (default: %s), one of:",
        common::comparisons[0].name);
    for (const common::Comparison& c : common::comparisons)
        fprintf(stderr, " %s", c.name);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(stderr, "  --atol/--rtol  tolerances against the configuration's reference (default: 0.002 / 0.001)\n");
}


int main(int argc, char** argv)
{
    VLABenchOptions options;
    const char* comparison = common::comparisons[0].name;

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--showActions"))
        {
            options.showActions = true;
            continue;
        }
        if (!strcmp(argv[i], "--help"))
        {
            usage(argv[0]);
            return 0;
        }
        if (!strcmp(argv[i], "--comparison"))
        {
            if (++i == argc)
            {
                fprintf(stderr, "error: --comparison needs a name\n");
                usage(argv[0]);
                return 1;
            }
            comparison = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--atol") || !strcmp(argv[i], "--rtol"))
        {
            const char* key = argv[i];
            if (++i == argc)
            {
                fprintf(stderr, "error: %s needs a value\n", key);
                usage(argv[0]);
                return 1;
            }
            double tolerance = 0.0;
            if (!parseNumber(argv[i], tolerance) || tolerance < 0)
            {
                fprintf(stderr, "error: invalid tolerance for %s\n", key);
                usage(argv[0]);
                return 1;
            }
            (!strcmp(key, "--atol") ? options.atol : options.rtol) = tolerance;
            continue;
        }
        fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
        usage(argv[0]);
        return 1;
    }

    const common::VlaConfig* found = common::findVlaConfig(comparison);
    if (!found)
    {
        fprintf(stderr, "error: unknown --comparison '%s'\n", comparison);
        usage(argv[0]);
        return 1;
    }
    const common::VlaConfig& config = *found;
    pi05::ModelConfig (*modelConfig)() = nullptr;

    for (auto checkpoint : modelConfigs)
        if (!strcmp(checkpoint().variant, config.dataset))
            modelConfig = checkpoint;

    if (!modelConfig)
    {
        fprintf(stderr, "error: vAi has no pi0.5 checkpoint for the dataset '%s'\n", config.dataset);
        return 1;
    }

    eva::Device device = getVaiDevice();
    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    // Load model. The chunk and the step count are the configuration's, not the checkpoint's.
    pi05::ModelConfig modelCfg = modelConfig();
    modelCfg.actionChunk = config.actionChunk;
    printf("model: pi0.5 %s\n", modelCfg.variant);
    fflush(stdout);
    if (modelCfg.actionDim != common::Pi05Model::actionDim ||
        modelCfg.visionTrf.imageSize != common::Pi05Model::imageSize ||
        modelCfg.visionTrf.imageChannels != common::Pi05Model::imageChannels || config.cameras > modelCfg.maxCameras)
    {
        fprintf(stderr, "error: the vAi model does not match the bench configuration\n");
        return 1;
    }

    // clang-format off
    pi05::Net net({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
    }, modelCfg, { .denoiseSteps = config.denoiseSteps });
    // clang-format on

    // Load weights
    loadWeights<"lerobot">(net);

    // vAi takes its images in SigLIP's range whichever framework it is compared with.
    const common::Inputs inputs = common::prepareInputs(config, common::signedCHW);
    const uint32_t B = config.cameras;
    const uint32_t C = common::Pi05Model::imageChannels;
    const uint32_t S = common::Pi05Model::imageSize;
    const uint32_t nPrompt = (uint32_t)inputs.tokens.size();
    if (nPrompt > modelCfg.maxPromptTokens)
    {
        fprintf(stderr, "error: the prompt has %u tokens, the model allows %u\n", nPrompt, modelCfg.maxPromptTokens);
        return 1;
    }

    // Reused across predictions, as the demos do: a new Tensor would land on a different
    // device buffer.
    Tensor inImages({ B, C, S, S });
    Tensor inTokenIds({ nPrompt });
    Tensor nextAction({ config.actionChunk, common::Pi05Model::actionDim });

    // Upload every input on each call. Readback waits for the complete action chunk.
    auto predict = [&] {
        inImages.setData(inputs.pixels.data());
        inTokenIds.setData(inputs.tokens.data());
        nextAction.setData(inputs.noise.data());
        net.prefill(inImages, inTokenIds);
        net.decode(&nextAction);
        nextAction.getData<float>();
    };
    auto actions = [&] { return nextAction.getData<float>(); };

    common::printVlaConfig(config, inputs);
    options.referencePath = config.referencePath().string();
    options.denoiseSteps = config.denoiseSteps;

    return runVLABench(
        "vAi",
        int(config.imageTokens()),
        int(nPrompt),
        options,
        config.actionChunk,
        common::Pi05Model::actionDim,
        predict,
        actions);
}
