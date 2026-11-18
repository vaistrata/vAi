#include "net.h"
#include "lerobot-weights.h"
#include "../common/workload.h"
#include "eva-runtime.h"
#include "bench-utils.h"
#include <cstdio>
#include <cstring>

using namespace vai;
namespace pi05 = model::experimental::pi05;
namespace common = bench::pi05;


// The checkpoints vAi loads, found by the workload's dataset: their variant is its folder.
constexpr pi05::ModelConfig (*modelConfigs[])() = {
    &pi05::ModelConfig::_3B,
    &pi05::ModelConfig::_LIBERO,
};


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [--workload NAME] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  --workload     the setting of the framework compared with (default: %s), one of:",
        common::workloads[0]->name);
    for (const common::Workload* w : common::workloads)
        fprintf(stderr, " %s", w->name);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(stderr, "  --atol/--rtol  tolerances against the workload's reference (default: 0.002 / 0.001)\n");
}


int main(int argc, char** argv)
{
    VLABenchOptions options;
    const char* workloadName = common::workloads[0]->name;

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
        if (!strcmp(argv[i], "--workload"))
        {
            if (++i == argc)
            {
                fprintf(stderr, "error: --workload needs a name\n");
                usage(argv[0]);
                return 1;
            }
            workloadName = argv[i];
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

    const common::Workload* workload = common::findWorkload(workloadName);
    if (!workload)
    {
        fprintf(stderr, "error: unknown --workload '%s'\n", workloadName);
        usage(argv[0]);
        return 1;
    }
    pi05::ModelConfig (*modelConfig)() = nullptr;
    for (auto config : modelConfigs)
        if (!strcmp(config().variant, workload->dataset))
            modelConfig = config;
    if (!modelConfig)
    {
        fprintf(stderr, "error: vAi has no pi0.5 checkpoint for the dataset '%s'\n", workload->dataset);
        return 1;
    }

    eva::Device device = getVaiDevice();
    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    // Load model. The chunk and the step count are the workload's, not the checkpoint's.
    pi05::ModelConfig modelCfg = modelConfig();
    modelCfg.actionChunk = workload->actionChunk;
    printf("model: pi0.5 %s\n", modelCfg.variant);
    fflush(stdout);
    if (modelCfg.actionDim != common::Workload::actionDim ||
        modelCfg.visionTrf.imageSize != common::Workload::imageSize ||
        modelCfg.visionTrf.imageChannels != common::Workload::imageChannels || workload->cameras > modelCfg.maxCameras)
    {
        fprintf(stderr, "error: the vAi model does not match the bench workload\n");
        return 1;
    }

    // clang-format off
    pi05::Net net({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
    }, modelCfg, { .denoiseSteps = workload->denoiseSteps });
    // clang-format on

    // Load weights
    loadWeights<"lerobot">(net);

    // vAi takes its images in SigLIP's range whichever framework it is compared with.
    const common::Inputs inputs = common::prepareInputs(*workload, common::signedCHW);
    const uint32_t B = workload->cameras;
    const uint32_t C = common::Workload::imageChannels;
    const uint32_t S = common::Workload::imageSize;
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
    Tensor nextAction({ workload->actionChunk, common::Workload::actionDim });

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

    common::printWorkload(*workload, inputs);
    options.referencePath = workload->referencePath().string();
    options.denoiseSteps = workload->denoiseSteps;
    return runVLABench(
        "vAi",
        int(workload->imageTokens()),
        int(nPrompt),
        options,
        workload->actionChunk,
        common::Workload::actionDim,
        predict,
        actions);
}
