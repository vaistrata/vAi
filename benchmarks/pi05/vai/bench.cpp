#include "net.h"
#include "lerobot-weights.h"
#include "../common/profile.h"
#include "../common/workload.h"
#include "eva-runtime.h"
#include "bench-utils.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;
namespace pi05 = model::production::pi05;
namespace common = bench::pi05;


struct SizeOption
{
    const char* name;
    pi05::ModelConfig (*config)();
};

constexpr SizeOption modelSizes[] = {
    { "3B", &pi05::ModelConfig::_3B },
};


void usage(const char* argv0)
{
    fprintf(
        stderr, "usage: %s [N ...] [--size NAME] [--profile NAME] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  N              prompt lengths in tokens, one table row each (default: 1 2 4 8 16 32 64 128 200)\n");
    fprintf(
        stderr,
        "  --size         checkpoint folder under data/ckpts/pi05/lerobot/ (default: %s), one of:",
        modelSizes[0].name);
    for (const SizeOption& s : modelSizes)
        fprintf(stderr, " %s", s.name);
    fprintf(stderr, "\n");
    fprintf(
        stderr, "  --profile      the convention of the framework compared with (default: native), one of: native\n");
    fprintf(stderr, "  --showActions  print the first five actions of the observation's prompt after the table\n");
    fprintf(stderr, "  --atol/--rtol  tolerances against reference.bin (default: 0.002 / 0.001)\n");
}


int main(int argc, char** argv)
{
    // N takes the first N tokens of llmParagraph in bench-inputs.h; the reference check
    // runs the observation's own prompt.
    VLABenchOptions options;
    const char* sizeName = modelSizes[0].name;
    const char* profileName = "native";

    std::vector<int> lengths;

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
        if (!strcmp(argv[i], "--size") || !strcmp(argv[i], "--profile"))
        {
            const char* key = argv[i];
            if (++i == argc)
            {
                fprintf(stderr, "error: %s needs a name\n", key);
                usage(argv[0]);
                return 1;
            }
            (!strcmp(key, "--size") ? sizeName : profileName) = argv[i];
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
        int n = 0;
        if (!parseInt(argv[i], n) || n <= 0)
        {
            usage(argv[0]);
            return 1;
        }
        lengths.push_back(n);
    }
    if (lengths.empty())
        lengths = { 1, 2, 4, 8, 16, 32, 64, 128, 200 };

    const SizeOption* size = nullptr;
    for (const SizeOption& s : modelSizes)
        if (!strcmp(s.name, sizeName))
        {
            size = &s;
            break;
        }
    const common::Workload* workload = common::findWorkload(sizeName);
    if (!size || !workload)
    {
        fprintf(stderr, "error: unknown --size '%s'\n", sizeName);
        usage(argv[0]);
        return 1;
    }
    // vAi follows only its own convention so far; openpi and lerobot come with their
    // pre- and post-processing.
    const common::ProfileSpec* profile = common::findProfile(profileName);
    if (!profile || profile->profile != common::Profile::Native)
    {
        fprintf(stderr, "error: unsupported --profile '%s'\n", profileName);
        usage(argv[0]);
        return 1;
    }

    eva::Device device = getVaiDevice();
    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    // Load model
    auto modelCfg = size->config();
    printf("model: pi0.5 %s\n", modelCfg.variant);
    printf("profile: %s\n", profile->name);
    fflush(stdout);

    // clang-format off
    pi05::Net net({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg);
    // clang-format on
    if (modelCfg.actionChunk != workload->actionChunk || modelCfg.actionDim != workload->actionDim ||
        net.runtimeCfg.denoiseSteps != workload->denoiseSteps ||
        modelCfg.visionTrf.imageSize != workload->imageSize ||
        modelCfg.visionTrf.imageChannels != workload->imageChannels)
    {
        fprintf(stderr, "error: the vAi model does not match the bench workload\n");
        return 1;
    }

    // Load weights
    loadWeights<"lerobot">(net);

    const common::Inputs inputs = common::prepareInputs(*workload, *profile);
    const std::vector<int> paragraphTokens = common::paragraphTokens();
    const uint32_t B = workload->cameras;
    const uint32_t C = workload->imageChannels;
    const uint32_t S = workload->imageSize;
    const uint32_t nObservation = (uint32_t)inputs.tokens.size();
    if (nObservation > modelCfg.maxPromptTokens)
    {
        fprintf(
            stderr,
            "error: the observation's prompt has %u tokens, the model allows %u\n",
            nObservation,
            modelCfg.maxPromptTokens);
        return 1;
    }
    for (int n : lengths)
    {
        if (n > (int)modelCfg.maxPromptTokens)
        {
            fprintf(stderr, "error: prompt %d exceeds the model's %u tokens\n", n, modelCfg.maxPromptTokens);
            return 1;
        }
        if (n > (int)paragraphTokens.size())
        {
            fprintf(stderr, "error: the paragraph has %zu tokens, %d requested\n", paragraphTokens.size(), n);
            return 1;
        }
    }

    Tensor inImages({ B, C, S, S });
    Tensor nextAction({ workload->actionChunk, workload->actionDim });
    auto actions = [&] { return nextAction.getData<float>(); };

    // Upload every input on each call. Readback waits for the complete action chunk.
    auto predictWith = [&](Tensor& inTokenIds, const std::vector<int>& tokens) {
        inImages.setData(inputs.pixels.data());
        inTokenIds.setData(tokens.data());
        nextAction.setData(inputs.noise.data());
        net.prefill(inImages, inTokenIds);
        net.decode(&nextAction);
        nextAction.getData<float>();
    };

    printf(
        "input: %u cameras x %ux%u, action chunk %ux%u, %u denoise steps, seed %llu\n",
        B,
        S,
        S,
        workload->actionChunk,
        workload->actionDim,
        workload->denoiseSteps,
        (unsigned long long)vlaNoiseSeed);

    // The observation's own prompt: the reference check and --showActions.
    Tensor observationTokenIds({ nObservation });
    auto predictObservation = [&] { predictWith(observationTokenIds, inputs.tokens); };
    printf("accuracy: the observation's %u-token prompt\n", nObservation);
    options.referencePath = common::referencePath(*profile, workload->dataDir());
    if (!checkVLAReference(options, workload->actionChunk, workload->actionDim, predictObservation, actions))
        return 1;

    printVLAHeader(options.nWarm, options.nReps);
    for (const int nPrompt : lengths)
    {
        const std::vector<int> promptTokens(paragraphTokens.begin(), paragraphTokens.begin() + nPrompt);

        // Reused across predictions, as the demos do: a new Tensor would land on a different
        // device buffer.
        Tensor inTokenIds({ (uint32_t)nPrompt });
        auto predict = [&] { predictWith(inTokenIds, promptTokens); };
        if (!runVLARow("vAi", nPrompt, options, workload->actionChunk, workload->actionDim, predict, actions))
            return 1;
    }

    if (options.showActions)
    {
        predictObservation();
        printActionChunk(actions(), workload->actionChunk, workload->actionDim);
    }
    return 0;
}
