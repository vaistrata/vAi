#include "vla.h"
#include "../common/workload.h"
#include "bench-utils.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace common = bench::pi05;


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [--model PATH] [--config PATH] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  --model        GGUF checkpoint (default: VLA_MODEL_PATH or %s)\n",
        (common::vsVlaCpp.dataDir() / "model.gguf").string().c_str());
    fprintf(stderr, "  --config       runtime config (default: VLA_CONFIG_PATH or %s)\n", VLACPP_RUNTIME_CONFIG);
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(
        stderr,
        "  --atol/--rtol  tolerances against %s (default: 0.002 / 0.001)\n",
        common::vsVlaCpp.referencePath().string().c_str());
}


int main(int argc, char** argv)
{
    // vla.cpp runs the LeRobot LIBERO policy.
    const common::Workload& workload = common::vsVlaCpp;
    VLABenchOptions options;
    const char* modelPath = std::getenv("VLA_MODEL_PATH");
    const char* configPath = std::getenv("VLA_CONFIG_PATH");

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
        if (!strcmp(argv[i], "--model") || !strcmp(argv[i], "--config"))
        {
            const char* option = argv[i];
            if (++i == argc || !*argv[i])
            {
                fprintf(stderr, "error: %s needs a path\n", option);
                usage(argv[0]);
                return 1;
            }
            if (!strcmp(option, "--model"))
                modelPath = argv[i];
            else
                configPath = argv[i];
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
        usage(argv[0]);
        return 1;
    }

    const std::string defaultModelPath = (workload.dataDir() / "model.gguf").string();
    if (!modelPath || !*modelPath)
        modelPath = defaultModelPath.c_str();
    // vla.cpp takes the weight dtype only from a runtime config file, BF16 without one; the bench's
    // own file sets FP16.
    if (!configPath || !*configPath)
        configPath = VLACPP_RUNTIME_CONFIG;

    if (vla_abi_version() != VLA_ABI_VERSION)
    {
        fprintf(stderr, "error: vla.cpp library ABI does not match vla.h\n");
        return 1;
    }

    // vla.cpp runs on the CUDA device VLA_DEVICE names when it loads the model.
    int nDevices = 0;
    if (cudaGetDeviceCount(&nDevices) != cudaSuccess || nDevices == 0)
    {
        fprintf(stderr, "error: no CUDA device found\n");
        return 1;
    }
    std::vector<std::string> gpuNames;
    for (int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, i) != cudaSuccess)
        {
            fprintf(stderr, "error: cannot read CUDA device %d\n", i);
            return 1;
        }
        gpuNames.push_back(props.name);
    }
    const int device = pickGpu(gpuNames);
    setenv("VLA_DEVICE", std::to_string(device).c_str(), 1);
    printf("device: CUDA %d, %s\n", device, gpuNames[device].c_str());

    // Load the GGUF checkpoint and vla.cpp runtime config.
    printf("model: pi0.5 %s\n", workload.dataset);
    printf("checkpoint: %s\nconfig: %s\n", modelPath, configPath);
    fflush(stdout);

    std::unique_ptr<vla_model, decltype(&vla_model_free)> model(
        vla_model_load(nullptr, modelPath, configPath), &vla_model_free);
    vla_config config{};
    if (!model || vla_model_config(model.get(), &config) != VLA_OK)
    {
        fprintf(stderr, "error: failed to load VLA model or read its config\n");
        return 1;
    }
    if (config.max_state_dim != int64_t(workload.actionDim) || config.max_action_dim != int64_t(workload.actionDim) ||
        config.n_suffix != int64_t(workload.actionChunk) || config.num_steps != int32_t(workload.denoiseSteps))
    {
        fprintf(
            stderr,
            "error: VLA config must match state/action dim %u, action chunk %u and %u denoise steps\n",
            workload.actionDim,
            workload.actionChunk,
            workload.denoiseSteps);
        return 1;
    }

    // vla.cpp normalizes internally; its float input is interleaved RGB [0, 1].
    const common::Inputs inputs = common::prepareInputs(workload, common::unitHWC);
    const uint32_t B = workload.cameras;
    const uint32_t C = workload.imageChannels;
    const uint32_t S = workload.imageSize;
    const uint32_t nPrompt = (uint32_t)inputs.tokens.size();
    if (nPrompt > workload.maxPromptTokens)
    {
        fprintf(stderr, "error: the prompt has %u tokens, the model allows %u\n", nPrompt, workload.maxPromptTokens);
        return 1;
    }

    std::vector<vla_image> images;
    images.reserve(B);
    const size_t cameraSize = size_t(C) * S * S;
    for (uint32_t slot = 0; slot < B; ++slot)
    {
        vla_image image{};
        image.data = inputs.pixels.data() + size_t(slot) * cameraSize;
        image.w = int32_t(S);
        image.h = int32_t(S);
        image.format = VLA_PIXEL_F32_RGB_01;
        images.push_back(image);
    }

    // All pointers are borrowed for each prediction; the backing vectors stay alive.
    const std::vector<int32_t> tokens(inputs.tokens.begin(), inputs.tokens.end());
    vla_inputs input{};
    input.images = images.data();
    input.n_images = int32_t(B);
    input.lang_tokens = tokens.data();
    input.n_lang = int32_t(nPrompt);
    input.state = inputs.state.data();
    input.noise = inputs.noise.data();
    input.timing_detail = VLA_TIMING_NONE;
    std::unique_ptr<float, decltype(&vla_free_actions)> nextAction(nullptr, &vla_free_actions);

    // One synchronous public call includes input upload and host action readback.
    auto predict = [&]() -> bool {
        float* actions = nullptr;
        int64_t numValues = 0;
        const int32_t status = vla_predict(model.get(), &input, &actions, &numValues);
        nextAction.reset(actions);
        if (status != VLA_OK)
        {
            fprintf(stderr, "error: vla_predict failed (status=%d)\n", int(status));
            return false;
        }
        if (!actions || numValues != int64_t(workload.actionCount()))
        {
            fprintf(
                stderr,
                "error: vla.cpp output must match action chunk [%u, %u]\n",
                workload.actionChunk,
                workload.actionDim);
            return false;
        }
        return true;
    };

    common::printWorkload(workload, inputs);
    // vla.cpp applies the GGUF's dataset statistics to the actions, so they match the
    // reference only for statistics that leave the actions as the model gives them
    // (q01 = -1, q99 = 1).
    options.referencePath = workload.referencePath().string();
    options.denoiseSteps = workload.denoiseSteps;
    auto actions = [&] { return (const float*)nextAction.get(); };
    return runVLABench(
        "vla.cpp",
        int(workload.imageTokens()),
        int(nPrompt),
        options,
        workload.actionChunk,
        workload.actionDim,
        predict,
        actions);
}
