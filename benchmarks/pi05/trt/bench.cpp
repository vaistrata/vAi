#include "runtime/pi05Policy.h"
#include "common/trtUtils.h"
#include "../common/profile.h"
#include "../common/workload.h"
#include "bench-utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pi05 = trt_edgellm::pi05;
namespace common = bench::pi05;


bool checkCuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "error: %s: %s\n", operation, cudaGetErrorString(result));
        return false;
    }
    return true;
}


struct CudaStream
{
    cudaStream_t handle{};
    CudaStream() = default;
    ~CudaStream()
    {
        if (handle)
            cudaStreamDestroy(handle);
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [--size NAME] [--engineDir PATH] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  --size         model folder under data/benchmarks/pi05/ (default: %s), one of:",
        common::workloads[0].size);
    for (const common::Workload& w : common::workloads)
        fprintf(stderr, " %s", w.size);
    fprintf(stderr, "\n");
    fprintf(
        stderr,
        "  --engineDir    prebuilt Edge-LLM engines (default: PI05_ENGINE_DIR or "
        "data/benchmarks/pi05/<size>/engines)\n");
    fprintf(stderr, "  EDGELLM_PLUGIN_PATH overrides the Edge-LLM plugin library\n");
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(stderr, "  --atol/--rtol  tolerances against reference.bin (default: 0.002 / 0.001)\n");
}


int main(int argc, char** argv)
{
    VLABenchOptions options;
    const char* sizeName = common::workloads[0].size;
    const char* engineDir = std::getenv("PI05_ENGINE_DIR");

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
        if (!strcmp(argv[i], "--engineDir"))
        {
            if (++i == argc || !*argv[i])
            {
                fprintf(stderr, "error: --engineDir needs a path\n");
                usage(argv[0]);
                return 1;
            }
            engineDir = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--size"))
        {
            if (++i == argc)
            {
                fprintf(stderr, "error: --size needs a name\n");
                usage(argv[0]);
                return 1;
            }
            sizeName = argv[i];
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

    // Edge-LLM runs OpenPI's pi05 configuration and returns normalized actions.
    const common::ProfileSpec& profile = common::profileSpec(common::Profile::OpenPI);
    const common::Workload* workload = common::findWorkload(sizeName);
    if (!workload)
    {
        fprintf(stderr, "error: unknown --size '%s'\n", sizeName);
        usage(argv[0]);
        return 1;
    }
    const std::string defaultEngineDir = (workload->dataDir() / "engines").string();
    if (!engineDir || !*engineDir)
        engineDir = defaultEngineDir.c_str();

    int nDevices = 0;
    if (!checkCuda(cudaGetDeviceCount(&nDevices), "cudaGetDeviceCount"))
        return 1;
    std::vector<std::string> gpuNames;
    for (int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp props{};
        if (!checkCuda(cudaGetDeviceProperties(&props, i), "cudaGetDeviceProperties"))
            return 1;
        gpuNames.push_back(props.name);
    }
    if (gpuNames.empty())
    {
        fprintf(stderr, "error: no CUDA device found\n");
        return 1;
    }
    const int device = pickGpu(gpuNames);
    if (!checkCuda(cudaSetDevice(device), "cudaSetDevice"))
        return 1;
    printf("device: CUDA %d, %s\n", device, gpuNames[device].c_str());

    // Register the action engine's AttentionPlugin before deserializing engines.
#ifdef VAI_EDGELLM_PLUGIN_PATH
    if (!std::getenv("EDGELLM_PLUGIN_PATH"))
    {
    #ifdef _WIN32
        if (_putenv_s("EDGELLM_PLUGIN_PATH", VAI_EDGELLM_PLUGIN_PATH) != 0)
    #else
        if (setenv("EDGELLM_PLUGIN_PATH", VAI_EDGELLM_PLUGIN_PATH, 0) != 0)
    #endif
        {
            fprintf(stderr, "error: failed to set EDGELLM_PLUGIN_PATH\n");
            return 1;
        }
    }
#endif
    const auto pluginHandle = trt_edgellm::loadEdgellmPluginLib();
    if (!pluginHandle)
    {
        fprintf(stderr, "error: failed to load Edge-LLM plugin; set EDGELLM_PLUGIN_PATH to the plugin library\n");
        return 1;
    }

    // Load engines built beforehand with pi05_policy_build.
    printf("model: pi0.5 %s\n", workload->size);
    printf("profile: %s\n", profile.name);
    printf("engines: %s\n", engineDir);
    fflush(stdout);

    CudaStream stream;
    if (!checkCuda(cudaStreamCreateWithFlags(&stream.handle, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
        return 1;
    // The runtime throws when the engines or their files are missing.
    std::optional<pi05::Pi05Policy> policy;
    try
    {
        policy.emplace(engineDir, stream.handle);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "error: failed to load the engines: %s\n", e.what());
        return 1;
    }
    auto& runtime = policy->runtime();
    const auto& engineConfig = runtime.getConfig();
    if (engineConfig.actionHorizon != int32_t(workload->actionChunk) ||
        engineConfig.actionDim != int32_t(workload->actionDim))
    {
        fprintf(
            stderr, "error: TensorRT engines must match action chunk [50, 32]; rebuild engines or set --engineDir\n");
        return 1;
    }
    runtime.setNumDenoiseSteps(int32_t(workload->denoiseSteps));
    // CUDA Graph is mandatory for this benchmark; there is no eager-mode option.
    runtime.setUseCudaGraph(true);
    printf("cuda graph: always on\n");

    const uint32_t B = workload->cameras;
    const uint32_t C = workload->imageChannels;
    const uint32_t S = workload->imageSize;
    policy->validateViewCount(int32_t(B));

    const common::Inputs inputs = common::prepareInputs(*workload, profile);
    const uint32_t nPrompt = (uint32_t)inputs.tokens.size();
    if (nPrompt > workload->maxPromptTokens)
    {
        fprintf(stderr, "error: the prompt has %u tokens, the model allows %u\n", nPrompt, workload->maxPromptTokens);
        return 1;
    }

    runtime.setInitialNoise(inputs.noise);
    // The engines take FP16 images.
    std::vector<__half> halfPixels;
    halfPixels.reserve(inputs.pixels.size());
    for (float value : inputs.pixels)
        halfPixels.push_back(__float2half(value));
    trt_edgellm::rt::Tensor inImages(
        trt_edgellm::rt::Coords{ int64_t(B), int64_t(C), int64_t(S), int64_t(S) },
        trt_edgellm::rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    std::vector<float> nextAction;

    // Each prediction includes H2D pixels, inference and the host action chunk.
    // Edge-LLM uploads host tokens and resets the configured initial noise internally.
    auto predict = [&]() -> bool {
        if (!checkCuda(cudaMemcpyAsync(
            inImages.rawPointer(),
            halfPixels.data(),
            halfPixels.size() * sizeof(__half),
            cudaMemcpyHostToDevice,
            stream.handle), "cudaMemcpyAsync(pixels)"))
            return false;

        auto chunk = policy->inferTensors(inImages, inputs.tokens, 1);
        if (!checkCuda(cudaStreamSynchronize(stream.handle), "cudaStreamSynchronize"))
            return false;

        nextAction = std::move(chunk.normalizedActions);
        if (nextAction.size() != workload->actionCount())
        {
            fprintf(stderr, "error: TensorRT returned an unexpected action chunk size\n");
            return false;
        }
        return true;
    };

    printf(
        "input: %u cameras x %ux%u, %u prompt tokens, action chunk %ux%u, %u denoise steps, seed %llu\n",
        B,
        S,
        S,
        nPrompt,
        workload->actionChunk,
        workload->actionDim,
        workload->denoiseSteps,
        (unsigned long long)vlaNoiseSeed);

    // The first predictions also initialize and capture the runtime's CUDA graphs: the
    // reference run and the warmup happen before any latency sample.
    options.referencePath = common::referencePath(profile, workload->dataDir());
    auto actions = [&] { return (const float*)nextAction.data(); };
    return runVLABench(
        "TensorRT Edge-LLM", int(nPrompt), options, workload->actionChunk, workload->actionDim, predict, actions);
}
