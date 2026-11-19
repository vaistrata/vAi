#include "net.h"
#include "../common/vla-config.h"
#include "bench-utils.h"
#include <migraphx/version.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// net.h and the bench's VlaConfig share one namespace.
namespace pi05 = bench::pi05;


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [bundle-dir] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  bundle-dir     the bundle to read, instead of %s\n",
        pi05::vlaConfigs::lerobotLibero.dataDir().string().c_str());
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(
        stderr,
        "  --atol/--rtol  tolerances against %s (default: 0.002 / 0.001)\n",
        pi05::vlaConfigs::lerobotLibero.referencePath().string().c_str());
}


bool checkHip(hipError_t result, const char* operation)
{
    if (result != hipSuccess)
    {
        fprintf(stderr, "error: %s: %s\n", operation, hipGetErrorString(result));
        return false;
    }
    return true;
}


struct HipStream
{
    hipStream_t handle{};
    HipStream() = default;
    ~HipStream()
    {
        if (handle)
        {
            (void)hipStreamSynchronize(handle);
            (void)hipStreamDestroy(handle);
        }
    }
    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;
};


int main(int argc, char** argv)
{
    // MIGraphX runs the ONNX export of the LeRobot LIBERO policy.
    const pi05::VlaConfig& config = pi05::vlaConfigs::lerobotLibero;
    VLABenchOptions options;
    std::filesystem::path bundle;

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
        if (argv[i][0] != '-')
        {
            if (!bundle.empty())
            {
                fprintf(stderr, "error: only one bundle directory is allowed\n");
                usage(argv[0]);
                return 1;
            }
            bundle = argv[i];
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
        fprintf(stderr, "error: unknown option: %s\n", argv[i]);
        usage(argv[0]);
        return 1;
    }
    if (bundle.empty())
        bundle = config.dataDir();

    int nDevices = 0;
    if (!checkHip(hipGetDeviceCount(&nDevices), "hipGetDeviceCount"))
        return 1;

    std::vector<std::string> gpuNames;
    for (int i = 0; i < nDevices; ++i)
    {
        hipDeviceProp_t props{};
        if (!checkHip(hipGetDeviceProperties(&props, i), "hipGetDeviceProperties"))
            return 1;
        gpuNames.push_back(std::string(props.name) + " (" + props.gcnArchName + ")");
    }

    if (gpuNames.empty())
    {
        fprintf(stderr, "error: no HIP device found\n");
        return 1;
    }

    const int device = pickGpu(gpuNames);
    if (!checkHip(hipSetDevice(device), "hipSetDevice"))
        return 1;
    int hipVersion = 0;
    if (!checkHip(hipRuntimeGetVersion(&hipVersion), "hipRuntimeGetVersion"))
        return 1;
    printf(
        "model: pi0.5 %s, MIGraphX %d.%d, HIP %d\n",
        config.dataset,
        MIGRAPHX_VERSION_MAJOR,
        MIGRAPHX_VERSION_MINOR,
        hipVersion);
    printf("device: HIP %d, %s\n", device, gpuNames[device].c_str());
    printf("precision: FP16 graphs (weights, activations, K/V cache); FP32 images, timestep, x\n");

    // The runtime reads these files while compiling; report a missing one before it does.
    for (const char* file : { "config.txt", "prefill.fp16.onnx", "step.fp16.onnx" })
    {
        std::error_code status;
        if (!std::filesystem::is_regular_file(bundle / file, status))
        {
            fprintf(
                stderr, "error: %s is missing; see benchmarks/SETUP.md, MIGraphX\n", (bundle / file).string().c_str());
            return 1;
        }
    }

    const bool folded = pi05::bundleConstantsFolded(bundle);
    printf("bundle: %s (constants %s)\n", bundle.string().c_str(), folded ? "folded" : "not folded");
    if (!folded)
        printf("note: MLIR runs a bundle without folded constants far slower; for the default GEMM path "
               "re-export it with benchmarks/pi05/migraphx/export-onnx.py\n");
    fflush(stdout);

    // Compile the bundle's prefill and step graphs (untimed).
    // Declared before the policy, so the policy drains it before it is destroyed.
    HipStream stream;
    if (!checkHip(hipStreamCreateWithFlags(&stream.handle, hipStreamNonBlocking), "hipStreamCreateWithFlags"))
        return 1;
    const double compileStart = nowSec();
    std::optional<pi05::Pi05Policy> policy;
    policy.emplace(bundle, stream.handle);
    printf("compile: %.1f s (gpu, offload_copy off)\n", nowSec() - compileStart);
    const char* noMlir = std::getenv("MIGRAPHX_DISABLE_MLIR");
    const char* gemmProvider = std::getenv("MIGRAPHX_SET_GEMM_PROVIDER");

    printf(
        "gemm: MIGRAPHX_DISABLE_MLIR=%s, MIGRAPHX_SET_GEMM_PROVIDER=%s\n",
        noMlir ? noMlir : "unset",
        gemmProvider ? gemmProvider : "unset");

    auto& runtime = policy->runtime();
    const auto& modelCfg = runtime.getConfig();

    printf(
        "graphs: prefill once + reusable Euler step x%u, %d layers, prefix length %d\n",
        config.denoiseSteps,
        modelCfg.numLayers,
        modelCfg.prefixLength);

    // The bundle keeps the shapes it was exported with.
    if (modelCfg.numDenoiseSteps != int32_t(config.denoiseSteps) ||
        modelCfg.actionHorizon != int32_t(config.actionChunk) ||
        modelCfg.actionDim != int32_t(pi05::Pi05Model::actionDim) || modelCfg.numViews != int32_t(config.cameras))
    {
        fprintf(
            stderr,
            "error: the bundle (%d steps, chunk [%d, %d], %d views) does not match the bench configuration "
            "(%u steps, chunk [%u, %u], %u views); re-export it with benchmarks/pi05/migraphx/export-onnx.py\n",
            modelCfg.numDenoiseSteps,
            modelCfg.actionHorizon,
            modelCfg.actionDim,
            modelCfg.numViews,
            config.denoiseSteps,
            config.actionChunk,
            pi05::Pi05Model::actionDim,
            config.cameras);
        return 1;
    }
    runtime.setNumDenoiseSteps(int32_t(config.denoiseSteps));

    const uint32_t B = config.cameras;
    const uint32_t C = pi05::Pi05Model::imageChannels;
    const uint32_t S = pi05::Pi05Model::imageSize;

    // The bundle keeps its export's shapes, so the prompt must have the exported length;
    // any other change to the observation fails the reference check.
    const pi05::Inputs inputs = pi05::prepareInputs(config, pi05::signedCHW);
    const uint32_t nPrompt = (uint32_t)inputs.tokens.size();
    if (nPrompt != uint32_t(modelCfg.numPromptTokens))
    {
        fprintf(
            stderr,
            "error: the prompt has %u tokens, the bundle was exported for %d\n",
            nPrompt,
            modelCfg.numPromptTokens);
        return 1;
    }
    if (inputs.noise.size() != config.actionCount())
    {
        fprintf(stderr, "error: the initial noise must hold %zu values\n", config.actionCount());
        return 1;
    }
    runtime.setInitialNoise(inputs.noise);
    pi05::Tensor inImages({ B, C, S, S }, migraphx_shape_float_type);
    std::vector<float> nextAction;

    // Each prediction includes H2D pixels, inference and the host action chunk.
    // The runtime uploads host tokens and the configured initial noise internally.
    auto predict = [&]() -> bool {
        if (!checkHip(
                hipMemcpyAsync(
                    inImages.rawPointer(),
                    inputs.pixels.data(),
                    inputs.pixels.size() * sizeof(float),
                    hipMemcpyHostToDevice,
                    stream.handle),
                "hipMemcpyAsync(pixels)"))
            return false;

        auto chunk = policy->inferTensors(inImages, inputs.tokens, 1);
        if (!checkHip(hipStreamSynchronize(stream.handle), "hipStreamSynchronize"))
            return false;

        nextAction = std::move(chunk.normalizedActions);
        if (nextAction.size() != config.actionCount())
        {
            fprintf(stderr, "error: MIGraphX returned an unexpected action chunk size\n");
            return false;
        }
        return true;
    };

    pi05::printVlaConfig(config, inputs);
    options.referencePath = config.referencePath().string();
    options.denoiseSteps = config.denoiseSteps;
    auto actions = [&] { return (const float*)nextAction.data(); };

    return runVLABench(
        "MIGraphX",
        int(config.imageTokens()),
        int(nPrompt),
        options,
        config.actionChunk,
        pi05::Pi05Model::actionDim,
        predict,
        actions);
}
