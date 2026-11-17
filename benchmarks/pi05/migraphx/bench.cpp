#include "net.h"
#include "../common/profile.h"
#include "../common/workload.h"
#include "bench-utils.h"
#include <migraphx/version.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// net.h and the bench's profile and workload share one namespace.
namespace pi05 = bench::pi05;


void usage(const char* argv0)
{
    fprintf(
        stderr,
        "usage: %s [bundle-dir] [--size NAME] [--showActions] [--precision fp16|fp32] [--atol N] [--rtol N]\n",
        argv0);
    fprintf(stderr, "  bundle-dir     the bundle to read, instead of data/benchmarks/pi05/<size>/\n");
    fprintf(
        stderr,
        "  --size         model folder under data/benchmarks/pi05/ (default: %s), one of:",
        pi05::workloads[0].size);
    for (const pi05::Workload& w : pi05::workloads)
        fprintf(stderr, " %s", w.size);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(stderr, "  --precision    fp16 dot/convolution, as vAi's FP16 weights (default), or fp32 throughout\n");
    fprintf(stderr, "  --atol/--rtol  tolerances against reference.bin (default: 0.002 / 0.001)\n");
}


struct HipStream
{
    hipStream_t handle{};
    HipStream() { pi05::hipCheck(hipStreamCreateWithFlags(&handle, hipStreamNonBlocking), "hipStreamCreate"); }
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
try
{
    VLABenchOptions options;
    const char* sizeName = pi05::workloads[0].size;
    std::string precision = "fp16";
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
                throw std::runtime_error("only one bundle directory is allowed");
            bundle = argv[i];
            continue;
        }
        const std::string key = argv[i];
        if (++i == argc)
            throw std::runtime_error(key + " needs a value");
        const std::string value = argv[i];
        if (key == "--size")
            sizeName = argv[i];
        else if (key == "--precision")
            precision = value;
        else if (key == "--atol" || key == "--rtol")
        {
            double tolerance = 0.0;
            if (!parseNumber(value.c_str(), tolerance) || tolerance < 0)
            {
                usage(argv[0]);
                throw std::runtime_error("invalid tolerance for " + key);
            }
            (key == "--atol" ? options.atol : options.rtol) = tolerance;
        }
        else
        {
            usage(argv[0]);
            throw std::runtime_error("unknown option: " + key);
        }
    }
    // The bundle is the ONNX export of vAi's own pipeline: the native convention.
    const pi05::ProfileSpec& profile = pi05::profileSpec(pi05::Profile::Native);
    const pi05::Workload* workload = pi05::findWorkload(sizeName);
    if (!workload)
        throw std::runtime_error(std::string("unknown --size '") + sizeName + "'");
    if (precision != "fp16" && precision != "fp32")
        throw std::runtime_error("--precision must be fp16 or fp32");
    if (bundle.empty())
        bundle = workload->dataDir();

    int nDevices = 0;
    pi05::hipCheck(hipGetDeviceCount(&nDevices), "hipGetDeviceCount");
    std::vector<std::string> gpuNames;
    for (int i = 0; i < nDevices; ++i)
    {
        hipDeviceProp_t props{};
        pi05::hipCheck(hipGetDeviceProperties(&props, i), "hipGetDeviceProperties");
        gpuNames.push_back(std::string(props.name) + " (" + props.gcnArchName + ")");
    }
    if (gpuNames.empty())
        throw std::runtime_error("no HIP device found");
    const int device = pickGpu(gpuNames);
    pi05::hipCheck(hipSetDevice(device), "hipSetDevice");
    int hipVersion = 0;
    pi05::hipCheck(hipRuntimeGetVersion(&hipVersion), "hipRuntimeGetVersion");
    printf(
        "model: pi0.5 %s, MIGraphX %d.%d, HIP %d\n",
        workload->size,
        MIGRAPHX_VERSION_MAJOR,
        MIGRAPHX_VERSION_MINOR,
        hipVersion);
    printf("device: HIP %d, %s\n", device, gpuNames[device].c_str());
    printf("profile: %s\n", profile.name);
    printf("precision: %s dot/convolution; FP32 model inputs/outputs\n", precision.c_str());
    printf("bundle: %s\n", bundle.string().c_str());
    fflush(stdout);

    // Compile the bundle's prefill and step graphs (untimed).
    // Declared before the policy, so the policy drains it before it is destroyed.
    HipStream stream;
    const double compileStart = nowSec();
    std::optional<pi05::Pi05Policy> policy;
    policy.emplace(bundle, stream.handle, precision);
    printf("compile: %.1f s (gpu, offload_copy off)\n", nowSec() - compileStart);
    auto& runtime = policy->runtime();
    const auto& modelCfg = runtime.getConfig();
    printf(
        "graphs: prefill once + reusable Euler step x%u, %d layers, prefix length %d\n",
        workload->denoiseSteps,
        modelCfg.numLayers,
        modelCfg.prefixLength);
    // reference.bin was produced with the exported schedule.
    if (modelCfg.numDenoiseSteps != int32_t(workload->denoiseSteps) ||
        modelCfg.actionHorizon != int32_t(workload->actionChunk) || modelCfg.actionDim != int32_t(workload->actionDim))
        throw std::runtime_error("the bundle does not match the bench workload");
    runtime.setNumDenoiseSteps(int32_t(workload->denoiseSteps));

    const uint32_t B = workload->cameras;
    const uint32_t C = workload->imageChannels;
    const uint32_t S = workload->imageSize;
    policy->validateViewCount(int32_t(B));

    // The bundle keeps its export's shapes, so a prompt of another length stops in
    // inferTensors and any other change to the observation fails the reference check.
    const pi05::Inputs inputs = pi05::prepareInputs(*workload, profile);
    const uint32_t nPrompt = (uint32_t)inputs.tokens.size();
    runtime.setInitialNoise(inputs.noise);
    pi05::Tensor inImages({ B, C, S, S }, migraphx_shape_float_type);
    std::vector<float> nextAction;

    // Each prediction includes H2D pixels, inference and the host action chunk.
    // The runtime uploads host tokens and the configured initial noise internally.
    auto predict = [&] {
        pi05::hipCheck(
            hipMemcpyAsync(
                inImages.rawPointer(),
                inputs.pixels.data(),
                inputs.pixels.size() * sizeof(float),
                hipMemcpyHostToDevice,
                stream.handle),
            "hipMemcpyAsync(pixels)");
        auto chunk = policy->inferTensors(inImages, inputs.tokens, 1);
        pi05::hipCheck(hipStreamSynchronize(stream.handle), "hipStreamSynchronize");
        nextAction = std::move(chunk.normalizedActions);
        if (nextAction.size() != workload->actionCount())
            throw std::runtime_error("MIGraphX returned an unexpected action chunk size");
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

    options.referencePath = pi05::referencePath(profile, bundle);
    auto actions = [&] { return (const float*)nextAction.data(); };
    return runVLABench(
        "MIGraphX", int(nPrompt), options, workload->actionChunk, workload->actionDim, predict, actions);
}
catch (const std::exception& error)
{
    fprintf(stderr, "error: %s\n", error.what());
    return 1;
}
