#include "vla.h"
#include "ggml.h"
#include "gguf.h"
#include "../common/vla-config.h"
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


// The dataset statistics vla.cpp applies to the actions it returns, read from the GGUF as
// vla.cpp reads them (src/models/pi05.cpp, load_stats): quantiles when pi05.norm_mode is
// "quantiles", else mean/std. A statistic the GGUF lacks is the identity, as in vla.cpp.
struct ActionStats
{
    bool quantiles = false;
    std::vector<float> q01, q99, mean, std;
    float eps = 0.0f;

    // Inverts vla.cpp's un-normalization for the first q01.size() dims of each row, giving the
    // actions as the model produced them, the space the reference holds.
    void normalize(float* actions, size_t rows, size_t dim) const
    {
        const size_t realDim = q01.size();
        for (size_t r = 0; r < rows; ++r)
        {
            float* row = actions + r * dim;
            for (size_t j = 0; j < realDim && j < dim; ++j)
            {
                if (quantiles)
                    row[j] = 2.0f * (row[j] - q01[j]) / (q99[j] - q01[j]) - 1.0f;
                else
                    row[j] = (row[j] - mean[j]) / (std[j] + eps);
            }
        }
    }
};


bool readTensor(gguf_context* ctx, FILE* file, const char* name, std::vector<float>& dst)
{
    const int64_t id = gguf_find_tensor(ctx, name);
    if (id < 0)
        return true;   // identity, already in dst
    if (gguf_get_tensor_type(ctx, id) != GGML_TYPE_F32 || gguf_get_tensor_size(ctx, id) != dst.size() * sizeof(float))
    {
        fprintf(stderr, "error: %s in the GGUF is not %zu FP32 values\n", name, dst.size());
        return false;
    }
    const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id);
    if (fseeko(file, off_t(offset), SEEK_SET) != 0 || fread(dst.data(), sizeof(float), dst.size(), file) != dst.size())
    {
        fprintf(stderr, "error: cannot read %s from the GGUF\n", name);
        return false;
    }
    return true;
}


bool readActionStats(const char* path, const vla_config& config, ActionStats& stats)
{
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_from_file(path, params), &gguf_free);
    auto closeFile = [](FILE* f) { fclose(f); };
    std::unique_ptr<FILE, decltype(closeFile)> file(fopen(path, "rb"), closeFile);
    if (!ctx || !file)
    {
        fprintf(stderr, "error: cannot read the GGUF header of %s\n", path);
        return false;
    }
    const int64_t modeKey = gguf_find_key(ctx.get(), "pi05.norm_mode");
    stats.quantiles =
        modeKey >= 0 && gguf_get_kv_type(ctx.get(), modeKey) == GGUF_TYPE_STRING &&
        !strcmp(gguf_get_val_str(ctx.get(), modeKey), "quantiles");
    const size_t n = size_t(config.real_action_dim);
    stats.q01.assign(n, -1.0f);
    stats.q99.assign(n, 1.0f);
    stats.mean.assign(n, 0.0f);
    stats.std.assign(n, 1.0f);
    stats.eps = config.norm_eps;
    if (stats.quantiles)
        return readTensor(ctx.get(), file.get(), "action_q01", stats.q01) &&
               readTensor(ctx.get(), file.get(), "action_q99", stats.q99);
    return readTensor(ctx.get(), file.get(), "action_mean", stats.mean) &&
           readTensor(ctx.get(), file.get(), "action_std", stats.std);
}


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [--model PATH] [--config PATH] [--showActions] [--atol N] [--rtol N]\n", argv0);
    fprintf(
        stderr,
        "  --model        GGUF checkpoint (default: VLA_MODEL_PATH or %s)\n",
        (common::vlaConfigs::lerobotLibero.dataDir() / "model.gguf").string().c_str());
    fprintf(stderr, "  --config       runtime config (default: VLA_CONFIG_PATH or %s)\n", VLACPP_RUNTIME_CONFIG);
    fprintf(stderr, "  --showActions  print the first five actions after the table\n");
    fprintf(
        stderr,
        "  --atol/--rtol  tolerances against %s (default: 0.002 / 0.001)\n",
        common::vlaConfigs::lerobotLibero.referencePath().string().c_str());
}


int main(int argc, char** argv)
{
    // vla.cpp runs the LeRobot LIBERO policy.
    const common::VlaConfig& vla = common::vlaConfigs::lerobotLibero;
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

    const std::string defaultModelPath = (vla.dataDir() / "model.gguf").string();
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
    printf("model: pi0.5 %s\n", vla.size);
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
    if (config.max_state_dim != int64_t(common::Pi05Model::actionDim) ||
        config.max_action_dim != int64_t(common::Pi05Model::actionDim) || config.n_suffix != int64_t(vla.actionChunk) ||
        config.num_steps != int32_t(vla.denoiseSteps))
    {
        fprintf(
            stderr,
            "error: VLA config must match state/action dim %u, action chunk %u and %u denoise steps\n",
            common::Pi05Model::actionDim,
            vla.actionChunk,
            vla.denoiseSteps);
        return 1;
    }

    // vla.cpp normalizes internally; its float input is interleaved RGB [0, 1].
    const common::Inputs inputs = common::prepareInputs(vla, common::unitHWC);
    const uint32_t B = vla.cameras;
    const uint32_t C = common::Pi05Model::imageChannels;
    const uint32_t S = common::Pi05Model::imageSize;
    const uint32_t nPrompt = (uint32_t)inputs.tokens.size();
    if (nPrompt > common::Pi05Model::maxPromptTokens)
    {
        fprintf(
            stderr,
            "error: the prompt has %u tokens, the model allows %u\n",
            nPrompt,
            common::Pi05Model::maxPromptTokens);
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

        if (!actions || numValues != int64_t(vla.actionCount()))
        {
            fprintf(
                stderr,
                "error: vla.cpp output must match action chunk [%u, %u]\n",
                vla.actionChunk,
                common::Pi05Model::actionDim);
            return false;
        }

        return true;
    };

    common::printVlaConfig(vla, inputs);
    // vla.cpp applies the GGUF's dataset statistics to the actions and zeroes the dims past
    // real_action_dim. The bench inverts the statistics outside the timed call, so any GGUF
    // compares with the reference, which holds the model's own actions, over the real dims.
    ActionStats stats;
    const bool denormalized = config.denormalized != 0;
    if (denormalized)
    {
        if (!readActionStats(modelPath, config, stats))
            return 1;
        printf(
            "actions: %s statistics of the GGUF inverted, %lld of %u dims real\n",
            stats.quantiles ? "quantile" : "mean/std",
            (long long)config.real_action_dim,
            common::Pi05Model::actionDim);
        options.checkedDims = unsigned(config.real_action_dim);
    }

    options.referencePath = vla.referencePath().string();
    options.denoiseSteps = vla.denoiseSteps;
    std::vector<float> modelActions(vla.actionCount());

    auto actions = [&] {
        if (!denormalized)
            return (const float*)nextAction.get();

        std::memcpy(modelActions.data(), nextAction.get(), modelActions.size() * sizeof(float));
        stats.normalize(modelActions.data(), vla.actionChunk, common::Pi05Model::actionDim);

        return (const float*)modelActions.data();
    };

    return runVLABench(
        "vla.cpp",
        int(vla.imageTokens()),
        int(nPrompt),
        options,
        vla.actionChunk,
        common::Pi05Model::actionDim,
        predict,
        actions);
}
