#ifndef BENCHMARKS_PI05_MIGRAPHX_NET_H
#define BENCHMARKS_PI05_MIGRAPHX_NET_H

// pi0.5 on MIGraphX, behind the same API shape as TensorRT Edge-LLM's pi05::Pi05Policy:
//
//     bench::pi05::Pi05Policy policy(bundleDir, stream);
//     auto& runtime = policy.runtime();
//     runtime.getConfig();                 // actionHorizon, actionDim, ...
//     runtime.setNumDenoiseSteps(10);
//     runtime.setInitialNoise(noise);
//     policy.validateViewCount(2);
//     auto chunk = policy.inferTensors(images, tokens, 1);   // chunk.normalizedActions
//
// The bundle is the v5 split export of migraphx/export-onnx.py: prefill.fp16.onnx, run once
// per prediction to fill the K/V cache, and step.fp16.onnx, one reusable Euler step. Both
// are FP16 graphs (weights, activations, the K/V cache between them) with FP32 images,
// timestep and x. Its config.txt names the camera views, steps and chunk it was exported with.

#include <migraphx/migraphx.hpp>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef MIGRAPHX_GPU_HIP_FLAGS_DEFAULT
    #define MIGRAPHX_GPU_HIP_FLAGS_DEFAULT ""
#endif
// mlir: dots fused into MLIR kernels. hipblaslt: every dot lowered to hipBLASLt, MLIR off.
// Either way a dot left outside MLIR goes to hipBLASLt.
#ifndef MIGRAPHX_GEMM_PROVIDER_DEFAULT
    #define MIGRAPHX_GEMM_PROVIDER_DEFAULT "mlir"
#endif

namespace bench::pi05 {

inline void hipCheck(hipError_t status, const char* what)
{
    if (status != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}


struct DeviceBuffer
{
    void* data = nullptr;
    explicit DeviceBuffer(size_t bytes) { hipCheck(hipMalloc(&data, bytes), "hipMalloc"); }
    ~DeviceBuffer()
    {
        if (data)
            (void)hipFree(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};


// A packed device tensor, as trt_edgellm::rt::Tensor on the GPU.
class Tensor
{
public:
    Tensor(std::vector<size_t> shape, migraphx_shape_datatype_t type)
        : shape_(std::move(shape)), type_(type), buffer_(migraphx::shape(type, shape_).bytes())
    {
    }

    void* rawPointer() const { return buffer_.data; }
    const std::vector<size_t>& getShape() const { return shape_; }
    migraphx_shape_datatype_t getDataType() const { return type_; }
    size_t bytes() const { return migraphx::shape(type_, shape_).bytes(); }

private:
    std::vector<size_t> shape_;
    migraphx_shape_datatype_t type_;
    DeviceBuffer buffer_;
};


namespace detail {

inline double nowSec()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}


inline void requireShape(const migraphx::shape& actual, const migraphx::shape& expected, const std::string& name)
{
    if (actual.dynamic() || !actual.standard() || actual.type() != expected.type() ||
        actual.lengths() != expected.lengths() || actual.bytes() != expected.bytes())
        throw std::runtime_error("unexpected type or packed shape: " + name);
}


inline std::optional<size_t> outputIndex(const std::string& name)
{
    std::string_view suffix;
    if (name.starts_with("#output_"))
        suffix = std::string_view(name).substr(8);
    else if (name.starts_with("main:#output_"))
        suffix = std::string_view(name).substr(13);
    else
        return std::nullopt;
    // MIGraphX 2.18 also uses names such as main:#output_:00035.
    if (suffix.starts_with(':'))
        suffix.remove_prefix(1);
    size_t index = 0;
    const auto [end, error] = std::from_chars(suffix.data(), suffix.data() + suffix.size(), index);
    if (error != std::errc{} || end != suffix.data() + suffix.size())
        throw std::runtime_error("invalid explicit output parameter: " + name);
    return index;
}


// MIGraphX's parameter names are not ordered. Bind explicit outputs by their numeric
// graph index so that #output_10 cannot be mistaken for the third output.
inline std::vector<std::string> validateContract(
    const migraphx::program& program, const std::map<std::string, migraphx::shape>& expectedInputs,
    const std::vector<migraphx::shape>& expectedOutputs, const std::string& tag)
{
    const auto outputs = program.get_output_shapes();
    if (outputs.size() != expectedOutputs.size())
        throw std::runtime_error(tag + ": unexpected output count");
    for (size_t index = 0; index < outputs.size(); ++index)
        requireShape(outputs[index], expectedOutputs[index], tag + " output " + std::to_string(index));

    const auto parameters = program.get_parameter_shapes();
    std::vector<std::string> outputNames(expectedOutputs.size());
    size_t inputCount = 0;
    for (const char* rawName : parameters.names())
    {
        const std::string name = rawName;
        const auto shape = parameters[rawName];
        if (const auto index = outputIndex(name))
        {
            if (*index >= expectedOutputs.size() || !outputNames[*index].empty())
                throw std::runtime_error(tag + ": invalid or duplicate output parameter " + name);
            requireShape(shape, expectedOutputs[*index], tag + " parameter " + name);
            outputNames[*index] = name;
        }
        else
        {
            const auto expected = expectedInputs.find(name);
            if (expected == expectedInputs.end())
                throw std::runtime_error(tag + ": unexpected input " + name);
            requireShape(shape, expected->second, tag + " input " + name);
            ++inputCount;
        }
    }
    if (inputCount != expectedInputs.size() ||
        std::any_of(outputNames.begin(), outputNames.end(), [](const auto& name) { return name.empty(); }))
        throw std::runtime_error(tag + ": missing input or explicit output parameter");
    return outputNames;
}


inline migraphx::program compileGraph(const std::filesystem::path& path, const char* tag)
{
    // The graph is FP16 as exported: no conversion here.
    printf("%s ONNX parsing: starting\n", tag);
    fflush(stdout);
    const double parseStart = nowSec();
    migraphx::onnx_options onnxOptions;
    auto program = migraphx::parse_onnx(path.string().c_str(), onnxOptions);
    printf("%s ONNX parsing: completed in %.1f s\n", tag, nowSec() - parseStart);
    printf("%s GPU compilation: starting (untimed)\n", tag);
    fflush(stdout);
    const double gpuCompileStart = nowSec();
    migraphx::compile_options compileOptions;
    compileOptions.set_offload_copy(false);
    program.compile(migraphx::target("gpu"), compileOptions);
    printf("%s GPU compilation: completed in %.1f s\n", tag, nowSec() - gpuCompileStart);
    fflush(stdout);
    return program;
}


struct BundleConfig
{
    int views = 0;
    int steps = 0;
    int horizon = 0;
    int actionDim = 0;
    int layers = 0;
    int prefixLength = 0;
};


inline BundleConfig readBundleConfig(const std::filesystem::path& bundle)
{
    int version = 0;
    BundleConfig config;
    std::ifstream file(bundle / "config.txt");
    if (!(file >> version))
        throw std::runtime_error(
            "missing pi05 bundle config in " + bundle.string() + "; see benchmarks/SETUP.md, MIGraphX");
    if (version >= 1 && version <= 4)
        throw std::runtime_error(
            "v" + std::to_string(version) +
            " pi05 bundle (FP32 graph) is unsupported; re-export with python "
            "benchmarks/pi05/migraphx/export-onnx.py --output NEW_DIR, then pass NEW_DIR to this benchmark");
    if (!(file >> config.views >> config.steps >> config.horizon >> config.actionDim >> config.layers >>
          config.prefixLength) ||
        version != 5 || config.views < 1 || config.views > 3 || config.steps < 1 || config.horizon < 1 ||
        config.actionDim != 32 || config.layers != 18 || config.prefixLength <= config.views * 256 ||
        config.prefixLength > config.views * 256 + 200)
        throw std::runtime_error(
            "invalid v5 split pi05 bundle in " + bundle.string() + "; see benchmarks/SETUP.md, MIGraphX");
    std::string extra;
    if (file >> extra)
        throw std::runtime_error("unexpected fields in v5 pi05 bundle config");
    return config;
}

}   // namespace detail


// Whether export-onnx.py folded the bundle's constants, from its metadata.json; bundles
// exported before it recorded this were not folded.
inline bool bundleConstantsFolded(const std::filesystem::path& bundle)
{
    std::ifstream file(bundle / "metadata.json");
    const std::string text{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
    constexpr std::string_view key = "\"constant_folding\"";
    const size_t at = text.find(key);
    if (at == std::string::npos)
        return false;
    const size_t value = text.find_first_not_of(" \t\r\n:", at + key.size());
    return value != std::string::npos && text.compare(value, 4, "true") == 0;
}


// The counterpart of Edge-LLM's runtime config, read from the bundle's config.txt.
struct Pi05Config
{
    int32_t numViews = 0;
    int32_t imageChannels = 3;
    int32_t imageSize = 224;
    int32_t actionHorizon = 0;
    int32_t actionDim = 0;
    int32_t numLayers = 0;
    int32_t prefixLength = 0;
    int32_t numPromptTokens = 0; // the exported prompt length; prompts must match it exactly
    int32_t numDenoiseSteps = 0;   // the steps the bundle was exported and referenced with
};


struct ActionChunk
{
    std::vector<float> normalizedActions; // [actionHorizon, actionDim]
};


// Owns the compiled prefill/step programs and every device binding. All bindings are made
// once and reused by every prediction: a step's output becomes the next step's input, and
// the K/V cache stays on the device, without any intermediate host copy.
class Pi05Runtime
{
public:
    Pi05Runtime(const std::filesystem::path& bundle, hipStream_t stream)
    : stream_(stream)
    , bundleConfig_(detail::readBundleConfig(bundle))
    , prefill_(compileWithFlags(bundle / "prefill.fp16.onnx", "Prefill"))
    , step_(compileWithFlags(bundle / "step.fp16.onnx", "Step"))
    {
        config_.numViews = bundleConfig_.views;
        config_.actionHorizon = bundleConfig_.horizon;
        config_.actionDim = bundleConfig_.actionDim;
        config_.numLayers = bundleConfig_.layers;
        config_.prefixLength = bundleConfig_.prefixLength;
        config_.numPromptTokens = bundleConfig_.prefixLength - config_.numViews * 256;
        config_.numDenoiseSteps = bundleConfig_.steps;

        bindGraphs();
        setNumDenoiseSteps(config_.numDenoiseSteps);
    }

    ~Pi05Runtime()
    {
        // Drain queued work before its host and device buffers go away.
        if (stream_)
            (void)hipStreamSynchronize(stream_);
    }

    Pi05Runtime(const Pi05Runtime&) = delete;
    Pi05Runtime& operator=(const Pi05Runtime&) = delete;

    const Pi05Config& getConfig() const { return config_; }

    // The Euler schedule t = 1, 1 - 1/n, ..., 1/n, as the export traced it.
    void setNumDenoiseSteps(int32_t steps)
    {
        if (steps <= 0)
            throw std::runtime_error("the number of denoise steps must be positive");
        hipCheck(hipStreamSynchronize(stream_), "hipStreamSynchronize");
        std::vector<float> timesteps(static_cast<size_t>(steps));
        for (int32_t index = 0; index < steps; ++index)
            timesteps[size_t(index)] = float(1.0 + index * (-1.0 / steps));
        timestepBuffer_ = std::make_unique<DeviceBuffer>(timesteps.size() * sizeof(float));
        hipCheck(
            hipMemcpy(timestepBuffer_->data, timesteps.data(), timesteps.size() * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy(timesteps)");

        const auto stepShapes = step_.get_parameter_shapes();
        // Each step needs its own handle: copies of a program_parameters share one.
        stepParameters_.clear();
        for (int32_t index = 0; index < steps; ++index)
        {
            auto& parameters = stepParameters_.emplace_back();
            parameters.add("x_t", migraphx::argument(stepShapes["x_t"], xBuffers_[index % 2]->data));
            auto* timestep = (char*)timestepBuffer_->data + size_t(index) * sizeof(float);
            parameters.add("timestep", migraphx::argument(stepShapes["timestep"], timestep));
            for (const auto& [name, outputIndex] : cacheOutputIndices_)
                parameters.add(
                    name.c_str(), migraphx::argument(stepShapes[name.c_str()], cacheBuffers_[outputIndex]->data));
            const auto& outputName = stepOutputNames_[0];
            parameters.add(
                outputName.c_str(),
                migraphx::argument(stepShapes[outputName.c_str()], xBuffers_[(index + 1) % 2]->data));
        }
        lastStepOutputs_.assign(size_t(steps), std::nullopt);
        numSteps_ = steps;
    }

    // Host noise [actionHorizon, actionDim]: uploaded at the start of every prediction.
    void setInitialNoise(const std::vector<float>& noise)
    {
        if (noise.size() != actionElements())
            throw std::runtime_error("initial noise must hold actionHorizon x actionDim values");
        noise_ = noise;
    }

    // images: device FP32 [numViews, channels, size, size]; tokens: host prompt token ids.
    // Runs prefill once and the Euler step numDenoiseSteps times on the runtime's stream,
    // then synchronizes and returns the action chunk.
    ActionChunk infer(const Tensor& images, const std::vector<int>& tokens)
    {
        if (noise_.empty())
            throw std::runtime_error("setInitialNoise() must be called before inference");
        if (images.getDataType() != migraphx_shape_float_type || images.bytes() != imageSpec_->bytes())
            throw std::runtime_error("images must be FP32 [3, 3, 224, 224]");
        if (tokens.size() != size_t(config_.numPromptTokens))
            throw std::runtime_error(
                "the prompt has " + std::to_string(tokens.size()) + " tokens, the bundle was exported for " +
                std::to_string(config_.numPromptTokens));

        bindImages(images.rawPointer());
        hostTokens_.assign(tokens.begin(), tokens.end());
        hipCheck(
            hipMemcpyAsync(
                tokensBuffer_->data,
                hostTokens_.data(),
                hostTokens_.size() * sizeof(int64_t),
                hipMemcpyHostToDevice,
                stream_),
            "hipMemcpyAsync(tokens)");
        hipCheck(
            hipMemcpyAsync(
                xBuffers_[0]->data, noise_.data(), noise_.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            "hipMemcpyAsync(noise)");

        lastPrefillOutputs_ = prefill_.run_async(prefillParameters_, stream_);
        for (int32_t index = 0; index < numSteps_; ++index)
            lastStepOutputs_[size_t(index)] = step_.run_async(stepParameters_[size_t(index)], stream_);

        ActionChunk chunk;
        chunk.normalizedActions.resize(actionElements());
        hipCheck(
            hipMemcpyAsync(
                chunk.normalizedActions.data(),
                xBuffers_[numSteps_ % 2]->data,
                actionElements() * sizeof(float),
                hipMemcpyDeviceToHost,
                stream_),
            "hipMemcpyAsync(actions)");
        hipCheck(hipStreamSynchronize(stream_), "hipStreamSynchronize");
        checkOutputAliases();
        return chunk;
    }

private:
    static migraphx::program compileWithFlags(const std::filesystem::path& path, const char* tag)
    {
        const char* flags = MIGRAPHX_GPU_HIP_FLAGS_DEFAULT;
        if (flags[0] && !std::getenv("MIGRAPHX_GPU_HIP_FLAGS"))
            setenv("MIGRAPHX_GPU_HIP_FLAGS", flags, 0);
        // MIGraphX reads these once, at its first compile; values already in the environment win.
        const bool mlir = std::string_view(MIGRAPHX_GEMM_PROVIDER_DEFAULT) != "hipblaslt";
        setenv("MIGRAPHX_DISABLE_MLIR", mlir ? "0" : "1", 0);
        setenv("MIGRAPHX_SET_GEMM_PROVIDER", "hipblaslt", 0);
        return detail::compileGraph(path, tag);
    }

    size_t actionElements() const { return size_t(config_.actionHorizon) * size_t(config_.actionDim); }

    void bindGraphs()
    {
        const size_t tokenCount = size_t(config_.numPromptTokens);
        const size_t prefixLength = size_t(config_.prefixLength);
        actionSpec_ = migraphx::shape(
            migraphx_shape_float_type, { 1, size_t(config_.actionHorizon), size_t(config_.actionDim) });
        imageSpec_ = migraphx::shape(
            migraphx_shape_float_type,
            { 1,
              size_t(config_.numViews),
              size_t(config_.imageChannels),
              size_t(config_.imageSize),
              size_t(config_.imageSize) });
        const migraphx::shape timestepSpec(migraphx_shape_float_type, { 1 });
        const migraphx::shape cacheMaskSpec(migraphx_shape_bool_type, { 1, prefixLength });
        const migraphx::shape cacheSpec(migraphx_shape_half_type, { 1, 1, prefixLength, 256 });
        const std::map<std::string, migraphx::shape> prefillInputSpecs{
            { "images", *imageSpec_ },
            { "image_masks", migraphx::shape(migraphx_shape_bool_type, { 1, size_t(config_.numViews) }) },
            { "tokens", migraphx::shape(migraphx_shape_int64_type, { 1, tokenCount }) },
            { "token_masks", migraphx::shape(migraphx_shape_bool_type, { 1, tokenCount }) },
        };
        prefillOutputSpecs_ = { cacheMaskSpec };
        std::map<std::string, migraphx::shape> stepInputSpecs{
            { "x_t", *actionSpec_ },
            { "timestep", timestepSpec },
            { "cache_mask", cacheMaskSpec },
        };
        cacheOutputIndices_ = { { "cache_mask", 0 } };
        for (int layer = 0; layer < config_.numLayers; ++layer)
            for (const char* kind : { "K", "V" })
            {
                const std::string name = std::string(kind) + std::to_string(layer);
                cacheOutputIndices_.emplace(name, prefillOutputSpecs_.size());
                prefillOutputSpecs_.push_back(cacheSpec);
                stepInputSpecs.emplace(name, cacheSpec);
            }
        prefillOutputNames_ = detail::validateContract(prefill_, prefillInputSpecs, prefillOutputSpecs_, "prefill");
        stepOutputNames_ = detail::validateContract(step_, stepInputSpecs, { *actionSpec_ }, "step");

        // Every view and every token is valid: the masks are constant, uploaded once.
        const auto prefillShapes = prefill_.get_parameter_shapes();
        imageMasksBuffer_ = std::make_unique<DeviceBuffer>(prefillShapes["image_masks"].bytes());
        tokensBuffer_ = std::make_unique<DeviceBuffer>(prefillShapes["tokens"].bytes());
        tokenMasksBuffer_ = std::make_unique<DeviceBuffer>(prefillShapes["token_masks"].bytes());
        hipCheck(hipMemset(imageMasksBuffer_->data, 1, prefillShapes["image_masks"].bytes()), "hipMemset(image_masks)");
        hipCheck(hipMemset(tokenMasksBuffer_->data, 1, prefillShapes["token_masks"].bytes()), "hipMemset(token_masks)");
        for (const auto& name : prefillOutputNames_)
            cacheBuffers_.push_back(std::make_unique<DeviceBuffer>(prefillShapes[name.c_str()].bytes()));

        const size_t actionBytes = actionSpec_->bytes();
        xBuffers_[0] = std::make_unique<DeviceBuffer>(actionBytes);
        xBuffers_[1] = std::make_unique<DeviceBuffer>(actionBytes);
    }

    // Prefill reads the caller's image tensor in place; rebind only when it changes.
    void bindImages(void* images)
    {
        if (images == boundImages_)
            return;
        const auto shapes = prefill_.get_parameter_shapes();
        migraphx::program_parameters parameters;
        parameters.add("images", migraphx::argument(shapes["images"], images));
        parameters.add("image_masks", migraphx::argument(shapes["image_masks"], imageMasksBuffer_->data));
        parameters.add("tokens", migraphx::argument(shapes["tokens"], tokensBuffer_->data));
        parameters.add("token_masks", migraphx::argument(shapes["token_masks"], tokenMasksBuffer_->data));
        for (size_t index = 0; index < prefillOutputNames_.size(); ++index)
        {
            const auto& name = prefillOutputNames_[index];
            parameters.add(name.c_str(), migraphx::argument(shapes[name.c_str()], cacheBuffers_[index]->data));
        }
        prefillParameters_ = parameters;
        boundImages_ = images;
    }

    // The compiled programs must write into the bound buffers, or the next graph reads stale data.
    void checkOutputAliases() const
    {
        if (lastPrefillOutputs_->size() != prefillOutputSpecs_.size())
            throw std::runtime_error("unsupported compiled prefill output count");
        for (size_t index = 0; index < prefillOutputSpecs_.size(); ++index)
        {
            const auto output = (*lastPrefillOutputs_)[index];
            detail::requireShape(
                output.get_shape(), prefillOutputSpecs_[index], "prefill returned output " + std::to_string(index));
            if (output.data() != cacheBuffers_[index]->data)
                throw std::runtime_error("unsupported compiled prefill output alias at index " + std::to_string(index));
        }
        for (int32_t index = 0; index < numSteps_; ++index)
        {
            const auto& outputs = *lastStepOutputs_[size_t(index)];
            if (outputs.size() != 1 || outputs[0].data() != xBuffers_[(index + 1) % 2]->data)
                throw std::runtime_error("unsupported compiled action output alias at step " + std::to_string(index));
            detail::requireShape(outputs[0].get_shape(), *actionSpec_, "step returned action output");
        }
    }

    hipStream_t stream_ = nullptr;
    detail::BundleConfig bundleConfig_;
    Pi05Config config_;
    migraphx::program prefill_;
    migraphx::program step_;

    std::optional<migraphx::shape> actionSpec_;
    std::optional<migraphx::shape> imageSpec_;
    std::vector<migraphx::shape> prefillOutputSpecs_;
    std::vector<std::string> prefillOutputNames_;
    std::vector<std::string> stepOutputNames_;
    std::map<std::string, size_t> cacheOutputIndices_;

    std::unique_ptr<DeviceBuffer> imageMasksBuffer_;
    std::unique_ptr<DeviceBuffer> tokensBuffer_;
    std::unique_ptr<DeviceBuffer> tokenMasksBuffer_;
    std::vector<std::unique_ptr<DeviceBuffer>> cacheBuffers_;
    std::unique_ptr<DeviceBuffer> xBuffers_[2];
    std::unique_ptr<DeviceBuffer> timestepBuffer_;

    void* boundImages_ = nullptr;
    migraphx::program_parameters prefillParameters_;
    std::vector<migraphx::program_parameters> stepParameters_;
    int32_t numSteps_ = 0;

    std::vector<int64_t> hostTokens_;
    std::vector<float> noise_;
    std::optional<migraphx::arguments> lastPrefillOutputs_;
    std::vector<std::optional<migraphx::arguments>> lastStepOutputs_;
};


// The counterpart of trt_edgellm::pi05::Pi05Policy: compiles the bundle in its constructor.
class Pi05Policy
{
public:
    Pi05Policy(const std::filesystem::path& bundleDir, hipStream_t stream)
    : runtime_(bundleDir, stream)
    {
    }

    Pi05Runtime& runtime() { return runtime_; }
    const Pi05Runtime& runtime() const { return runtime_; }

    void validateViewCount(int32_t views) const
    {
        if (views != runtime_.getConfig().numViews)
            throw std::runtime_error(
                "the bundle takes " + std::to_string(runtime_.getConfig().numViews) + " camera views, not " +
                std::to_string(views));
    }

    ActionChunk inferTensors(const Tensor& images, const std::vector<int>& tokens, int32_t batchSize = 1)
    {
        if (batchSize != 1)
            throw std::runtime_error("the bundle was exported for batch size 1");
        return runtime_.infer(images, tokens);
    }

private:
    Pi05Runtime runtime_;
};

}   // namespace bench::pi05

#endif   // BENCHMARKS_PI05_MIGRAPHX_NET_H
