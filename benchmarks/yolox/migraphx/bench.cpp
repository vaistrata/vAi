#include "benchmark.hpp"
#include "processing.hpp"

#include <migraphx/migraphx.hpp>

#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
    #include <hip/hip_runtime_api.h>
#endif

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef MIGRAPHX_GPU_HIP_FLAGS_DEFAULT
    #define MIGRAPHX_GPU_HIP_FLAGS_DEFAULT ""
#endif

#ifndef YOLOX_WEIGHTS_DIR
    #define YOLOX_WEIGHTS_DIR ""
#endif

namespace {

#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
void throwIfMigraphxHipFailed(hipError_t status, const char* action)
{
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(action) + ": " + hipGetErrorString(status));
    }
}
#endif


struct MigraphxConfig
{
    yolox_benchmark::Precision precision = yolox_benchmark::Precision::FP32;
    int gpuIndex = 0;
};

std::string getMigraphxOnnxPath(Variant variant, yolox_benchmark::Precision precision)
{
    return yolox_benchmark::modelArtifactPath(
        variant, "." + yolox_benchmark::toCliName(precision) + ".onnx");
}

std::vector<float> loadInputData(const std::string& imagePath)
{
    std::cout << "\n[Input] Loading image: " << imagePath << std::endl;
    std::vector<float> prepData = yolox_benchmark::preprocessImage(imagePath);
    std::cout << "[Input] Preprocessed image: 3x640x640 (CHW)" << std::endl;
    return prepData;
}

migraphx_shape_datatype_t migraphxTypeForPrecision(yolox_benchmark::Precision precision)
{
    return precision == yolox_benchmark::Precision::FP16
        ? migraphx_shape_half_type
        : migraphx_shape_float_type;
}

void validatePrecisionShape(
    const migraphx::shape& shape,
    const std::string& label,
    yolox_benchmark::Precision precision)
{
    if (shape.type() != migraphxTypeForPrecision(precision)) {
        throw std::runtime_error(
            "MIGraphX " + label + " tensor must be "
            + yolox_benchmark::precisionDisplayName(precision) + ".");
    }
}

std::string formatShapeElements(const migraphx::shape& shape)
{
    std::string formatted = "[";
    const auto lengths = shape.lengths();
    for (size_t index = 0; index < lengths.size(); ++index) {
        if (index != 0) {
            formatted += "x";
        }
        formatted += std::to_string(lengths[index]);
    }
    formatted += "] ";
    formatted += std::to_string(shape.elements());
    formatted += " elements";
    formatted += ", strides=[";
    const auto strides = shape.strides();
    for (size_t index = 0; index < strides.size(); ++index) {
        if (index != 0) {
            formatted += ",";
        }
        formatted += std::to_string(strides[index]);
    }
    formatted += "]";
    return formatted;
}

template <typename T>
std::vector<T> materializeLogicalOutput(
    const std::vector<T>& physicalOutput,
    const migraphx::shape& shape)
{
    std::vector<T> logicalOutput(shape.elements());
    for (size_t index = 0; index < logicalOutput.size(); ++index) {
        const size_t physicalIndex = shape.index(index);
        if (physicalIndex >= physicalOutput.size()) {
            throw std::runtime_error(
                "MIGraphX output stride points outside the downloaded buffer.");
        }
        logicalOutput[index] = physicalOutput[physicalIndex];
    }
    return logicalOutput;
}

std::string makeBenchName(
    const MigraphxConfig& runtimeConfig,
    Variant variant)
{
    std::string name = "MIGraphX_GPU";
    name += "_" + yolox_benchmark::precisionDisplayName(runtimeConfig.precision);
    name += "_YOLOX_" + variantToString(variant) + "_inference";
    return name;
}

class YOLOXMigraphxBenchmark
{
private:
    BenchmarkConfig benchmarkConfig_;
    MigraphxConfig runtimeConfig_;
    migraphx::program program_;
    migraphx::shape inputShape_;
    migraphx::shape outputShape_;
    migraphx::shape outputParameterShape_;
    std::string inputName_;
    std::string outputParameterName_;
    std::vector<float> lastOutput_;
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
    void* deviceInput_ = nullptr;
    void* deviceOutput_ = nullptr;
    size_t deviceInputBytes_ = 0;
    size_t deviceOutputBytes_ = 0;
    hipStream_t stream_ = nullptr;
    hipEvent_t startEvent_ = nullptr;
    hipEvent_t stopEvent_ = nullptr;
    migraphx::argument deviceInputArgument_;
    migraphx::argument deviceOutputArgument_;
    migraphx::program_parameters deviceParameters_;
    std::optional<migraphx::arguments> lastDeviceOutputs_;
#endif

public:
    YOLOXMigraphxBenchmark(BenchmarkConfig config, MigraphxConfig runtimeConfig)
        : benchmarkConfig_(std::move(config))
        , runtimeConfig_(std::move(runtimeConfig))
    {
    }

    ~YOLOXMigraphxBenchmark()
    {
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
        if (startEvent_) {
            (void)hipEventDestroy(startEvent_);
        }
        if (stopEvent_) {
            (void)hipEventDestroy(stopEvent_);
        }
        if (stream_) {
            (void)hipStreamDestroy(stream_);
        }
        if (deviceInput_) {
            (void)hipFree(deviceInput_);
        }
        if (deviceOutput_) {
            (void)hipFree(deviceOutput_);
        }
#endif
    }

    void prepare(const std::string& modelPath)
    {
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
        throwIfMigraphxHipFailed(
            hipSetDevice(runtimeConfig_.gpuIndex), "hipSetDevice");
#endif
        migraphx::onnx_options onnxOptions;
        program_ = migraphx::parse_onnx(modelPath.c_str(), onnxOptions);

        migraphx::compile_options compileOptions;
        program_.compile(migraphx::target("gpu"), compileOptions);

        const auto parameterShapes = program_.get_parameter_shapes();
        const auto parameterNames = parameterShapes.names();
        if (parameterNames.empty()) {
            throw std::runtime_error("MIGraphX model has no input parameters.");
        }

        std::vector<std::string> parameterSummary;
        size_t outputParameterCount = 0;
        bool foundInput = false;
        for (const auto* rawParameterName : parameterNames) {
            const std::string parameterName(rawParameterName ? rawParameterName : "");
            const auto shape = parameterShapes[parameterName.c_str()];
            parameterSummary.push_back(parameterName + "=" + formatShapeElements(shape));
            if (!foundInput && shape.type() == migraphxTypeForPrecision(runtimeConfig_.precision) && shape.elements() == yolox_benchmark::kInputSize) {
                inputName_ = parameterName;
                inputShape_ = shape;
                foundInput = true;
            } else if (parameterName.find("#output_") != std::string::npos) {
                outputParameterName_ = parameterName;
                outputParameterShape_ = shape;
                ++outputParameterCount;
            }
        }
        if (!foundInput) {
            std::string message = "MIGraphX model has no 3x640x640 "
                + yolox_benchmark::precisionDisplayName(runtimeConfig_.precision)
                + " input parameter.";
            for (const auto& parameter : parameterSummary) {
                message += "\n - " + parameter;
            }
            throw std::runtime_error(message);
        }
        validatePrecisionShape(inputShape_, "input", runtimeConfig_.precision);
        if (outputParameterCount != 1) {
            throw std::runtime_error(
                "MIGraphX pure inference expected exactly one #output_ "
                "parameter, found " + std::to_string(outputParameterCount) + ".");
        }
        const auto outputShapes = program_.get_output_shapes();
        if (outputShapes.size() == 0) {
            throw std::runtime_error("MIGraphX model has no outputs.");
        }
        outputShape_ = outputShapes[0];
        validatePrecisionShape(outputShape_, "output", runtimeConfig_.precision);
        if (outputShape_.elements() != yolox_benchmark::kOutputSize) {
            throw std::runtime_error(
                "MIGraphX output contract mismatch. Expected raw [8400,85] ("
                + std::to_string(yolox_benchmark::kOutputSize) + " floats), got "
                + std::to_string(outputShape_.elements()) + " floats.");
        }
        validatePrecisionShape(
            outputParameterShape_, "output parameter", runtimeConfig_.precision);
        if (outputParameterShape_.elements() != outputShape_.elements()) {
            throw std::runtime_error(
                "MIGraphX output parameter does not match graph output size.");
        }
        if (inputShape_.elements() != yolox_benchmark::kInputSize) {
            throw std::runtime_error(
                "MIGraphX input contract mismatch. Expected 3x640x640 ("
                + std::to_string(yolox_benchmark::kInputSize) + " floats), got "
                + std::to_string(inputShape_.elements()) + " floats.");
        }

        std::cout << "[MIGraphX] Model loaded from: " << modelPath << std::endl;
        std::cout << "[MIGraphX] Target: GPU"
                  << ", Offload copy: off"
                  << ", Precision: " << yolox_benchmark::toCliName(runtimeConfig_.precision)
                  << std::endl;
        std::cout << "[MIGraphX] Input: " << inputName_ << std::endl;
        std::cout << "[MIGraphX] Parameters:" << std::endl;
        for (const auto& parameter : parameterSummary) {
            std::cout << " - " << parameter << std::endl;
        }
        std::cout << "[MIGraphX] Output: "
                  << formatShapeElements(outputShape_)
                  << ", standard=" << (outputShape_.standard() ? "yes" : "no")
                  << std::endl;
        std::cout << "[MIGraphX] Bound output: "
                  << outputParameterName_ << " "
                  << formatShapeElements(outputParameterShape_) << std::endl;
    }

#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
    void ensureHipResources(const std::vector<float>& input)
    {
        if (input.size() != yolox_benchmark::kInputSize) {
            throw std::runtime_error(
                "MIGraphX input size mismatch. Expected "
                + std::to_string(yolox_benchmark::kInputSize) + " floats, got "
                + std::to_string(input.size()) + " floats.");
        }

        if (!stream_) {
            throwIfMigraphxHipFailed(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        }
        if (!startEvent_) {
            throwIfMigraphxHipFailed(hipEventCreate(&startEvent_), "hipEventCreate(start)");
        }
        if (!stopEvent_) {
            throwIfMigraphxHipFailed(hipEventCreate(&stopEvent_), "hipEventCreate(stop)");
        }

        const size_t inputBytes = yolox_benchmark::kInputSize
            * yolox_benchmark::precisionElementSize(runtimeConfig_.precision);
        if (!deviceInput_) {
            throwIfMigraphxHipFailed(hipMalloc(&deviceInput_, inputBytes), "hipMalloc(input)");
            deviceInputBytes_ = inputBytes;
            deviceInputArgument_ = migraphx::argument(inputShape_, deviceInput_);
            deviceParameters_ = migraphx::program_parameters{};
            deviceParameters_.add(inputName_.c_str(), deviceInputArgument_);
            const size_t outputBytes = outputParameterShape_.bytes();
            throwIfMigraphxHipFailed(
                hipMalloc(&deviceOutput_, outputBytes), "hipMalloc(output)");
            deviceOutputBytes_ = outputBytes;
            deviceOutputArgument_ = migraphx::argument(
                outputParameterShape_, deviceOutput_);
            deviceParameters_.add(
                outputParameterName_.c_str(), deviceOutputArgument_);
        }

        std::vector<uint16_t> halfInput;
        const void* inputSource = input.data();
        if (runtimeConfig_.precision == yolox_benchmark::Precision::FP16) {
            halfInput = yolox_benchmark::floatsToHalfBits(input);
            inputSource = halfInput.data();
        }
        throwIfMigraphxHipFailed(
            hipMemcpy(deviceInput_, inputSource, deviceInputBytes_, hipMemcpyHostToDevice),
            "hipMemcpy(input)");
        throwIfMigraphxHipFailed(hipDeviceSynchronize(), "hipDeviceSynchronize(input)");
    }

    migraphx::arguments runAsyncOnStream()
    {
        return program_.run_async(deviceParameters_, stream_);
    }

    BenchmarkResult runPureInference(std::vector<float>& input)
    {
        ensureHipResources(input);

        const BenchmarkConfig& benchmarkConfig = benchmarkConfig_;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Measurement: MIGraphX run_async on HIP stream including decode; no H2D/D2H or NMS" << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Timing: HIP event elapsed time on the benchmark stream" << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Warming up... " << std::flush;
        for (int index = 0; index < benchmarkConfig.warmup; ++index) {
            auto outputs = runAsyncOnStream();
            (void)outputs;
            throwIfMigraphxHipFailed(hipStreamSynchronize(stream_), "hipStreamSynchronize(warmup)");
        }
        std::cout << "Done." << std::endl;

        std::cout << "[" << benchmarkConfig.runtime_name << "] Running "
                  << benchmarkConfig.iterations << " iterations..." << std::endl;

        std::vector<double> elapsedTimes;
        elapsedTimes.reserve(benchmarkConfig.iterations);
        for (int index = 0; index < benchmarkConfig.iterations; ++index) {
            throwIfMigraphxHipFailed(hipEventRecord(startEvent_, stream_), "hipEventRecord(start)");
            auto outputs = runAsyncOnStream();
            throwIfMigraphxHipFailed(hipEventRecord(stopEvent_, stream_), "hipEventRecord(stop)");
            throwIfMigraphxHipFailed(hipEventSynchronize(stopEvent_), "hipEventSynchronize(stop)");
            float elapsedMs = 0.0f;
            throwIfMigraphxHipFailed(hipEventElapsedTime(&elapsedMs, startEvent_, stopEvent_), "hipEventElapsedTime");
            elapsedTimes.emplace_back(static_cast<double>(elapsedMs));
            (void)outputs;
        }

        lastDeviceOutputs_ = runAsyncOnStream();
        throwIfMigraphxHipFailed(hipStreamSynchronize(stream_), "hipStreamSynchronize(final output)");
        return yolox_benchmark::makeBenchmarkResult(benchmarkConfig.runtime_name, elapsedTimes);
    }

    void downloadDeviceOutput()
    {
        if (!lastDeviceOutputs_ || lastDeviceOutputs_->size() == 0) {
            throw std::runtime_error("MIGraphX device output is not initialized.");
        }
        const auto output = (*lastDeviceOutputs_)[0];
        const auto outputShape = output.get_shape();
        validatePrecisionShape(outputShape, "output", runtimeConfig_.precision);
        yolox_benchmark::validateModelOutputSize(outputShape.elements(), "MIGraphX");
        if (!deviceOutput_ || deviceOutputBytes_ != outputParameterShape_.bytes()) {
            throw std::runtime_error(
                "MIGraphX bound output buffer is not initialized.");
        }
        if (output.data() != deviceOutput_) {
            std::cout << "[MIGraphX] Returned output uses a different view; "
                         "downloading the explicitly bound #output_ buffer."
                      << std::endl;
        }
        const auto& boundShape = outputParameterShape_;
        if (runtimeConfig_.precision == yolox_benchmark::Precision::FP16) {
            const size_t physicalElements = boundShape.bytes() / sizeof(uint16_t);
            std::vector<uint16_t> physicalOutput(physicalElements);
            throwIfMigraphxHipFailed(
                hipMemcpy(physicalOutput.data(), deviceOutput_, boundShape.bytes(), hipMemcpyDeviceToHost),
                "hipMemcpy(output)");
            const std::vector<uint16_t> logicalOutput =
                materializeLogicalOutput(physicalOutput, boundShape);
            lastOutput_ = yolox_benchmark::halfBitsToFloats(
                logicalOutput.data(), logicalOutput.size());
        } else {
            const size_t physicalElements = boundShape.bytes() / sizeof(float);
            std::vector<float> physicalOutput(physicalElements);
            throwIfMigraphxHipFailed(
                hipMemcpy(physicalOutput.data(), deviceOutput_, boundShape.bytes(), hipMemcpyDeviceToHost),
                "hipMemcpy(output)");
            lastOutput_ = materializeLogicalOutput(physicalOutput, boundShape);
        }
    }
#endif

    const std::vector<float>& lastOutput() const
    {
        return lastOutput_;
    }
};

void configureGpuHipFlags()
{
    constexpr const char* defaultHipFlags = MIGRAPHX_GPU_HIP_FLAGS_DEFAULT;
    if (defaultHipFlags[0] == '\0' || std::getenv("MIGRAPHX_GPU_HIP_FLAGS") != nullptr) {
        return;
    }

#if defined(_WIN32)
    _putenv_s("MIGRAPHX_GPU_HIP_FLAGS", defaultHipFlags);
#else
    setenv("MIGRAPHX_GPU_HIP_FLAGS", defaultHipFlags, 0);
#endif
}

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --size S          Model size: s, m, l, x (default: l)\n"
              << "  --gpu N           Requested GPU index (default: 0; use HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES)\n"
              << "  --warmup N        Number of warmup iterations (default: "
              << yolox_benchmark::kDefaultWarmup << ")\n"
              << "  --iterations N    Number of benchmark iterations (default: "
              << yolox_benchmark::kDefaultIterations << ")\n"
              << "  --precision P     Precision: fp32, fp16 (default: fp32)\n"
              << "  --image PATH      Image input (default: "
              << yolox_benchmark::defaultImagePath() << ")\n"
              << "  --help            Show this help message\n";
}

BenchmarkResult migraphxBenchmark(
    std::vector<float>& inputData,
    Variant variant,
    int warmup,
    int iterations,
    const MigraphxConfig& runtimeConfig,
    const std::string& modelPath,
    std::vector<Detection>& lastDetections)
{
    const std::string variantName = variantToString(variant);
    std::cout << "\n=== MIGraphX GPU YOLOX-" << variantName << " Benchmark ===" << std::endl;

    const std::string benchName = makeBenchName(runtimeConfig, variant);
    BenchmarkConfig benchmarkConfig(warmup, iterations, benchName);
    YOLOXMigraphxBenchmark benchmark(benchmarkConfig, runtimeConfig);
    benchmark.prepare(modelPath);

    BenchmarkResult result;
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
    result = benchmark.runPureInference(inputData);
    benchmark.downloadDeviceOutput();
    lastDetections = yolox_benchmark::postprocessOutput(benchmark.lastOutput());
#else
    throw std::runtime_error("MIGraphX pure inference requires HIP runtime support at build time.");
#endif
    result.print();
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    yolox_benchmark::CommonCliOptions commonOptions;
    MigraphxConfig runtimeConfig;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (yolox_benchmark::parseCommonOption(argument, index, argc, argv, commonOptions)) {
                continue;
            }
            if (argument == "--gpu") {
                runtimeConfig.gpuIndex = yolox_benchmark::parseIntegerOption(
                    yolox_benchmark::takeOptionValue(index, argc, argv, argument),
                    argument);
                continue;
            }
            std::cerr << "Unknown argument: " << argument << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    try {
        yolox_benchmark::validateCommonCliOptions(commonOptions);
        if (runtimeConfig.gpuIndex < 0) {
            throw std::invalid_argument("--gpu must be greater than or equal to zero.");
        }
        const Variant variant = commonOptions.variant;
        const int warmup = commonOptions.warmup;
        const int iterations = commonOptions.iterations;
        runtimeConfig.precision = commonOptions.precision;
        const std::string& imagePath = commonOptions.imagePath;
        const std::string modelPath = getMigraphxOnnxPath(variant, runtimeConfig.precision);
        configureGpuHipFlags();

        std::cout << "========================================" << std::endl;
        std::cout << "  YOLOX MIGraphX Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Variant      : YOLOX-" << variantToString(variant) << std::endl;
        std::cout << "Target       : GPU" << std::endl;
        std::cout << "GPU          : " << runtimeConfig.gpuIndex << std::endl;
        std::cout << "Precision    : " << yolox_benchmark::toCliName(runtimeConfig.precision) << std::endl;
        std::cout << "Offload copy : off" << std::endl;
        std::cout << "Warmup       : " << warmup << ", Iterations: " << iterations << std::endl;
        std::cout << "Model        : " << modelPath << std::endl;
        std::cout << "Input        : " << imagePath << std::endl;

        auto inputData = loadInputData(imagePath);
        std::vector<Detection> detections;
        migraphxBenchmark(
            inputData,
            variant,
            warmup,
            iterations,
            runtimeConfig,
            modelPath,
            detections);
        yolox_benchmark::printDetectionSummary(detections);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[MIGraphX] " << error.what() << std::endl;
        return 1;
    }
}
