#include "benchmark.hpp"
#include "processing.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef YOLOX_WEIGHTS_DIR
    #define YOLOX_WEIGHTS_DIR ""
#endif

namespace {

std::string tf32ArtifactTag(bool useTf32)
{
    return useTf32 ? "tf32_on" : "tf32_off";
}

bool isTf32Relevant(yolox_benchmark::Precision precision)
{
    return precision == yolox_benchmark::Precision::FP32;
}

bool effectiveTf32Mode(yolox_benchmark::Precision precision, bool useTf32)
{
    return isTf32Relevant(precision) && useTf32;
}

std::string getTensorRtEnginePath(
    Variant variant,
    yolox_benchmark::Precision precision,
    bool useTf32)
{
    std::string filename = "." + yolox_benchmark::toCliName(precision);
    if (isTf32Relevant(precision)) {
        filename += "." + tf32ArtifactTag(useTf32);
    }
    filename += ".engine";
    return yolox_benchmark::modelArtifactPath(variant, filename);
}

std::vector<float> loadInputData(const std::string& imagePath)
{
    std::cout << "\n[Input] Loading image: " << imagePath << std::endl;
    std::vector<float> prepData = yolox_benchmark::preprocessImage(imagePath);
    std::cout << "[Input] Preprocessed image: 3x640x640 (CHW)" << std::endl;
    return prepData;
}

struct TensorRtConfig {
    int gpuIndex = 0;
    yolox_benchmark::Precision precision = yolox_benchmark::Precision::FP32;
    bool useTf32 = false;
};

class TensorRtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << std::endl;
        }
    }
};

template <typename T>
struct TensorRtDeleter {
    void operator()(T* object) const
    {
        if (!object) {
            return;
        }
#if NV_TENSORRT_MAJOR >= 10
        delete object;
#else
        object->destroy();
#endif
    }
};

void throwIfCudaFailed(cudaError_t status, const char* action)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

std::vector<char> readBinaryFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open TensorRT engine file: " + path);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine file is empty: " + path);
    }

    std::vector<char> bytes(static_cast<size_t>(size));
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Failed to read TensorRT engine file: " + path);
    }
    return bytes;
}

bool hasDynamicDims(const nvinfer1::Dims& dims)
{
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] < 0) {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims makeStaticInputDims()
{
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = 1;
    dims.d[1] = static_cast<int>(yolox_benchmark::kInputChannels);
    dims.d[2] = static_cast<int>(yolox_benchmark::kInputHeight);
    dims.d[3] = static_cast<int>(yolox_benchmark::kInputWidth);
    return dims;
}

size_t dimsVolume(const nvinfer1::Dims& dims)
{
    if (dims.nbDims <= 0) {
        throw std::runtime_error("TensorRT tensor shape is empty.");
    }

    size_t volume = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error("TensorRT tensor shape contains unresolved dimensions.");
        }
        volume *= static_cast<size_t>(dims.d[index]);
    }
    return volume;
}

size_t dataTypeSize(nvinfer1::DataType type)
{
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported TensorRT tensor data type.");
    }
}

std::string dataTypeName(nvinfer1::DataType type)
{
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
        default:
            return "UNKNOWN";
    }
}

struct BufferBinding {
    std::string name;
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
    size_t bytes = 0;
    void* device = nullptr;
};

class CudaEvent
{
public:
    explicit CudaEvent(const char* label)
    {
        throwIfCudaFailed(cudaEventCreate(&event_), label);
    }

    ~CudaEvent()
    {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

double elapsedBetween(cudaEvent_t start, cudaEvent_t stop)
{
    float elapsedMs = 0.0f;
    throwIfCudaFailed(cudaEventElapsedTime(&elapsedMs, start, stop), "cudaEventElapsedTime");
    return static_cast<double>(elapsedMs);
}

class YOLOXTensorRtBenchmark
{
private:
    BenchmarkConfig benchmarkConfig_;
    TensorRtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TensorRtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtDeleter<nvinfer1::IExecutionContext>> context_;
    std::vector<BufferBinding> buffers_;
#if NV_TENSORRT_MAJOR < 10
    std::vector<void*> legacyBindings_;
#endif
    int inputBufferPosition_ = -1;
    int outputBufferPosition_ = -1;
    TensorRtConfig config_;
    cudaStream_t stream_ = nullptr;

    void releaseBuffers()
    {
        for (auto& buffer : buffers_) {
            if (buffer.device) {
                cudaFree(buffer.device);
                buffer.device = nullptr;
            }
        }
        buffers_.clear();
#if NV_TENSORRT_MAJOR < 10
        legacyBindings_.clear();
#endif
        inputBufferPosition_ = -1;
        outputBufferPosition_ = -1;
    }

    void resolveInputShape()
    {
#if NV_TENSORRT_MAJOR >= 10
        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* tensorName = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kINPUT) {
                continue;
            }

            const nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
            if (hasDynamicDims(dims) && !context_->setInputShape(tensorName, makeStaticInputDims())) {
                throw std::runtime_error("Failed to set TensorRT input shape for tensor: " + std::string(tensorName));
            }
        }
#else
        for (int index = 0; index < engine_->getNbBindings(); ++index) {
            if (!engine_->bindingIsInput(index)) {
                continue;
            }

            const nvinfer1::Dims dims = engine_->getBindingDimensions(index);
            if (hasDynamicDims(dims) && !context_->setBindingDimensions(index, makeStaticInputDims())) {
                throw std::runtime_error("Failed to set TensorRT input shape for binding index: " + std::to_string(index));
            }
        }
#endif
    }

    void allocateBuffers()
    {
        releaseBuffers();
        resolveInputShape();

#if NV_TENSORRT_MAJOR >= 10
        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* tensorName = engine_->getIOTensorName(index);
            const bool isInput = engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;
            const nvinfer1::Dims dims = context_->getTensorShape(tensorName);
            const nvinfer1::DataType type = engine_->getTensorDataType(tensorName);
            const size_t bytes = dimsVolume(dims) * dataTypeSize(type);

            buffers_.push_back({ tensorName, type, bytes, nullptr });
            const int bufferPosition = static_cast<int>(buffers_.size() - 1);
            auto& buffer = buffers_.back();
            throwIfCudaFailed(cudaMalloc(&buffer.device, buffer.bytes), "cudaMalloc");
            if (!context_->setTensorAddress(buffer.name.c_str(), buffer.device)) {
                throw std::runtime_error("Failed to bind TensorRT tensor address: " + buffer.name);
            }
            if (isInput) {
                if (inputBufferPosition_ >= 0) {
                    throw std::runtime_error("TensorRT benchmark expects a single input tensor.");
                }
                inputBufferPosition_ = bufferPosition;
            } else {
                if (outputBufferPosition_ >= 0) {
                    throw std::runtime_error("TensorRT benchmark expects a single raw output tensor.");
                }
                outputBufferPosition_ = bufferPosition;
            }
        }
#else
        legacyBindings_.assign(engine_->getNbBindings(), nullptr);
        for (int index = 0; index < engine_->getNbBindings(); ++index) {
            const bool isInput = engine_->bindingIsInput(index);
            const nvinfer1::Dims dims = context_->getBindingDimensions(index);
            const nvinfer1::DataType type = engine_->getBindingDataType(index);
            const size_t bytes = dimsVolume(dims) * dataTypeSize(type);

            buffers_.push_back({ engine_->getBindingName(index), type, bytes, nullptr });
            const int bufferPosition = static_cast<int>(buffers_.size() - 1);
            auto& buffer = buffers_.back();
            throwIfCudaFailed(cudaMalloc(&buffer.device, buffer.bytes), "cudaMalloc");
            legacyBindings_[index] = buffer.device;
            if (isInput) {
                if (inputBufferPosition_ >= 0) {
                    throw std::runtime_error("TensorRT benchmark expects a single input tensor.");
                }
                inputBufferPosition_ = bufferPosition;
            } else {
                if (outputBufferPosition_ >= 0) {
                    throw std::runtime_error("TensorRT benchmark expects a single raw output tensor.");
                }
                outputBufferPosition_ = bufferPosition;
            }
        }
#endif

        if (inputBufferPosition_ < 0) {
            throw std::runtime_error("TensorRT engine has no input tensor.");
        }
        if (outputBufferPosition_ < 0) {
            throw std::runtime_error("TensorRT engine has no output tensor.");
        }
    }

public:
    YOLOXTensorRtBenchmark(BenchmarkConfig config, TensorRtConfig runtimeConfig)
        : benchmarkConfig_(std::move(config))
        , config_(std::move(runtimeConfig))
    {
    }

    ~YOLOXTensorRtBenchmark()
    {
        releaseBuffers();
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    void prepare(const std::string& weightsPath)
    {
        throwIfCudaFailed(cudaSetDevice(config_.gpuIndex), "cudaSetDevice");
        if (!stream_) {
            throwIfCudaFailed(cudaStreamCreate(&stream_), "cudaStreamCreate");
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        const auto engineBytes = readBinaryFile(weightsPath);
        engine_.reset(runtime_->deserializeCudaEngine(engineBytes.data(), engineBytes.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine: " + weightsPath);
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }
        allocateBuffers();
        const BufferBinding& inputBuffer = buffers_[inputBufferPosition_];
        std::cout << "[TensorRT] Engine loaded from: " << weightsPath << std::endl;
        std::cout << "[TensorRT] GPU: " << config_.gpuIndex
                  << ", Requested precision: " << yolox_benchmark::precisionDisplayName(config_.precision)
                  << ", Input dtype: " << dataTypeName(inputBuffer.type) << std::endl;
    }

    void uploadInput(const std::vector<float>& inputData)
    {
        if (inputBufferPosition_ < 0) {
            throw std::runtime_error("TensorRT input binding is not initialized.");
        }

        if (inputData.size() != yolox_benchmark::kInputSize) {
            throw std::runtime_error("Expected 3x640x640 input tensor.");
        }

        const BufferBinding& inputBuffer = buffers_[inputBufferPosition_];
        const size_t expectedBytes = inputData.size() * dataTypeSize(inputBuffer.type);
        if (inputBuffer.type != nvinfer1::DataType::kFLOAT && inputBuffer.type != nvinfer1::DataType::kHALF) {
            throw std::runtime_error("TensorRT benchmark expects an FP32 or FP16 input tensor.");
        }
        if (inputBuffer.bytes != expectedBytes) {
            throw std::runtime_error(
                "TensorRT engine input shape does not match the benchmark's fixed 1x3x640x640 input.");
        }

        std::vector<uint16_t> halfInput;
        const void* hostInput = inputData.data();
        if (inputBuffer.type == nvinfer1::DataType::kHALF) {
            halfInput = yolox_benchmark::floatsToHalfBits(inputData);
            hostInput = halfInput.data();
        }
        throwIfCudaFailed(
            cudaMemcpy(
                inputBuffer.device,
                hostInput,
                inputBuffer.bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(input)");
    }

    void enqueueOnly()
    {
#if NV_TENSORRT_MAJOR >= 10
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed.");
        }
#else
        if (!context_->enqueueV2(legacyBindings_.data(), stream_, nullptr)) {
            throw std::runtime_error("TensorRT enqueueV2 failed.");
        }
#endif
    }

    std::vector<float> downloadOutput()
    {
        throwIfCudaFailed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(output download)");

        if (outputBufferPosition_ < 0) {
            throw std::runtime_error("TensorRT engine has no output tensor.");
        }
        const BufferBinding& outputBuffer = buffers_[outputBufferPosition_];
        if (outputBuffer.type != nvinfer1::DataType::kFLOAT && outputBuffer.type != nvinfer1::DataType::kHALF) {
            throw std::runtime_error("TensorRT benchmark expects an FP32 or FP16 raw output tensor.");
        }
        const size_t elementCount = outputBuffer.bytes / dataTypeSize(outputBuffer.type);
        std::vector<float> output;
        if (outputBuffer.type == nvinfer1::DataType::kFLOAT) {
            output.resize(elementCount);
            throwIfCudaFailed(
                cudaMemcpy(
                    output.data(),
                    outputBuffer.device,
                    outputBuffer.bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(output)");
        } else {
            std::vector<uint16_t> halfOutput(elementCount);
            throwIfCudaFailed(
                cudaMemcpy(
                    halfOutput.data(),
                    outputBuffer.device,
                    outputBuffer.bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(output)");
            output = yolox_benchmark::halfBitsToFloats(halfOutput.data(), elementCount);
        }
        yolox_benchmark::validateModelOutputSize(output.size(), "TensorRT");
        return output;
    }

    void ensurePrepared() const
    {
        if (!context_ || inputBufferPosition_ < 0 || outputBufferPosition_ < 0) {
            throw std::runtime_error("TensorRT benchmark is not prepared. Call prepare() first.");
        }
    }

    BenchmarkResult runPureInference(const std::vector<float>& inputData)
    {
        ensurePrepared();
        uploadInput(inputData);

        const BenchmarkConfig& benchmarkConfig = benchmarkConfig_;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Measurement: TensorRT enqueue including decode; no H2D/D2H or NMS" << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Warming up... " << std::flush;
        for (int index = 0; index < benchmarkConfig.warmup; ++index) {
            enqueueOnly();
        }
        throwIfCudaFailed(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(warmup)");
        std::cout << "Done." << std::endl;

        std::cout << "[" << benchmarkConfig.runtime_name << "] Running "
                  << benchmarkConfig.iterations << " iterations..." << std::endl;

        CudaEvent start("cudaEventCreate(start)");
        CudaEvent stop("cudaEventCreate(stop)");

        std::vector<double> elapsedTimes;
        elapsedTimes.reserve(benchmarkConfig.iterations);
        for (int index = 0; index < benchmarkConfig.iterations; ++index) {
            throwIfCudaFailed(cudaEventRecord(start.get(), stream_), "cudaEventRecord(start)");
            enqueueOnly();
            throwIfCudaFailed(cudaEventRecord(stop.get(), stream_), "cudaEventRecord(stop)");
            throwIfCudaFailed(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");
            elapsedTimes.emplace_back(elapsedBetween(start.get(), stop.get()));
        }

        return yolox_benchmark::makeBenchmarkResult(benchmarkConfig.runtime_name, elapsedTimes);
    }
};

bool parseOnOff(const std::string& value, const std::string& flagName)
{
    if (value == "on" || value == "1" || value == "true") {
        return true;
    }
    if (value == "off" || value == "0" || value == "false") {
        return false;
    }
    throw std::invalid_argument("Unsupported " + flagName + " value: " + value);
}

std::vector<Detection> tensorRtBenchmark(
    const std::vector<float>& inputData,
    Variant variant,
    int warmup,
    int iterations,
    const TensorRtConfig& runtimeConfig,
    const std::string& weightsPath)
{
    const std::string variantName = variantToString(variant);
    std::cout << "\n=== TensorRT " << yolox_benchmark::precisionDisplayName(runtimeConfig.precision)
              << " YOLOX-" << variantName << " Benchmark ===" << std::endl;

    std::string benchName = std::string("TensorRT_")
        + yolox_benchmark::precisionDisplayName(runtimeConfig.precision) + "_";
    if (isTf32Relevant(runtimeConfig.precision)) {
        benchName += runtimeConfig.useTf32 ? "TF32_ON_" : "TF32_OFF_";
    }
    benchName += "YOLOX_" + variantName;
    BenchmarkConfig benchmarkConfig(warmup, iterations, benchName);
    YOLOXTensorRtBenchmark benchmark(benchmarkConfig, runtimeConfig);
    benchmark.prepare(weightsPath);

    BenchmarkResult result = benchmark.runPureInference(inputData);
    result.print();
    return yolox_benchmark::postprocessOutput(benchmark.downloadOutput());
}

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --size S        Model size: s, m, l, x (default: l)\n"
              << "  --gpu N         CUDA device index (default: 0)\n"
              << "  --warmup N      Number of warmup iterations (default: "
              << BenchmarkConfig::kDefaultWarmup << ")\n"
              << "  --iterations N  Number of benchmark iterations (default: "
              << BenchmarkConfig::kDefaultIterations << ")\n"
              << "  --tf32 MODE     CUDA/TensorRT TF32 mode: on, off (default: off)\n"
              << "  --precision P   Precision: fp32, fp16 (default: fp32)\n"
              << "  --image PATH    Path to input image (default: "
              << yolox_benchmark::defaultImagePath() << ")\n"
              << "  --help          Show this help message\n";
}

} // namespace

int main(int argc, char** argv)
{
    yolox_benchmark::CommonCliOptions options;
    TensorRtConfig runtimeConfig;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (argument == "--gpu") {
                runtimeConfig.gpuIndex = yolox_benchmark::parseIntegerOption(
                    yolox_benchmark::takeOptionValue(index, argc, argv, argument),
                    argument);
                continue;
            }
            if (argument == "--tf32") {
                runtimeConfig.useTf32 = parseOnOff(
                    yolox_benchmark::takeOptionValue(index, argc, argv, argument),
                    argument);
                continue;
            }
            if (!yolox_benchmark::parseCommonOption(
                    argument, index, argc, argv, options)) {
                throw std::invalid_argument("Unknown argument: " + argument);
            }
        }

        yolox_benchmark::validateCommonCliOptions(options);
        if (runtimeConfig.gpuIndex < 0) {
            throw std::invalid_argument("--gpu must be greater than or equal to zero.");
        }
        runtimeConfig.precision = options.precision;
        const std::string weightsPath = getTensorRtEnginePath(
            options.variant,
            runtimeConfig.precision,
            effectiveTf32Mode(runtimeConfig.precision, runtimeConfig.useTf32));
        std::cout << "========================================" << std::endl;
        std::cout << "  YOLOX TensorRT Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Variant: YOLOX-" << variantToString(options.variant) << std::endl;
        std::cout << "GPU    : " << runtimeConfig.gpuIndex << std::endl;
        std::cout << "Precision: " << yolox_benchmark::toCliName(runtimeConfig.precision) << std::endl;
        if (isTf32Relevant(runtimeConfig.precision)) {
            std::cout << "TF32   : " << (runtimeConfig.useTf32 ? "on" : "off") << std::endl;
        } else {
            std::cout << "TF32   : ignored for fp16" << std::endl;
        }
        std::cout << "Warmup : " << options.warmup
                  << ", Iterations: " << options.iterations << std::endl;
        std::cout << "Weights: " << weightsPath << std::endl;
        std::cout << "Input  : " << options.imagePath << std::endl;

        auto inputData = loadInputData(options.imagePath);
        const std::vector<Detection> detections = tensorRtBenchmark(
            inputData,
            options.variant,
            options.warmup,
            options.iterations,
            runtimeConfig,
            weightsPath);
        yolox_benchmark::printDetectionSummary(detections);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TensorRT] " << error.what() << std::endl;
        return 1;
    }
}
