#include "benchmark.hpp"
#include "processing.hpp"
#include "onnxruntime_cxx_api.h"

#if defined(VAI_YOLOX_HAVE_CUDA_RUNTIME)
    #include <cuda_runtime_api.h>
#endif


#if defined(_WIN32)
    #include <windows.h>
#endif

#include <cstdlib>
#include <cstdint>
#include <iomanip>
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

constexpr uint32_t kAmdPciVendorId = 0x1002;

#if defined(VAI_YOLOX_HAVE_CUDA_RUNTIME)
void throwIfOnnxCudaFailed(cudaError_t status, const char* action)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}
#endif

} // namespace

#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
namespace yolox_onnx_hip {
void setDevice(int device);
void* mallocDevice(size_t bytes, const char* action);
void freeDevice(void* ptr) noexcept;
void copyHostToDevice(void* dst, const void* src, size_t bytes, const char* action);
void copyDeviceToHost(void* dst, const void* src, size_t bytes, const char* action);
void deviceSynchronize(const char* action);
} // namespace yolox_onnx_hip
#endif

namespace {

std::string getOnnxPath(Variant variant, yolox_benchmark::Precision precision) {
    return yolox_benchmark::modelArtifactPath(
        variant, "." + yolox_benchmark::toCliName(precision) + ".onnx");
}

enum class OnnxEp {
    CUDA,
    ROCm
};

struct OnnxEpConfig {
    OnnxEp ep = OnnxEp::CUDA;
    int cuda_device = 0;
    yolox_benchmark::Precision precision = yolox_benchmark::Precision::FP32;
};

const char* toProviderName(OnnxEp ep)
{
    switch (ep) {
        case OnnxEp::CUDA: return "CUDAExecutionProvider";
        case OnnxEp::ROCm: return "MIGraphXExecutionProvider";
        default:           return "UNKNOWN";
    }
}

std::string toCliName(OnnxEp ep)
{
    switch (ep) {
        case OnnxEp::CUDA: return "cuda";
        case OnnxEp::ROCm: return "rocm";
        default:           return "unknown";
    }
}

bool hasAvailableProvider(const std::vector<std::string>& availableProviders, const char* providerName)
{
    for (const auto& provider : availableProviders) {
        if (provider == providerName) {
            return true;
        }
    }
    return false;
}

std::string formatProviderLabel(OnnxEp ep)
{
    if (ep == OnnxEp::CUDA) {
        return "CUDA";
    }
    if (ep == OnnxEp::ROCm) {
        return "ROCm";
    }
    return "UNKNOWN";
}

std::vector<const char*> buildRawStringView(const std::vector<std::string>& values)
{
    std::vector<const char*> raw;
    raw.reserve(values.size());
    for (const auto& value : values) {
        raw.push_back(value.c_str());
    }
    return raw;
}

ONNXTensorElementDataType onnxTensorElementType(yolox_benchmark::Precision precision)
{
    return precision == yolox_benchmark::Precision::FP16
        ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

void appendExecutionProviderByName(
    Ort::SessionOptions& sessionOptions,
    const char* providerName,
    const std::vector<std::string>& optionKeys,
    const std::vector<std::string>& optionValues)
{
    const auto rawKeys = buildRawStringView(optionKeys);
    const auto rawValues = buildRawStringView(optionValues);
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(
        static_cast<OrtSessionOptions*>(sessionOptions),
        providerName,
        rawKeys.empty() ? nullptr : rawKeys.data(),
        rawValues.empty() ? nullptr : rawValues.data(),
        rawKeys.size()));
}

void configureProviderOptions(
    Ort::SessionOptions& sessionOptions,
    OnnxEp ep,
    const OnnxEpConfig& config)
{
    if (ep == OnnxEp::CUDA) {
        Ort::CUDAProviderOptions cudaOptions;
        cudaOptions.Update({
            { "device_id", std::to_string(config.cuda_device) },
            { "use_tf32", "0" },
        });
        sessionOptions.AppendExecutionProvider_CUDA_V2(*cudaOptions);
        return;
    }

    if (ep == OnnxEp::ROCm) {
        appendExecutionProviderByName(
            sessionOptions,
            "MIGraphX",
            {
                "device_id",
            },
            {
                std::to_string(config.cuda_device),
            });
        return;
    }
}

#if defined(_WIN32)
std::wstring toWidePath(const std::string& path)
{
    if (path.empty()) {
        return L"";
    }

    const int length = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (length <= 0) {
        throw std::runtime_error("Failed to convert model path to UTF-16.");
    }

    std::wstring widePath(length, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, widePath.data(), length);
    if (!widePath.empty() && widePath.back() == L'\0') {
        widePath.pop_back();
    }
    return widePath;
}
#endif

//==============================================================================
// ONNX Runtime YOLOX Benchmark Class
//==============================================================================
class YOLOXOnnxBenchmark
{
private:
    BenchmarkConfig benchmarkConfig_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions sessionOptions_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    OnnxEpConfig epConfig_;
    std::vector<float> lastOutput_;
#if defined(VAI_YOLOX_HAVE_CUDA_RUNTIME)
    void* cudaInput_ = nullptr;
    void* cudaOutput_ = nullptr;
    size_t cudaInputBytes_ = 0;
    size_t cudaOutputBytes_ = 0;
#endif
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
    void* hipInput_ = nullptr;
    void* hipOutput_ = nullptr;
    size_t hipInputBytes_ = 0;
    size_t hipOutputBytes_ = 0;
#endif

public:
    YOLOXOnnxBenchmark(BenchmarkConfig config, OnnxEpConfig epConfig)
        : benchmarkConfig_(std::move(config))
        , env_(ORT_LOGGING_LEVEL_WARNING, "YOLOX")
        , epConfig_(std::move(epConfig)) {}

    ~YOLOXOnnxBenchmark()
    {
#if defined(VAI_YOLOX_HAVE_CUDA_RUNTIME)
        if (cudaInput_) {
            cudaFree(cudaInput_);
        }
        if (cudaOutput_) {
            cudaFree(cudaOutput_);
        }
#endif
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
        if (hipInput_) {
            yolox_onnx_hip::freeDevice(hipInput_);
        }
        if (hipOutput_) {
            yolox_onnx_hip::freeDevice(hipOutput_);
        }
#endif
    }

    void prepare(const std::string& modelPath)
    {
        const auto availableProviders = Ort::GetAvailableProviders();
        const char* providerName = toProviderName(epConfig_.ep);
        if (!hasAvailableProvider(availableProviders, providerName)) {
            throw std::runtime_error(
                std::string(providerName) + " is not available in this ONNX Runtime build.");
        }

        try {
            sessionOptions_ = Ort::SessionOptions{};
            sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            sessionOptions_.SetIntraOpNumThreads(1);
            sessionOptions_.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            configureProviderOptions(sessionOptions_, epConfig_.ep, epConfig_);

#if defined(_WIN32)
            const std::wstring widePath = toWidePath(modelPath);
            session_ = std::make_unique<Ort::Session>(env_, widePath.c_str(), sessionOptions_);
#else
            session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
#endif
        } catch (const Ort::Exception& error) {
            throw std::runtime_error(
                "Failed to create ONNX Runtime session for requested EP="
                + toCliName(epConfig_.ep) + ".\n - "
                + formatProviderLabel(epConfig_.ep) + ": " + error.what());
        }

        inputNames_.push_back("images");
        outputNames_.push_back("output0");

        std::cout << "[ONNX] Model loaded from: " << modelPath << std::endl;
        std::cout << "[ONNX] Execution Provider: "
                  << formatProviderLabel(epConfig_.ep) << std::endl;
        std::cout << "[ONNX] Precision: " << yolox_benchmark::toCliName(epConfig_.precision) << std::endl;
    }

    size_t inputBytes() const
    {
        return yolox_benchmark::kInputSize * yolox_benchmark::precisionElementSize(epConfig_.precision);
    }

    size_t outputBytes() const
    {
        return yolox_benchmark::kOutputSize * yolox_benchmark::precisionElementSize(epConfig_.precision);
    }

#if defined(VAI_YOLOX_HAVE_CUDA_RUNTIME)
    void ensureCudaBuffers()
    {
        throwIfOnnxCudaFailed(cudaSetDevice(epConfig_.cuda_device), "cudaSetDevice");
        const size_t inputBufferBytes = inputBytes();
        const size_t outputBufferBytes = outputBytes();
        if (!cudaInput_) {
            throwIfOnnxCudaFailed(cudaMalloc(&cudaInput_, inputBufferBytes), "cudaMalloc(input)");
            cudaInputBytes_ = inputBufferBytes;
        }
        if (!cudaOutput_) {
            throwIfOnnxCudaFailed(cudaMalloc(&cudaOutput_, outputBufferBytes), "cudaMalloc(output)");
            cudaOutputBytes_ = outputBufferBytes;
        }
    }

    BenchmarkResult runPureInferenceCuda(const std::vector<float>& inputData)
    {
        if (epConfig_.ep != OnnxEp::CUDA) {
            throw std::runtime_error("ONNX Runtime pure inference timing is currently implemented only for CUDA EP.");
        }
        if (inputData.size() != yolox_benchmark::kInputSize) {
            throw std::runtime_error("Expected 3x640x640 input tensor.");
        }

        ensureCudaBuffers();
        std::vector<uint16_t> halfInput;
        const void* inputSource = inputData.data();
        if (epConfig_.precision == yolox_benchmark::Precision::FP16) {
            halfInput = yolox_benchmark::floatsToHalfBits(inputData);
            inputSource = halfInput.data();
        }
        throwIfOnnxCudaFailed(
            cudaMemcpy(cudaInput_, inputSource, cudaInputBytes_, cudaMemcpyHostToDevice),
            "cudaMemcpy(input)");

        std::vector<int64_t> inputShape = {1, 3, 640, 640};
        std::vector<int64_t> outputShape = {1, static_cast<int64_t>(yolox_benchmark::kOutputAnchors), static_cast<int64_t>(yolox_benchmark::kOutputRows)};
        Ort::MemoryInfo cudaMemInfo("Cuda", OrtDeviceAllocator, epConfig_.cuda_device, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor(
            cudaMemInfo,
            cudaInput_,
            cudaInputBytes_,
            inputShape.data(),
            inputShape.size(),
            onnxTensorElementType(epConfig_.precision));
        Ort::Value outputTensor = Ort::Value::CreateTensor(
            cudaMemInfo,
            cudaOutput_,
            cudaOutputBytes_,
            outputShape.data(),
            outputShape.size(),
            onnxTensorElementType(epConfig_.precision));
        Ort::IoBinding binding(*session_);
        binding.BindInput(inputNames_.front(), inputTensor);
        binding.BindOutput(outputNames_.front(), outputTensor);

        const BenchmarkConfig& benchmarkConfig = benchmarkConfig_;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Measurement: ONNX Runtime Run with CUDA IoBinding including decode; no H2D/D2H or NMS" << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Warming up... " << std::flush;
        for (int index = 0; index < benchmarkConfig.warmup; ++index) {
            session_->Run(Ort::RunOptions{nullptr}, binding);
            binding.SynchronizeOutputs();
        }
        std::cout << "Done." << std::endl;

        std::cout << "[" << benchmarkConfig.runtime_name << "] Running "
                  << benchmarkConfig.iterations << " iterations..." << std::endl;

        std::vector<double> elapsedTimes;
        elapsedTimes.reserve(benchmarkConfig.iterations);
        Timer timer;
        for (int index = 0; index < benchmarkConfig.iterations; ++index) {
            timer.start();
            session_->Run(Ort::RunOptions{nullptr}, binding);
            binding.SynchronizeOutputs();
            elapsedTimes.emplace_back(timer.stop_ms());
        }
        return yolox_benchmark::makeBenchmarkResult(benchmarkConfig.runtime_name, elapsedTimes);
    }

    void downloadCudaOutput()
    {
        if (!cudaOutput_) {
            throw std::runtime_error("ONNX Runtime CUDA output buffer is not initialized.");
        }
        if (epConfig_.precision == yolox_benchmark::Precision::FP16) {
            std::vector<uint16_t> outputHalf(yolox_benchmark::kOutputSize);
            throwIfOnnxCudaFailed(
                cudaMemcpy(outputHalf.data(), cudaOutput_, cudaOutputBytes_, cudaMemcpyDeviceToHost),
                "cudaMemcpy(output)");
            lastOutput_ = yolox_benchmark::halfBitsToFloats(outputHalf.data(), outputHalf.size());
        } else {
            lastOutput_.resize(yolox_benchmark::kOutputSize);
            throwIfOnnxCudaFailed(
                cudaMemcpy(lastOutput_.data(), cudaOutput_, cudaOutputBytes_, cudaMemcpyDeviceToHost),
                "cudaMemcpy(output)");
        }
        yolox_benchmark::validateModelOutputSize(lastOutput_.size(), "ONNX Runtime");
    }

#endif

#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
    void ensureHipBuffers()
    {
        yolox_onnx_hip::setDevice(epConfig_.cuda_device);
        const size_t inputBufferBytes = inputBytes();
        const size_t outputBufferBytes = outputBytes();
        if (!hipInput_) {
            hipInput_ = yolox_onnx_hip::mallocDevice(inputBufferBytes, "hipMalloc(input)");
            hipInputBytes_ = inputBufferBytes;
        }
        if (!hipOutput_) {
            hipOutput_ = yolox_onnx_hip::mallocDevice(outputBufferBytes, "hipMalloc(output)");
            hipOutputBytes_ = outputBufferBytes;
        }
    }

    BenchmarkResult runPureInferenceMigraphx(const std::vector<float>& inputData)
    {
        if (epConfig_.ep != OnnxEp::ROCm) {
            throw std::runtime_error("ONNX Runtime MIGraphX pure inference timing requires MIGraphX EP.");
        }
        if (inputData.size() != yolox_benchmark::kInputSize) {
            throw std::runtime_error("Expected 3x640x640 input tensor.");
        }

        ensureHipBuffers();
        std::vector<uint16_t> halfInput;
        const void* inputSource = inputData.data();
        if (epConfig_.precision == yolox_benchmark::Precision::FP16) {
            halfInput = yolox_benchmark::floatsToHalfBits(inputData);
            inputSource = halfInput.data();
        }
        yolox_onnx_hip::copyHostToDevice(hipInput_, inputSource, hipInputBytes_, "hipMemcpy(input)");
        yolox_onnx_hip::deviceSynchronize("hipDeviceSynchronize(input)");

        std::vector<int64_t> inputShape = {1, 3, 640, 640};
        std::vector<int64_t> outputShape = {1, static_cast<int64_t>(yolox_benchmark::kOutputAnchors), static_cast<int64_t>(yolox_benchmark::kOutputRows)};
        Ort::MemoryInfo migraphxMemInfo(
            "Cuda",
            OrtMemoryInfoDeviceType_GPU,
            kAmdPciVendorId,
            static_cast<uint32_t>(epConfig_.cuda_device),
            OrtDeviceMemoryType_DEFAULT,
            0,
            OrtDeviceAllocator);
        Ort::Value inputTensor = Ort::Value::CreateTensor(
            migraphxMemInfo,
            hipInput_,
            hipInputBytes_,
            inputShape.data(),
            inputShape.size(),
            onnxTensorElementType(epConfig_.precision));
        Ort::Value outputTensor = Ort::Value::CreateTensor(
            migraphxMemInfo,
            hipOutput_,
            hipOutputBytes_,
            outputShape.data(),
            outputShape.size(),
            onnxTensorElementType(epConfig_.precision));
        Ort::IoBinding binding(*session_);
        binding.BindInput(inputNames_.front(), inputTensor);
        binding.BindOutput(outputNames_.front(), outputTensor);

        session_->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();
        yolox_onnx_hip::deviceSynchronize("hipDeviceSynchronize(binding probe)");

        const BenchmarkConfig& benchmarkConfig = benchmarkConfig_;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Measurement: ONNX Runtime MIGraphX Run with HIP IoBinding including decode; no H2D/D2H or NMS" << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] MemoryInfo: Cuda GPU device memory, vendor=0x1002 (AMD), device=" << epConfig_.cuda_device << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Timing: host Run + output synchronization. MIGraphX EP does not expose a user HIP stream in OrtMIGraphXProviderOptions." << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name << "] Warming up... " << std::flush;
        for (int index = 0; index < benchmarkConfig.warmup; ++index) {
            session_->Run(Ort::RunOptions{nullptr}, binding);
            binding.SynchronizeOutputs();
        }
        yolox_onnx_hip::deviceSynchronize("hipDeviceSynchronize(warmup)");
        std::cout << "Done." << std::endl;

        std::cout << "[" << benchmarkConfig.runtime_name << "] Running "
                  << benchmarkConfig.iterations << " iterations..." << std::endl;

        std::vector<double> elapsedTimes;
        elapsedTimes.reserve(benchmarkConfig.iterations);
        Timer timer;
        for (int index = 0; index < benchmarkConfig.iterations; ++index) {
            timer.start();
            session_->Run(Ort::RunOptions{nullptr}, binding);
            binding.SynchronizeOutputs();
            yolox_onnx_hip::deviceSynchronize("hipDeviceSynchronize(iteration)");
            elapsedTimes.emplace_back(timer.stop_ms());
        }
        return yolox_benchmark::makeBenchmarkResult(benchmarkConfig.runtime_name, elapsedTimes);

    }

    void downloadHipOutput()
    {
        if (!hipOutput_) {
            throw std::runtime_error("ONNX Runtime MIGraphX output buffer is not initialized.");
        }
        if (epConfig_.precision == yolox_benchmark::Precision::FP16) {
            std::vector<uint16_t> outputHalf(yolox_benchmark::kOutputSize);
            yolox_onnx_hip::copyDeviceToHost(outputHalf.data(), hipOutput_, hipOutputBytes_, "hipMemcpy(output)");
            lastOutput_ = yolox_benchmark::halfBitsToFloats(outputHalf.data(), outputHalf.size());
        } else {
            lastOutput_.resize(yolox_benchmark::kOutputSize);
            yolox_onnx_hip::copyDeviceToHost(lastOutput_.data(), hipOutput_, hipOutputBytes_, "hipMemcpy(output)");
        }
        yolox_benchmark::validateModelOutputSize(lastOutput_.size(), "ONNX Runtime");
    }
#endif

    const std::vector<float>& lastOutput() const
    {
        return lastOutput_;
    }
};

#ifndef ONNX_MIGRAPHX_GPU_HIP_FLAGS
    #define ONNX_MIGRAPHX_GPU_HIP_FLAGS ""
#endif

OnnxEp parseBackend(const std::string& value)
{
    if (value == "cuda") {
        return OnnxEp::CUDA;
    }
    if (value == "rocm") {
        return OnnxEp::ROCm;
    }
    throw std::invalid_argument("Unsupported --backend value: " + value);
}


void configureMIGraphXRuntimeEnvironment(const OnnxEpConfig& epConfig)
{
    if (epConfig.ep != OnnxEp::ROCm) {
        return;
    }

    constexpr const char* defaultHipFlags = ONNX_MIGRAPHX_GPU_HIP_FLAGS;
    if (defaultHipFlags[0] == '\0' || std::getenv("MIGRAPHX_GPU_HIP_FLAGS") != nullptr) {
        return;
    }

#if defined(_WIN32)
    _putenv_s("MIGRAPHX_GPU_HIP_FLAGS", defaultHipFlags);
#else
    setenv("MIGRAPHX_GPU_HIP_FLAGS", defaultHipFlags, 0);
#endif
}

void exitWithoutMIGraphXAtexitHandlersIfNeeded(OnnxEp ep, int status)
{
    if (ep != OnnxEp::ROCm) {
        return;
    }

    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    std::_Exit(status);
}

BenchmarkResult onnx_benchmark(
    const std::vector<float>& inputData,
    Variant variant,
    int warmup,
    int iterations,
    const OnnxEpConfig& epConfig,
    std::vector<Detection>& lastDetections)
{
    std::string variantName = variantToString(variant);
    std::cout << "\n=== ONNX Runtime YOLOX-" << variantName << " Benchmark ===" << std::endl;

    const std::string benchName = "ONNX_YOLOX_" + variantName + "_inference_"
        + formatProviderLabel(epConfig.ep) + "_"
        + yolox_benchmark::precisionDisplayName(epConfig.precision);

    BenchmarkConfig bm_cfg(warmup, iterations, benchName);
    YOLOXOnnxBenchmark bench(bm_cfg, epConfig);
    bench.prepare(getOnnxPath(variant, epConfig.precision));

    BenchmarkResult result;
    if (epConfig.ep == OnnxEp::CUDA) {
#if defined(VAI_YOLOX_HAVE_CUDA_RUNTIME)
        result = bench.runPureInferenceCuda(inputData);
        bench.downloadCudaOutput();
        lastDetections = yolox_benchmark::postprocessOutput(bench.lastOutput());
#else
        throw std::runtime_error("ONNX Runtime CUDA pure inference requires CUDA runtime support at build time.");
#endif
    } else if (epConfig.ep == OnnxEp::ROCm) {
#if defined(VAI_YOLOX_HAVE_HIP_RUNTIME)
        result = bench.runPureInferenceMigraphx(inputData);
        bench.downloadHipOutput();
        lastDetections = yolox_benchmark::postprocessOutput(bench.lastOutput());
#else
        throw std::runtime_error("ONNX Runtime MIGraphX pure inference requires HIP runtime support at build time.");
#endif
    } else {
        throw std::runtime_error("ONNX Runtime pure inference timing received an unknown backend.");
    }

    result.print();
    return result;
}

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n"
         << "Options:\n"
         << "  --size S        Model size: s, m, l, x (default: l)\n"
         << "  --backend TYPE  Backend: cuda, rocm (default: cuda)\n"
         << "  --gpu N         GPU device index for CUDA/ROCm (default: 0)\n"
         << "  --warmup N      Number of warmup iterations (default: "
         << yolox_benchmark::kDefaultWarmup << ")\n"
         << "  --iterations N  Number of benchmark iterations (default: "
         << yolox_benchmark::kDefaultIterations << ")\n"
         << "  --precision P   Precision: fp32, fp16 (default: fp32; fp16 requires cuda or rocm backend)\n"
         << "  --image PATH    Image input (default: "
         << yolox_benchmark::defaultImagePath() << ")\n"
         << "  --help          Show this help message\n";
}

} // namespace

int main(int argc, char** argv)
{
    yolox_benchmark::CommonCliOptions commonOptions;
    OnnxEpConfig epConfig;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (yolox_benchmark::parseCommonOption(arg, i, argc, argv, commonOptions)) {
                continue;
            }
            if (arg == "--backend" || arg == "--ep") {
                epConfig.ep = parseBackend(yolox_benchmark::takeOptionValue(i, argc, argv, arg));
                continue;
            }
            if (arg == "--gpu") {
                epConfig.cuda_device = yolox_benchmark::parseIntegerOption(
                    yolox_benchmark::takeOptionValue(i, argc, argv, arg), arg);
                continue;
            }
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    Variant variant = commonOptions.variant;
    const int warmup = commonOptions.warmup;
    const int iterations = commonOptions.iterations;
    epConfig.precision = commonOptions.precision;
    const std::string& imagePath = commonOptions.imagePath;

    try {
        yolox_benchmark::validateCommonCliOptions(commonOptions);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  YOLOX ONNX Runtime Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Variant: YOLOX-" << variantToString(variant) << std::endl;
    std::cout << "Backend: " << toCliName(epConfig.ep) << std::endl;
    std::cout << "Precision: " << yolox_benchmark::toCliName(epConfig.precision) << std::endl;
    std::cout << "GPU    : " << epConfig.cuda_device << std::endl;
    std::cout << "Warmup : " << warmup << ", Iterations: " << iterations << std::endl;

    configureMIGraphXRuntimeEnvironment(epConfig);

    std::vector<float> inputData;
    try {
        std::cout << "\n[Input] Loading image before benchmark: " << imagePath << std::endl;
        inputData = yolox_benchmark::preprocessImage(imagePath);
        std::cout << "[Input] Preprocessed image: 3x640x640 (CHW)" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    try {
        std::vector<Detection> detections;
        onnx_benchmark(
            inputData,
            variant,
            warmup,
            iterations,
            epConfig,
            detections);
        yolox_benchmark::printDetectionSummary(detections);
        exitWithoutMIGraphXAtexitHandlersIfNeeded(epConfig.ep, 0);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        exitWithoutMIGraphXAtexitHandlersIfNeeded(epConfig.ep, 1);
        return 1;
    }
}
