#include "benchmark.hpp"
#include "processing.hpp"

#include "command.h"
#include "gpu.h"
#include "layer.h"
#include "net.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if !NCNN_VULKAN
    #error "bench-yolox-ncnn requires an ncnn build with Vulkan enabled"
#endif

#ifndef YOLOX_WEIGHTS_DIR
    #define YOLOX_WEIGHTS_DIR ""
#endif

namespace {

std::string getNcnnPath(Variant variant)
{
    return yolox_benchmark::modelArtifactPath(variant, ".param");
}

std::string backendDisplayName(
    yolox_benchmark::Precision precision = yolox_benchmark::Precision::FP32)
{
    return "Vulkan " + yolox_benchmark::precisionDisplayName(precision);
}

std::string ncnnGpuDeviceLabel(int gpuIndex)
{
    std::string deviceName;
    if (gpuIndex >= 0 && gpuIndex < ncnn::get_gpu_count()) {
        const char* name = ncnn::get_gpu_info(gpuIndex).device_name();
        if (name) {
            deviceName = name;
        }
    }
    return std::to_string(gpuIndex) + " ("
        + (deviceName.empty() ? "Vulkan device" : deviceName) + ")";
}

class YOLOXNcnnBenchmark
{
private:
    BenchmarkConfig benchmarkConfig_;
    ncnn::Net net_;
    yolox_benchmark::Precision precision_ = yolox_benchmark::Precision::FP32;
    int gpuIndex_ = 0;
    int outputBlobIndex_ = -1;
    std::vector<float> lastOutput_;
#if NCNN_VULKAN
    ncnn::VkMat deviceInput_;
    ncnn::VkMat deviceOutput_;
    ncnn::VkAllocator* vulkanBlobAllocator_ = nullptr;
    ncnn::VkAllocator* vulkanStagingAllocator_ = nullptr;

    void ensureVulkanAllocators(const ncnn::VulkanDevice* vkdev)
    {
        if (!vkdev) {
            throw std::runtime_error("NCNN Vulkan device is not initialized.");
        }
        if (!vulkanBlobAllocator_) {
            vulkanBlobAllocator_ = vkdev->acquire_blob_allocator();
        }
        if (!vulkanStagingAllocator_) {
            vulkanStagingAllocator_ = vkdev->acquire_staging_allocator();
        }
        net_.opt.blob_vkallocator = vulkanBlobAllocator_;
        net_.opt.workspace_vkallocator = vulkanBlobAllocator_;
        net_.opt.staging_vkallocator = vulkanStagingAllocator_;
    }

    void releaseVulkanAllocators()
    {
        deviceInput_.release();
        deviceOutput_.release();

        const ncnn::VulkanDevice* vkdev = net_.vulkan_device();
        net_.opt.blob_vkallocator = nullptr;
        net_.opt.workspace_vkallocator = nullptr;
        net_.opt.staging_vkallocator = nullptr;

        if (!vkdev) {
            vulkanBlobAllocator_ = nullptr;
            vulkanStagingAllocator_ = nullptr;
            return;
        }
        if (vulkanBlobAllocator_) {
            vkdev->reclaim_blob_allocator(vulkanBlobAllocator_);
            vulkanBlobAllocator_ = nullptr;
        }
        if (vulkanStagingAllocator_) {
            vkdev->reclaim_staging_allocator(vulkanStagingAllocator_);
            vulkanStagingAllocator_ = nullptr;
        }
    }

#endif

public:
    YOLOXNcnnBenchmark(
        BenchmarkConfig config,
        yolox_benchmark::Precision precision,
        int gpuIndex = 0)
        : benchmarkConfig_(std::move(config))
        , precision_(precision)
        , gpuIndex_(gpuIndex)
    {
    }

    ~YOLOXNcnnBenchmark()
    {
#if NCNN_VULKAN
        releaseVulkanAllocators();
#endif
    }

    void prepare(const std::string& modelPath)
    {
#if NCNN_VULKAN
        releaseVulkanAllocators();
#endif
        net_.clear();

        ncnn::Option opt;
        const bool useFp16 = precision_ == yolox_benchmark::Precision::FP16;
        opt.use_fp16_packed = useFp16;
        opt.use_fp16_storage = useFp16;
        opt.use_fp16_arithmetic = useFp16;
        opt.use_bf16_storage = false;
        opt.use_bf16_packed = false;
        opt.use_int8_inference = false;
        opt.use_int8_storage = false;
        opt.use_int8_arithmetic = false;
        opt.use_int8_packed = false;
        opt.use_cooperative_matrix = true;
        opt.use_vulkan_compute = true;
        net_.set_vulkan_device(gpuIndex_);

        net_.opt = opt;

        std::string paramPath = modelPath;
        std::string binPath = modelPath;
        size_t pos = binPath.find(".param");
        if (pos != std::string::npos) {
            binPath.replace(pos, 6, ".bin");
        }

        const int retParam = net_.load_param(paramPath.c_str());
        const int retModel = net_.load_model(binPath.c_str());
        if (retParam != 0 || retModel != 0) {
            throw std::runtime_error("ncnn model load failed: " + paramPath);
        }

        const std::vector<int>& outputIndexes = net_.output_indexes();
        if (outputIndexes.size() != 1) {
            throw std::runtime_error(
                "ncnn model must expose exactly one [1,8400,85] output, got "
                + std::to_string(outputIndexes.size()) + ".");
        }
        outputBlobIndex_ = outputIndexes.front();

        std::cout << "[NCNN] Model loaded from: " << modelPath << std::endl;
        std::cout << "[NCNN] Backend: " << backendDisplayName(precision_)
                  << ", GPU: " << ncnnGpuDeviceLabel(gpuIndex_) << std::endl;
        if (useFp16) {
            std::cout << "[NCNN] FP16 packed/storage/arithmetic: on/on/on" << std::endl;
        }
        std::cout << "[NCNN] Cooperative matrix: on" << std::endl;
    }

    std::vector<float> copyMatToFloatVector(const ncnn::Mat& output) const
    {
        const size_t elementCount = output.total() * static_cast<size_t>(output.elempack);
        const size_t elementBytes = output.elemsize / static_cast<size_t>(output.elempack);
        if (elementBytes == sizeof(float)) {
            const float* data = static_cast<const float*>(output.data);
            return std::vector<float>(data, data + elementCount);
        }
        if (elementBytes == sizeof(uint16_t)) {
            const uint16_t* data = static_cast<const uint16_t*>(output.data);
            return yolox_benchmark::halfBitsToFloats(data, elementCount);
        }
        throw std::runtime_error("NCNN output tensor has unsupported element size: " + std::to_string(elementBytes));
    }

#if NCNN_VULKAN
    void uploadInputToVulkan(ncnn::Mat& input)
    {
        const ncnn::VulkanDevice* vkdev = net_.vulkan_device();
        if (!vkdev) {
            throw std::runtime_error("NCNN Vulkan device is not initialized.");
        }
        ensureVulkanAllocators(vkdev);
        ncnn::VkTransfer transfer(vkdev);
        transfer.record_upload(input, deviceInput_, net_.opt, false);
        const int ret = transfer.submit_and_wait();
        if (ret != 0) {
            throw std::runtime_error("NCNN Vulkan input upload failed with code " + std::to_string(ret) + ".");
        }
        if (deviceInput_.empty()) {
            throw std::runtime_error("NCNN Vulkan input upload produced an empty VkMat.");
        }
    }

    void executeVulkanModel()
    {
        const ncnn::VulkanDevice* vkdev = net_.vulkan_device();
        if (!vkdev) {
            throw std::runtime_error("NCNN Vulkan device is not initialized.");
        }

        ncnn::Extractor extractor = net_.create_extractor();
        const int inputResult = extractor.input("in0", deviceInput_);
        if (inputResult != 0) {
            throw std::runtime_error(
                "NCNN Vulkan input binding failed with code "
                + std::to_string(inputResult) + ".");
        }

        ncnn::VkCompute command(vkdev);
        const int outputResult = extractor.extract(
            outputBlobIndex_,
            deviceOutput_,
            command);
        if (outputResult != 0) {
            throw std::runtime_error(
                "NCNN Vulkan output extraction failed with code "
                + std::to_string(outputResult) + ".");
        }

        const int submitResult = command.submit_and_wait();
        if (submitResult != 0) {
            throw std::runtime_error(
                "NCNN Vulkan command submit failed with code "
                + std::to_string(submitResult) + ".");
        }
        if (deviceOutput_.empty()) {
            throw std::runtime_error("NCNN Vulkan returned an empty output tensor.");
        }
    }

    void materializeVulkanOutput()
    {
        const ncnn::VulkanDevice* vkdev = net_.vulkan_device();
        if (!vkdev) {
            throw std::runtime_error("NCNN Vulkan device is not initialized.");
        }
        if (deviceOutput_.empty()) {
            throw std::runtime_error("NCNN Vulkan output buffer is not initialized.");
        }

        ensureVulkanAllocators(vkdev);
        ncnn::VkCompute command(vkdev);
        ncnn::Mat output;
        command.record_download(deviceOutput_, output, net_.opt);
        const int submitResult = command.submit_and_wait();
        if (submitResult != 0 || output.empty()) {
            throw std::runtime_error(
                "NCNN Vulkan output download failed with code "
                + std::to_string(submitResult) + ".");
        }

        if (output.elempack != 1) {
            ncnn::Mat unpacked;
            ncnn::convert_packing(output, unpacked, 1, net_.opt);
            if (unpacked.empty()) {
                throw std::runtime_error("NCNN output unpack failed.");
            }
            output = unpacked;
        }

        lastOutput_ = copyMatToFloatVector(output);
        yolox_benchmark::validateModelOutputSize(lastOutput_.size(), "NCNN");
    }

    BenchmarkResult runInference(ncnn::Mat& input)
    {
        if (!net_.vulkan_device()) {
            throw std::runtime_error("NCNN Vulkan device is not initialized.");
        }

        uploadInputToVulkan(input);
        const BenchmarkConfig& benchmarkConfig = benchmarkConfig_;

        std::cout << "[" << benchmarkConfig.runtime_name
                  << "] Measurement: ncnn Vulkan model inference; input upload, output download, and NMS excluded"
                  << std::endl;
        std::cout << "[" << benchmarkConfig.runtime_name
                  << "] Timing: host wall time around synchronized extraction"
                  << std::endl;

        std::cout << "[" << benchmarkConfig.runtime_name << "] Warming up... " << std::flush;
        for (int index = 0; index < benchmarkConfig.warmup; ++index) {
            executeVulkanModel();
        }
        std::cout << "Done." << std::endl;

        std::cout << "[" << benchmarkConfig.runtime_name << "] Running "
                  << benchmarkConfig.iterations << " iterations..." << std::endl;

        std::vector<double> elapsedTimes;
        elapsedTimes.reserve(benchmarkConfig.iterations);
        for (int index = 0; index < benchmarkConfig.iterations; ++index) {
            Timer timer;
            timer.start();

            executeVulkanModel();

            elapsedTimes.emplace_back(timer.stop_ms());
        }

        return yolox_benchmark::makeBenchmarkResult(
            benchmarkConfig.runtime_name,
            elapsedTimes);
    }

    void downloadVulkanOutput()
    {
        materializeVulkanOutput();
    }

#endif

    const std::vector<float>& lastOutput() const
    {
        return lastOutput_;
    }
};

#if !defined(_WIN32)
std::string forceRadvWave32()
{
    const char* currentValue = std::getenv("RADV_PERFTEST");
    std::string settings = currentValue ? currentValue : "";
    const std::string delimitedSettings = "," + settings + ",";
    if (delimitedSettings.find(",cswave32,") == std::string::npos) {
        if (!settings.empty()) {
            settings += ',';
        }
        settings += "cswave32";
        if (setenv("RADV_PERFTEST", settings.c_str(), 1) != 0) {
            throw std::runtime_error("Failed to force RADV compute shaders to wave32.");
        }
    }
    return settings;
}
#endif

std::string benchmarkName(
    Variant variant,
    yolox_benchmark::Precision precision)
{
    return "NCNN_Vulkan_" + yolox_benchmark::precisionDisplayName(precision)
        + "_YOLOX_" + variantToString(variant) + "_inference";
}

BenchmarkResult runInference(
    std::vector<float>& inputData,
    Variant variant,
    yolox_benchmark::Precision precision,
    int gpuIndex,
    int warmup,
    int iterations,
    std::vector<Detection>& detections)
{
    BenchmarkConfig config(
        warmup,
        iterations,
        benchmarkName(
            variant,
            precision));
    YOLOXNcnnBenchmark benchmark(config, precision, gpuIndex);
    benchmark.prepare(getNcnnPath(variant));

    ncnn::Mat input(
        yolox_benchmark::kInputWidth,
        yolox_benchmark::kInputHeight,
        yolox_benchmark::kInputChannels,
        inputData.data());
    BenchmarkResult result = benchmark.runInference(input);
    benchmark.downloadVulkanOutput();
    detections = yolox_benchmark::postprocessOutput(benchmark.lastOutput());
    return result;
}

void printUsage(const char* programName)
{
    std::cout
        << "Usage: " << programName << " [options]\n"
        << "  --size S          Model size: s, m, l, x\n"
        << "  --gpu N           Vulkan device index\n"
        << "  --precision P     fp32 or fp16\n"
        << "  --warmup N        Warmup iterations\n"
        << "  --iterations N    Measured iterations\n"
        << "  --image PATH      Input image (default: "
        << yolox_benchmark::defaultImagePath() << ")\n"
        << "  --help            Show this message\n";
}

} // namespace

int main(int argc, char** argv)
try
{
#if !defined(_WIN32)
    const std::string radvPerfTest = forceRadvWave32();
#endif
    yolox_benchmark::CommonCliOptions options;
    int gpuIndex = 0;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (argument == "--gpu") {
            gpuIndex = yolox_benchmark::parseIntegerOption(
                yolox_benchmark::takeOptionValue(index, argc, argv, argument),
                argument);
            continue;
        }
        if (!yolox_benchmark::parseCommonOption(argument, index, argc, argv, options)) {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    yolox_benchmark::validateCommonCliOptions(options);
    const int gpuCount = ncnn::get_gpu_count();
    if (gpuCount <= 0) {
        throw std::runtime_error("No Vulkan device is available for ncnn.");
    }
    if (gpuIndex < 0 || gpuIndex >= gpuCount) {
        throw std::invalid_argument(
            "--gpu must be in range [0, " + std::to_string(gpuCount - 1) + "].");
    }

    std::cout << "ncnn Vulkan device: " << ncnnGpuDeviceLabel(gpuIndex) << std::endl;
#if !defined(_WIN32)
    std::cout << "RADV_PERFTEST: " << radvPerfTest << std::endl;
#endif
    std::cout << "Compute shader subgroup: wave32" << std::endl;

    std::vector<float> inputData = yolox_benchmark::preprocessImage(options.imagePath);
    std::vector<Detection> detections;
    BenchmarkResult result = runInference(
        inputData,
        options.variant,
        options.precision,
        gpuIndex,
        options.warmup,
        options.iterations,
        detections);

    result.print();
    if (!detections.empty()) {
        yolox_benchmark::printDetectionSummary(detections);
    }
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
}
