#include "benchmark.hpp"
#include "processing.hpp"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef YOLOX_WEIGHTS_DIR
    #define YOLOX_WEIGHTS_DIR ""
#endif

namespace {

using executorch::extension::Module;
using executorch::extension::from_blob;

constexpr const char* kRuntimeName = "Executorch_Vulkan";
constexpr const char* kDisplayName = "YOLOX ExecuTorch Vulkan";
#if defined(_WIN32)
constexpr const char* kVulkanSharedLibrary = "vulkan-1.dll";
#else
constexpr const char* kVulkanSharedLibrary = "libvulkan.so.1";
#endif

struct InputTensorHolder
{
    std::vector<uint16_t> half_data;
    executorch::extension::TensorPtr tensor;
};

std::string getArtifactPath(
    Variant variant,
    yolox_benchmark::Precision precision)
{
    return yolox_benchmark::modelArtifactPath(
        variant, "." + yolox_benchmark::toCliName(precision) + ".pte");
}

std::vector<float> loadInputData(const std::string& imagePath)
{
    std::cout << "\n[Input] Loading image: " << imagePath << std::endl;
    std::vector<float> prepData = yolox_benchmark::preprocessImage(imagePath);
    std::cout << "[Input] Preprocessed image: 3x640x640 (CHW)" << std::endl;
    return prepData;
}

void printBanner(
    Variant variant,
    int warmup,
    int iterations,
    const std::string& modelPath)
{
    std::cout << "========================================" << std::endl;
    std::cout << "  " << kDisplayName << " Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Variant : YOLOX-" << variantToString(variant) << std::endl;
    std::cout << "Model   : " << modelPath << std::endl;
    std::cout << "Warmup  : " << warmup << ", Iterations: " << iterations << std::endl;
}

void verifyVulkanSharedLibrary()
{
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(kVulkanSharedLibrary);
    if (!handle) {
        throw std::runtime_error(
            std::string("Required shared library '") + kVulkanSharedLibrary
            + "' is not available.");
    }
    FreeLibrary(handle);
#else
    void* handle = dlopen(kVulkanSharedLibrary, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error(
            std::string("Required shared library '") + kVulkanSharedLibrary
            + "' is not available: " + dlerror());
    }
    dlclose(handle);
#endif
}

std::vector<float> copyModelOutput(const std::vector<executorch::runtime::EValue>& outputs)
{
    const executorch::runtime::EValue* tensorOutput = nullptr;
    const executorch::runtime::EValue* fallbackTensorOutput = nullptr;
    for (const auto& output : outputs) {
        if (!output.isTensor()) {
            continue;
        }
        if (!fallbackTensorOutput) {
            fallbackTensorOutput = &output;
        }

        const auto& tensor = output.toTensor();
        const size_t elementCount = tensor.numel() > 0
            ? static_cast<size_t>(tensor.numel())
            : 0;
        if (elementCount != yolox_benchmark::kOutputSize) {
            continue;
        }
        if (tensorOutput) {
            throw std::runtime_error("ExecuTorch benchmark found multiple YOLOX-sized output tensors.");
        }
        tensorOutput = &output;
    }

    if (!tensorOutput) {
        tensorOutput = fallbackTensorOutput;
    }
    if (!tensorOutput) {
        throw std::runtime_error("ExecuTorch forward() returned no tensor outputs.");
    }

    const auto& tensor = tensorOutput->toTensor();
    if (tensor.numel() <= 0) {
        throw std::runtime_error("ExecuTorch returned an empty output tensor.");
    }

    const size_t elementCount = static_cast<size_t>(tensor.numel());
    std::vector<float> output;
    if (tensor.scalar_type() == executorch::aten::ScalarType::Float) {
        const float* data = tensor.const_data_ptr<float>();
        output.assign(data, data + elementCount);
    } else if (tensor.scalar_type() == executorch::aten::ScalarType::Half) {
        const uint16_t* data = tensor.const_data_ptr<uint16_t>();
        output = yolox_benchmark::halfBitsToFloats(data, elementCount);
    } else {
        throw std::runtime_error("ExecuTorch YOLOX output tensor must be float32 or float16.");
    }
    yolox_benchmark::validateModelOutputSize(output.size(), "ExecuTorch");
    return output;
}

class YOLOXExecuTorchBenchmark
{
private:
    BenchmarkConfig benchmarkConfig_;
    std::unique_ptr<Module> module_;
    yolox_benchmark::Precision precision_ = yolox_benchmark::Precision::FP32;
    bool loggedOutputInfo_ = false;

    InputTensorHolder makeInputTensor(std::vector<float>& inputData) const
    {
        const std::vector<executorch::aten::SizesType> shape = { 1, 3, 640, 640 };
        InputTensorHolder input;
        if (precision_ == yolox_benchmark::Precision::FP16) {
            input.half_data = yolox_benchmark::floatsToHalfBits(inputData);
            input.tensor = from_blob(
                input.half_data.data(),
                shape,
                executorch::aten::ScalarType::Half,
                executorch::aten::TensorShapeDynamism::STATIC);
            return input;
        }

        input.tensor = from_blob(
            inputData.data(),
            shape,
            executorch::aten::ScalarType::Float,
            executorch::aten::TensorShapeDynamism::STATIC);
        return input;
    }

public:
    YOLOXExecuTorchBenchmark(
        BenchmarkConfig config,
        yolox_benchmark::Precision precision)
        : benchmarkConfig_(std::move(config))
        , precision_(precision)
    {
    }

    void prepare(const std::string& modelPath)
    {
        executorch::runtime::runtime_init();
        verifyVulkanSharedLibrary();

        module_ = std::make_unique<Module>(
            modelPath,
            Module::LoadMode::Mmap);

        if (const auto error = module_->load();
            error != executorch::runtime::Error::Ok) {
            throw std::runtime_error(
                "ExecuTorch failed to load program from '" + modelPath + "'.");
        }

        if (const auto error = module_->load_forward();
            error != executorch::runtime::Error::Ok) {
            throw std::runtime_error(
                "ExecuTorch failed to load the 'forward' method from '" + modelPath + "'.");
        }

        const auto meta = module_->method_meta("forward");
        if (meta.ok()) {
            const auto inputMeta = meta->input_tensor_meta(0);
            if (inputMeta.ok()) {
                const auto sizes = inputMeta->sizes();
                std::cout << "[ExecuTorch] Input shape: [";
                for (size_t index = 0; index < sizes.size(); ++index) {
                    std::cout << sizes[index];
                    if (index + 1 != sizes.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
            }
        }

        std::cout << "[ExecuTorch] Model loaded from: " << modelPath << std::endl;
        std::cout << "[ExecuTorch] Backend: " << kDisplayName << std::endl;
    }

    double forwardOnly(InputTensorHolder& input)
    {
        const auto start = std::chrono::steady_clock::now();
        auto outputs = module_->forward(input.tensor);
        const auto end = std::chrono::steady_clock::now();

        if (!outputs.ok()) {
            throw std::runtime_error(
                std::string("ExecuTorch forward() call failed: ") +
                executorch::runtime::to_string(outputs.error()));
        }
        if (outputs->empty()) {
            throw std::runtime_error("ExecuTorch forward() returned no outputs.");
        }
        if (!loggedOutputInfo_) {
            std::cout << "[ExecuTorch] Output count: " << outputs->size() << std::endl;
            loggedOutputInfo_ = true;
        }
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::vector<float> forwardAndCopyOutput(InputTensorHolder& input)
    {
        auto outputs = module_->forward(input.tensor);
        if (!outputs.ok()) {
            throw std::runtime_error(
                std::string("ExecuTorch forward() call failed: ") +
                executorch::runtime::to_string(outputs.error()));
        }
        if (outputs->empty()) {
            throw std::runtime_error("ExecuTorch forward() returned no outputs.");
        }

        return copyModelOutput(*outputs);
    }

    BenchmarkResult run(
        std::vector<float>& inputData,
        std::vector<Detection>& lastDetections)
    {
        std::cout << "[" << benchmarkConfig_.runtime_name
                  << "] Measurement: prepared ExecuTorch input tensor reused; "
                  << "module->forward() including decode; no host input conversion, output materialization, or NMS"
                  << std::endl;
        auto input = makeInputTensor(inputData);

        std::cout << "[" << benchmarkConfig_.runtime_name << "] Warming up... "
                  << std::flush;
        for (int index = 0; index < benchmarkConfig_.warmup; ++index) {
            forwardOnly(input);
        }
        std::cout << "Done." << std::endl;

        std::cout << "[" << benchmarkConfig_.runtime_name << "] Running "
                  << benchmarkConfig_.iterations << " iterations..." << std::endl;
        std::vector<double> elapsedTimes;
        elapsedTimes.reserve(benchmarkConfig_.iterations);
        for (int index = 0; index < benchmarkConfig_.iterations; ++index) {
            elapsedTimes.push_back(forwardOnly(input));
        }

        BenchmarkResult result = yolox_benchmark::makeBenchmarkResult(
            benchmarkConfig_.runtime_name, elapsedTimes);
        lastDetections = yolox_benchmark::postprocessOutput(
            forwardAndCopyOutput(input));
        return result;
    }
};

std::string makeBenchmarkName(
    Variant variant,
    yolox_benchmark::Precision precision)
{
    return std::string(kRuntimeName)
        + "_" + yolox_benchmark::precisionDisplayName(precision)
        + "_YOLOX_" + variantToString(variant)
        + "_inference";
}

BenchmarkResult runBenchmark(
    std::vector<float>& inputData,
    Variant variant,
    int warmup,
    int iterations,
    const std::string& modelPath,
    yolox_benchmark::Precision precision,
    std::vector<Detection>& lastDetections)
{
    const std::string variantName = variantToString(variant);

    std::cout << "\n=== " << kDisplayName
              << " YOLOX-" << variantName << " Benchmark ===" << std::endl;

    const std::string benchName = makeBenchmarkName(variant, precision);
    BenchmarkConfig benchmarkConfig(warmup, iterations, benchName);
    YOLOXExecuTorchBenchmark benchmark(benchmarkConfig, precision);
    benchmark.prepare(modelPath);
    BenchmarkResult result = benchmark.run(inputData, lastDetections);
    result.print();
    return result;
}

struct VulkanDeviceInfo
{
    uint32_t index = 0;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    std::string type;
    std::string name;
};

struct VulkanSelection
{
    VulkanDeviceInfo device;
    uint32_t runtimeIndex = 0;
    std::optional<std::string> icdPath;
};

std::string trim(std::string value)
{
    const auto isSpace = [](unsigned char character) {
        return std::isspace(character) != 0;
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](unsigned char character) {
        return !isSpace(character);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char character) {
        return !isSpace(character);
    }).base(), value.end());
    return value;
}

std::string hexId(uint32_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::nouppercase << std::setfill('0') << std::setw(4) << value;
    return stream.str();
}

std::string vulkanSelector(const VulkanDeviceInfo& device)
{
    return hexId(device.vendorId) + ":" + hexId(device.deviceId);
}

std::string vulkanDeviceLabel(const VulkanDeviceInfo& device)
{
    std::ostringstream stream;
    stream << device.index << " (" << device.name << ", "
           << device.type
           << ", vendor=0x" << hexId(device.vendorId)
           << ", device=0x" << hexId(device.deviceId) << ")";
    return stream.str();
}

std::optional<uint32_t> parseHexProperty(const std::string& line)
{
    const size_t marker = line.find("0x");
    if (marker == std::string::npos) {
        return std::nullopt;
    }

    try {
        return static_cast<uint32_t>(std::stoul(line.substr(marker + 2), nullptr, 16));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::string valueAfterEquals(const std::string& line)
{
    const size_t marker = line.find('=');
    if (marker == std::string::npos) {
        return {};
    }
    return trim(line.substr(marker + 1));
}

std::vector<VulkanDeviceInfo> enumerateVulkanDevices()
{
    std::vector<VulkanDeviceInfo> devices;
#if defined(_WIN32)
    FILE* pipe = _popen("vulkaninfo --summary 2>NUL", "r");
#else
    FILE* pipe = popen("vulkaninfo --summary 2>/dev/null", "r");
#endif
    if (!pipe) {
        throw std::runtime_error("failed to run vulkaninfo for Vulkan device enumeration.");
    }

    VulkanDeviceInfo current;
    bool inDevice = false;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        const std::string line = trim(buffer);
        if (line.rfind("GPU", 0) == 0 && !line.empty() && line.back() == ':') {
            if (inDevice) {
                devices.push_back(current);
            }
            current = VulkanDeviceInfo{};
            current.index = static_cast<uint32_t>(devices.size());
            inDevice = true;
            continue;
        }

        if (!inDevice) {
            continue;
        }
        if (line.rfind("vendorID", 0) == 0) {
            if (auto value = parseHexProperty(line)) {
                current.vendorId = *value;
            }
        } else if (line.rfind("deviceID", 0) == 0) {
            if (auto value = parseHexProperty(line)) {
                current.deviceId = *value;
            }
        } else if (line.rfind("deviceType", 0) == 0) {
            current.type = valueAfterEquals(line);
        } else if (line.rfind("deviceName", 0) == 0) {
            current.name = valueAfterEquals(line);
        }
    }
    if (inDevice) {
        devices.push_back(current);
    }

#if defined(_WIN32)
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    if (status != 0 || devices.empty()) {
        throw std::runtime_error("vulkaninfo did not return any Vulkan physical devices.");
    }
    return devices;
}

void setEnvironmentVariable(const std::string& name, const std::string& value)
{
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (setenv(name.c_str(), value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + name);
    }
#endif
}

std::optional<std::string> firstExistingFile(const std::vector<const char*>& paths)
{
    for (const char* path : paths) {
        FILE* file = fopen(path, "r");
        if (file) {
            fclose(file);
            return std::string(path);
        }
    }
    return std::nullopt;
}

std::optional<std::string> icdPathForVendor(uint32_t vendorId)
{
    if (vendorId == 0x10de) {
        return firstExistingFile({
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/etc/vulkan/icd.d/nvidia_icd.json",
        });
    }
    if (vendorId == 0x1002) {
        return firstExistingFile({
            "/usr/share/vulkan/icd.d/radeon_icd.json",
            "/etc/vulkan/icd.d/radeon_icd.json",
            "/usr/share/vulkan/icd.d/amd_icd64.json",
            "/etc/vulkan/icd.d/amd_icd64.json",
        });
    }
    if (vendorId == 0x8086) {
        return firstExistingFile({
            "/usr/share/vulkan/icd.d/intel_icd.json",
            "/etc/vulkan/icd.d/intel_icd.json",
            "/usr/share/vulkan/icd.d/intel_hasvk_icd.json",
            "/etc/vulkan/icd.d/intel_hasvk_icd.json",
        });
    }
    if (vendorId == 0x10005) {
        return firstExistingFile({
            "/usr/share/vulkan/icd.d/lvp_icd.json",
            "/etc/vulkan/icd.d/lvp_icd.json",
        });
    }
    return std::nullopt;
}

std::optional<std::string> configureVulkanIcd(const VulkanDeviceInfo& selected)
{
    if (const char* existing = std::getenv("VK_ICD_FILENAMES")) {
        if (existing[0] != '\0') {
            return std::string(existing);
        }
    }

    auto icdPath = icdPathForVendor(selected.vendorId);
    if (icdPath) {
        setEnvironmentVariable("VK_ICD_FILENAMES", *icdPath);
    }
    return icdPath;
}

VulkanSelection configureVulkanGpuSelection(int gpuIndex)
{
    const auto devices = enumerateVulkanDevices();
    if (gpuIndex < 0 || static_cast<size_t>(gpuIndex) >= devices.size()) {
        std::ostringstream message;
        message << "--gpu must be in range [0, " << (devices.size() - 1)
                << "] for Vulkan backend. Available Vulkan devices:";
        for (const auto& device : devices) {
            message << "\n  GPU" << vulkanDeviceLabel(device);
        }
        throw std::invalid_argument(message.str());
    }

    const VulkanDeviceInfo selected = devices[static_cast<size_t>(gpuIndex)];
    const auto icdPath = configureVulkanIcd(selected);
    setEnvironmentVariable("MESA_VK_DEVICE_SELECT", vulkanSelector(selected));

    // Selecting an ICD and setting MESA_VK_DEVICE_SELECT can filter or reorder
    // the physical-device list. Resolve the selected device again in the exact
    // environment that ExecuTorch will use instead of forwarding its original
    // global vulkaninfo index.
    const auto runtimeDevices = enumerateVulkanDevices();
    const auto runtimeDevice = std::find_if(
        runtimeDevices.begin(),
        runtimeDevices.end(),
        [&](const VulkanDeviceInfo& device) {
            return device.vendorId == selected.vendorId
                && device.deviceId == selected.deviceId
                && device.name == selected.name;
        });
    if (runtimeDevice == runtimeDevices.end()) {
        throw std::runtime_error(
            "selected Vulkan device disappeared after configuring the Vulkan ICD.");
    }

    const uint32_t runtimeIndex = static_cast<uint32_t>(
        std::distance(runtimeDevices.begin(), runtimeDevice));
    setEnvironmentVariable("ETVK_DEVICE_INDEX", std::to_string(runtimeIndex));
    return VulkanSelection{selected, runtimeIndex, icdPath};
}

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --size S        Model size: s, m, l, x (default: l)\n"
              << "  --gpu N         Vulkan GPU device index (default: 0)\n"
              << "  --warmup N      Number of warmup iterations (default: "
              << BenchmarkConfig::kDefaultWarmup << ")\n"
              << "  --iterations N  Number of benchmark iterations (default: "
              << BenchmarkConfig::kDefaultIterations << ")\n"
              << "  --precision P   Precision: fp32, fp16 (default: fp32)\n"
              << "  --image PATH    Image input (default: "
              << yolox_benchmark::defaultImagePath() << ")\n"
              << "  --help          Show this help message\n";
}

} // namespace

int main(int argc, char** argv)
{
    yolox_benchmark::CommonCliOptions options;
    int gpuIndex = 0;

    try {
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
            if (!yolox_benchmark::parseCommonOption(
                    argument, index, argc, argv, options)) {
                throw std::invalid_argument("Unknown argument: " + argument);
            }
        }

        yolox_benchmark::validateCommonCliOptions(options);
        if (gpuIndex < 0) {
            throw std::invalid_argument("--gpu must be greater than or equal to zero.");
        }
        const std::string modelPath = getArtifactPath(options.variant, options.precision);
        const VulkanSelection selectedVulkanDevice = configureVulkanGpuSelection(gpuIndex);

        printBanner(options.variant, options.warmup, options.iterations, modelPath);
        std::cout << "Backend : Vulkan" << std::endl;
        std::cout << "Precision: " << yolox_benchmark::toCliName(options.precision) << std::endl;
        std::cout << "GPU     : " << vulkanDeviceLabel(selectedVulkanDevice.device)
                  << " via ETVK_DEVICE_INDEX=" << selectedVulkanDevice.runtimeIndex
                  << ", MESA_VK_DEVICE_SELECT=" << vulkanSelector(selectedVulkanDevice.device);
        if (selectedVulkanDevice.icdPath) {
            std::cout << ", VK_ICD_FILENAMES=" << *selectedVulkanDevice.icdPath;
        }
        std::cout << std::endl;
        auto inputData = loadInputData(options.imagePath);
        std::vector<Detection> detections;
        runBenchmark(
            inputData,
            options.variant,
            options.warmup,
            options.iterations,
            modelPath,
            options.precision,
            detections);
        yolox_benchmark::printDetectionSummary(detections);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ExecuTorch] " << error.what() << std::endl;
        return 1;
    }
}
