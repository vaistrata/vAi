#include "benchmark.hpp"
#include "processing.hpp"

#include "net.h"
#include "megvii-weights.h"
#include "vai/engine/weights.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolox = model::experimental::yolox;

#ifndef YOLOX_WEIGHTS_DIR
#define YOLOX_WEIGHTS_DIR ""
#endif

namespace {

yolox::Tier toTier(Variant variant)
{
    switch (variant) {
        case Variant::SMALL: return yolox::Tier::SMALL;
        case Variant::MEDIUM: return yolox::Tier::MEDIUM;
        case Variant::LARGE: return yolox::Tier::LARGE;
        case Variant::XLARGE: return yolox::Tier::XLARGE;
    }
    throw std::invalid_argument("Unsupported YOLOX variant.");
}

vai::Dtype toDType(yolox_benchmark::Precision precision)
{
    return precision == yolox_benchmark::Precision::FP16
        ? vai::Dtype::Fp16
        : vai::Dtype::Fp32;
}

std::string weightsPath(Variant variant)
{
    return yolox_benchmark::modelArtifactPath(variant, ".safetensors");
}

std::vector<float> outputToFloats(vai::Tensor& output)
{
    yolox_benchmark::validateModelOutputSize(output.numElements(), "vAi");
    if (output.dtype() == vai::Dtype::Fp32) {
        const float* data = output.getData<float>();
        return {data, data + output.numElements()};
    }
    if (output.dtype() == vai::Dtype::Fp16) {
        const uint16_t* data = output.getData<uint16_t>();
        return yolox_benchmark::halfBitsToFloats(data, output.numElements());
    }
    throw std::runtime_error("vAi returned an unsupported output dtype.");
}

BenchmarkResult runInference(yolox::Net& net, vai::Tensor& input, const BenchmarkConfig& config,
                             std::vector<float>& lastOutput)
{
    std::cout << "[" << config.runtime_name
              << "] Measurement: Vulkan GPU timestamp around the complete forward command buffer; "
                 "input upload, output download, CPU postprocess, and NMS are excluded"
              << std::endl;

    eva::Device device = net.device();
    eva::QueryPool timingPool = device.createTimestampQueryPool(2);
    eva::CommandBuffer timingStart = device.newCommandBuffer(eva::queue_compute)
        .begin()
        .resetQueryPool(timingPool)
        .writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, timingPool, 0)
        .end();
    eva::CommandBuffer timingStop = device.newCommandBuffer(eva::queue_compute)
        .begin()
        .writeTimestamp(eva::PIPELINE_STAGE::ALL_COMMANDS, timingPool, 1)
        .end();

    vai::Tensor output;
    for (int index = 0; index < config.warmup; ++index) {
        net({{"in0", input}}, {{"out0", &output}});
        net.sync();
    }

    std::vector<double> elapsedTimes;
    elapsedTimes.reserve(config.iterations);
    for (int index = 0; index < config.iterations; ++index) {
        timingStart.submit();
        net({{"in0", input}}, {{"out0", &output}});
        timingStop.submit().wait();
        net.sync();
        elapsedTimes.push_back(timingPool.getElapsedMs(0, 1));
    }

    lastOutput = outputToFloats(output);
    return yolox_benchmark::makeBenchmarkResult(config.runtime_name, elapsedTimes);
}

void printUsage(const char* programName)
{
    std::cout
        << "Usage: " << programName << " [options]\n"
        << "  --size S          Model size: s, m, l, x\n"
        << "  --precision P     Weight/compute precision: fp32, fp16\n"
        << "  --IO P            Input/output storage precision: fp32, fp16\n"
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
    yolox_benchmark::CommonCliOptions options;
    yolox_benchmark::Precision ioPrecision = yolox_benchmark::Precision::FP32;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (argument == "--IO") {
            const auto parsed = yolox_benchmark::parsePrecision(
                yolox_benchmark::takeOptionValue(index, argc, argv, argument));
            if (!parsed) {
                throw std::invalid_argument("Unsupported --IO value. Expected fp32 or fp16.");
            }
            ioPrecision = *parsed;
            continue;
        }
        if (!yolox_benchmark::parseCommonOption(argument, index, argc, argv, options)) {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    yolox_benchmark::validateCommonCliOptions(options);
    if (options.precision == yolox_benchmark::Precision::FP32
        && ioPrecision == yolox_benchmark::Precision::FP16) {
        throw std::invalid_argument("--IO fp16 requires --precision fp16.");
    }

    vai::initVAIEngine();
    yolox::Net net(
        {
            .device = vai::getVaiDevice(),
            .wDefaultDtype = toDType(options.precision),
            .oDefaultDtype = toDType(ioPrecision),
        },
        yolox::getConfigCOCO(toTier(options.variant)),
        {
            .imageSize = yolox_benchmark::kInputHeight,
            .confThreshold = yolox_benchmark::kConfidenceThreshold,
            .iouThreshold = yolox_benchmark::kIouThreshold,
        });

    const std::string modelPath = weightsPath(options.variant);
    std::cout << "[vAi] Loading weights: " << modelPath << std::endl;
    yolox::megvii::loadWeights(net, vai::Checkpoint(modelPath.c_str()));

    const std::string runtimeName = "vAi_Vulkan_"
        + yolox_benchmark::precisionDisplayName(options.precision)
        + "_IO" + yolox_benchmark::precisionDisplayName(ioPrecision)
        + "_YOLOX_" + variantToString(options.variant)
        + "_inference";
    BenchmarkConfig config(options.warmup, options.iterations, runtimeName);

    std::vector<float> inputData = yolox_benchmark::preprocessImage(options.imagePath);
    vai::Tensor input(
        {yolox_benchmark::kInputChannels,
         yolox_benchmark::kInputHeight,
         yolox_benchmark::kInputWidth},
        toDType(ioPrecision));
    input.setFloats(std::move(inputData));

    std::vector<float> lastOutput;
    BenchmarkResult result = runInference(net, input, config, lastOutput);
    std::vector<Detection> detections = yolox_benchmark::postprocessOutput(lastOutput);

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
