#include "processing.hpp"

#include "vai/utils/image-processor.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace yolox_benchmark {
namespace {

constexpr std::array<std::string_view, kCocoClassCount> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

uint16_t floatToHalfBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    const uint32_t floatExponent = (bits >> 23) & 0xffu;
    if (floatExponent == 0xffu) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        uint32_t payload = mantissa >> 13;
        if (payload == 0) {
            payload = 1;
        }
        return static_cast<uint16_t>(sign | 0x7c00u | payload);
    }

    int32_t exponent = static_cast<int32_t>(floatExponent) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t halfMantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway
            || (remainder == halfway && (halfMantissa & 1u) != 0)) {
            ++halfMantissa;
        }
        return static_cast<uint16_t>(sign | halfMantissa);
    }

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    uint32_t halfExponent = static_cast<uint32_t>(exponent) << 10;
    uint32_t halfMantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u
        || (remainder == 0x1000u && (halfMantissa & 1u) != 0)) {
        ++halfMantissa;
        if (halfMantissa == 0x0400u) {
            halfMantissa = 0;
            halfExponent += 0x0400u;
        }
    }
    return static_cast<uint16_t>(sign | halfExponent | halfMantissa);
}

float halfBitsToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    int32_t exponent = static_cast<int32_t>((value >> 10) & 0x1fu);
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign
                | (static_cast<uint32_t>(exponent + 127 - 15) << 23)
                | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign
            | (static_cast<uint32_t>(exponent + 127 - 15) << 23)
            | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float calculateIoU(const Detection& a, const Detection& b)
{
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float interW = std::max(0.0f, ix2 - ix1);
    const float interH = std::max(0.0f, iy2 - iy1);
    const float intersection = interW * interH;
    const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float unionArea = areaA + areaB - intersection;
    return unionArea <= 0.0f ? 0.0f : intersection / unionArea;
}

std::vector<Detection> applyNms(std::vector<Detection> detections, float iouThreshold)
{
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> result;
    result.reserve(detections.size());
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        result.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].classId != detections[j].classId) {
                continue;
            }
            if (calculateIoU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

std::vector<Detection> postprocess(
    const std::vector<float>& rawData,
    float confidenceThreshold,
    float iouThreshold)
{
    const uint32_t anchorCount = static_cast<uint32_t>(rawData.size() / kOutputRows);

    std::vector<Detection> detections;
    detections.reserve(256);
    for (uint32_t anchor = 0; anchor < anchorCount; ++anchor) {
        const float* row = rawData.data() + static_cast<size_t>(anchor) * kOutputRows;
        float bestClassConfidence = row[5];
        uint32_t bestClass = 0;
        for (uint32_t classId = 1; classId < kCocoClassCount; ++classId) {
            const float confidence = row[5u + classId];
            if (confidence > bestClassConfidence) {
                bestClassConfidence = confidence;
                bestClass = classId;
            }
        }
        const float bestConfidence = row[4] * bestClassConfidence;
        if (bestConfidence >= confidenceThreshold) {
            const float halfWidth = row[2] * 0.5f;
            const float halfHeight = row[3] * 0.5f;
            detections.push_back({
                row[0] - halfWidth,
                row[1] - halfHeight,
                row[0] + halfWidth,
                row[1] + halfHeight,
                bestConfidence,
                bestClass,
            });
        }
    }
    return applyNms(std::move(detections), iouThreshold);
}

} // namespace

std::vector<uint16_t> floatsToHalfBits(const std::vector<float>& values)
{
    std::vector<uint16_t> converted;
    converted.reserve(values.size());
    for (float value : values) {
        converted.push_back(floatToHalfBits(value));
    }
    return converted;
}

std::vector<float> halfBitsToFloats(const uint16_t* data, size_t count)
{
    std::vector<float> converted;
    converted.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        converted.push_back(halfBitsToFloat(data[index]));
    }
    return converted;
}

std::vector<float> preprocessImage(const std::string& path)
{
    return vai::utils::ImageProcessor(path, kInputChannels)
        .letterbox(kInputHeight, vai::utils::Align::TopLeft, 114.0f)
        .bgr()
        .chw();
}

void validateModelOutputSize(size_t outputSize, const std::string& runtimeName)
{
    if (outputSize != kOutputSize) {
        throw std::runtime_error(
            runtimeName + " output contract mismatch. Expected raw [8400,85] ("
            + std::to_string(kOutputSize) + " floats), got "
            + std::to_string(outputSize) + " floats.");
    }
}

std::vector<Detection> postprocessOutput(const std::vector<float>& output)
{
    validateModelOutputSize(output.size(), "YOLOX");
    return postprocess(output, kConfidenceThreshold, kIouThreshold);
}

void printDetectionSummary(const std::vector<Detection>& detections)
{
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();

    std::cout << "\n=== Detections (" << detections.size() << ") ===" << std::endl;
    for (size_t index = 0; index < detections.size(); ++index) {
        const Detection& detection = detections[index];
        if (detection.classId >= kCocoClasses.size()) {
            throw std::out_of_range(
                "invalid COCO class ID: " + std::to_string(detection.classId));
        }
        std::cout << "[" << index << "] " << kCocoClasses[detection.classId]
                  << " (" << std::fixed << std::setprecision(2)
                  << detection.confidence * 100.0f << "%): "
                  << "[" << std::setprecision(1)
                  << detection.x1 << ", "
                  << detection.y1 << ", "
                  << detection.x2 << ", "
                  << detection.y2 << "]" << std::endl;
    }

    std::cout.flags(flags);
    std::cout.precision(precision);
}

} // namespace yolox_benchmark
