#ifndef EXPERIMENTAL_YOLOX_PROCESSING_H
#define EXPERIMENTAL_YOLOX_PROCESSING_H

#include "vai/utils/image-processor.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace model::experimental::yolox {

constexpr uint32_t NUM_COCO_CLASSES = 80u;

struct Detection
{
    float x1, y1, x2, y2;
    float confidence;
    uint32_t classId;
    std::string className;
};

inline const char* const COCO_CLASSES[NUM_COCO_CLASSES] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};


inline float calculateIoU(const Detection& a, const Detection& b)
{
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float unionArea = areaA + areaB - intersection;
    return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

inline std::vector<Detection> applyNms(std::vector<Detection> detections, float iouThreshold)
{
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppressed[i])
            continue;
        result.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppressed[j] || detections[i].classId != detections[j].classId)
                continue;
            if (calculateIoU(detections[i], detections[j]) > iouThreshold)
                suppressed[j] = true;
        }
    }
    return result;
}


inline std::vector<Detection> postprocess(const std::vector<float>& rawData, const vai::utils::ImageGeometry& geometry,
                                          float confThreshold, float iouThreshold,
                                          uint32_t numClasses = NUM_COCO_CLASSES)
{
    const uint32_t channels = 5u + numClasses;
    if (geometry.scaleX <= 0.0f || numClasses == 0u || rawData.size() % channels != 0u)
        throw std::invalid_argument("invalid YOLOX output or letterbox scale");

    std::vector<Detection> detections;
    const uint32_t anchors = static_cast<uint32_t>(rawData.size() / channels);
    detections.reserve(256);

    for (uint32_t anchor = 0; anchor < anchors; ++anchor)
    {
        const float* row = rawData.data() + static_cast<size_t>(anchor) * channels;
        uint32_t classId = 0u;
        float classConfidence = row[5];
        for (uint32_t c = 1; c < numClasses; ++c)
        {
            if (row[5u + c] > classConfidence)
            {
                classConfidence = row[5u + c];
                classId = c;
            }
        }

        const float confidence = row[4] * classConfidence;
        if (confidence < confThreshold)
            continue;

        const float halfWidth = 0.5f * row[2];
        const float halfHeight = 0.5f * row[3];
        Detection detection;
        detection.x1 = row[0] - halfWidth;
        detection.y1 = row[1] - halfHeight;
        detection.x2 = row[0] + halfWidth;
        detection.y2 = row[1] + halfHeight;
        detection.confidence = confidence;
        detection.classId = classId;
        detection.className = classId < NUM_COCO_CLASSES ? COCO_CLASSES[classId] : std::to_string(classId);
        detections.push_back(std::move(detection));
    }

    detections = applyNms(std::move(detections), iouThreshold);
    for (Detection& detection : detections)
    {
        const float w = static_cast<float>(geometry.sourceWidth);
        const float h = static_cast<float>(geometry.sourceHeight);
        detection.x1 = std::clamp((detection.x1 - geometry.padX) / geometry.scaleX, 0.0f, w);
        detection.y1 = std::clamp((detection.y1 - geometry.padY) / geometry.scaleY, 0.0f, h);
        detection.x2 = std::clamp((detection.x2 - geometry.padX) / geometry.scaleX, 0.0f, w);
        detection.y2 = std::clamp((detection.y2 - geometry.padY) / geometry.scaleY, 0.0f, h);
    }
    return detections;
}

}   // namespace model::experimental::yolox

#endif   // EXPERIMENTAL_YOLOX_PROCESSING_H
