#ifndef EXPERIMENTAL_RTDETRV2_PROCESSING_H
#define EXPERIMENTAL_RTDETRV2_PROCESSING_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

namespace model::experimental::rtdetrv2 {

inline constexpr const char* cocoClasses[] = {
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
    "teddy bear",     "hair drier", "toothbrush"
};

struct Detection
{
    uint32_t label;
    float score, x1, y1, x2, y2;
};

inline std::vector<Detection> postprocess(
    const float* logits, const float* boxes, uint32_t width, uint32_t height, float threshold = 0)
{
    constexpr uint32_t numClasses = std::size(cocoClasses);
    std::vector<float> scores(size_t(300) * numClasses);
    std::vector<uint32_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < scores.size(); i++)
        scores[i] = 1.f / (1.f + std::exp(-logits[i]));
    auto better = [&](uint32_t a, uint32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); };
    std::partial_sort(order.begin(), order.begin() + 300, order.end(), better);
    std::vector<Detection> result;
    for (uint32_t i = 0; i < 300; i++)
    {
        uint32_t index = order[i], q = index / numClasses;
        if (scores[index] < threshold)
            continue;
        const auto* b = boxes + size_t(q) * 4;
        result.push_back(
            { index % numClasses,
              scores[index],
              (b[0] - b[2] * .5f) * width,
              (b[1] - b[3] * .5f) * height,
              (b[0] + b[2] * .5f) * width,
              (b[1] + b[3] * .5f) * height });
    }
    return result;
}

}   // namespace model::experimental::rtdetrv2

#endif   // EXPERIMENTAL_RTDETRV2_PROCESSING_H
