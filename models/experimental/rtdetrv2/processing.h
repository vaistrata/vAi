#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace model::experimental::rtdetrv2 {
struct Detection
{
    uint32_t label;
    float score, x1, y1, x2, y2;
};
inline std::vector<Detection> postprocess(
    const float* logits, const float* boxes, uint32_t width, uint32_t height, float threshold = 0)
{
    std::vector<float> scores(size_t(300) * 80);
    std::vector<uint32_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < scores.size(); i++)
        scores[i] = 1.f / (1.f + std::exp(-logits[i]));
    auto better = [&](uint32_t a, uint32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); };
    std::partial_sort(order.begin(), order.begin() + 300, order.end(), better);
    std::vector<Detection> result;
    for (uint32_t i = 0; i < 300; i++)
    {
        uint32_t index = order[i], q = index / 80;
        if (scores[index] < threshold)
            continue;
        const auto* b = boxes + size_t(q) * 4;
        result.push_back(
            { index % 80,
              scores[index],
              (b[0] - b[2] * .5f) * width,
              (b[1] - b[3] * .5f) * height,
              (b[0] + b[2] * .5f) * width,
              (b[1] + b[3] * .5f) * height });
    }
    return result;
}
}   // namespace model::experimental::rtdetrv2
