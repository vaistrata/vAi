#ifndef EXPERIMENTAL_QWEN3VL_PROCESSING_H
#define EXPERIMENTAL_QWEN3VL_PROCESSING_H

#include "vai/utils/image-processor.h"
#include <cstdint>
#include <string>
#include <vector>

namespace model::experimental::qwen3vl {


// One user turn of the chat template, its images ahead of the text, the reply opened.
inline std::string chatPrompt(const std::string& text, uint32_t imageCount)
{
    std::string prompt = "<|im_start|>user\n";
    for (uint32_t i = 0; i < imageCount; ++i)
        prompt += "<|vision_start|><|image_pad|><|vision_end|>";
    return prompt + text + "<|im_end|>\n<|im_start|>assistant\n";
}


// Each <|image_pad|> becomes tokensPerImage of them, one per token the merger hands an image.
inline std::string expandImagePlaceholders(const std::string& prompt, uint32_t tokensPerImage)
{
    const std::string pad = "<|image_pad|>";

    std::string expanded;
    for (uint32_t i = 0; i < tokensPerImage; ++i)
        expanded += pad;

    std::string out;
    size_t last = 0;
    for (size_t pos; (pos = prompt.find(pad, last)) != std::string::npos; last = pos + pad.size())
        out += prompt.substr(last, pos - last) + expanded;

    return out + prompt.substr(last);
}


// The images stacked as [B, 3, imageSize, imageSize].
inline std::vector<float> loadImages(const std::vector<std::string>& paths, uint32_t imageSize)
{
    std::vector<float> pixels;
    pixels.reserve(paths.size() * 3 * imageSize * imageSize);
    for (const std::string& path : paths)
    {
        // clang-format off
        std::vector<float> chw = vai::utils::ImageProcessor(path)
            .resize(imageSize, vai::utils::Resize::BicubicAntialiasInt16) // HF Qwen2VLImageProcessorFast: torchvision uint8 bicubic antialias
            .normalize(-1.0f, 1.0f)
            .chw();
        // clang-format on
        pixels.insert(pixels.end(), chw.begin(), chw.end());
    }

    return pixels;
}


}   // namespace model::experimental::qwen3vl

#endif   // EXPERIMENTAL_QWEN3VL_PROCESSING_H
