#ifndef PRODUCTION_GEMMA3_PROCESSING_H
#define PRODUCTION_GEMMA3_PROCESSING_H

#include "vai/utils/image-processor.h"
#include <cstdint>
#include <string>
#include <vector>

namespace model::production::gemma3 {


// Replace: <start_of_image> ->
// [HuggingFace] \n\n<start_of_image><image_soft_token>...<image_soft_token><end_of_image>\n\n
// [Llama.cpp] <start_of_image><image_soft_token>...<image_soft_token><end_of_image>
inline std::string expandImagePlaceholders(const std::string& prompt, uint32_t tokensPerImage)
{
    const std::string boi = "<start_of_image>";

    std::string expanded = "\n\n" + boi;
    for (uint32_t i = 0; i < tokensPerImage; ++i)
        expanded += "<image_soft_token>";
    expanded += "<end_of_image>\n\n";

    std::string out;
    size_t last = 0;
    for (size_t pos; (pos = prompt.find(boi, last)) != std::string::npos; last = pos + boi.size())
        out += prompt.substr(last, pos - last) + expanded;

    return out + prompt.substr(last);
}


// Each <image:file> becomes <start_of_image>, and its file goes into paths.
inline std::string takeImageTags(const std::string& message, std::vector<std::string>& paths)
{
    const std::string open = "<image:";

    std::string out;
    size_t last = 0;
    for (size_t pos; (pos = message.find(open, last)) != std::string::npos;)
    {
        const size_t close = message.find('>', pos);
        if (close == std::string::npos)
            break;
        out += message.substr(last, pos - last) + "<start_of_image>";
        paths.push_back(message.substr(pos + open.size(), close - pos - open.size()));
        last = close + 1;
    }

    return out + message.substr(last);
}


// The images stacked as [B, 3, imageSize, imageSize].
inline std::vector<float> loadImages(const std::vector<std::string>& paths, uint32_t imageSize)
{
    std::vector<float> pixels;
    pixels.reserve(paths.size() * 3 * imageSize * imageSize);
    for (const std::string& path : paths)
    {
        std::vector<float> chw =
            vai::utils::ImageProcessor(path)
                .resize(imageSize,
                        vai::utils::Resize::
                            BilinearAntialiasInt16)   // HF Gemma3ImageProcessor: torchvision uint8 antialias
                .normalize(-1.0f, 1.0f)
                .chw();
        pixels.insert(pixels.end(), chw.begin(), chw.end());
    }

    return pixels;
}


}   // namespace model::production::gemma3

#endif   // PRODUCTION_GEMMA3_PROCESSING_H
