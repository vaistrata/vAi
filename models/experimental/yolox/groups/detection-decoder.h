#ifndef EXPERIMENTAL_YOLOX_GROUPS_DETECTION_DECODER_H
#define EXPERIMENTAL_YOLOX_GROUPS_DETECTION_DECODER_H

#include "../nodes/decode.h"
#include <string>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// DetectionDecoder - Decode YOLOX predictions into its canonical inference output
// Inputs per scale: regobj [5,H,W], cls [numClasses,H,W]
// Output: out0 [N, 5 + numClasses]
/////////////////////////////////////////////////////////////////////////////////////////
struct DetectionDecoder : public vai::NodeGroup
{
    struct Config
    {
        uint32_t imageSize;
        uint32_t numClasses;
    };

    node::Decode decode;

    DetectionDecoder(Config config)
    : decode({ config.imageSize, config.numClasses })
    {
        for (const char* scale : { "p3", "p4", "p5" })
            for (const char* input : { "regobj", "cls" })
            {
                const std::string slot = std::string(input) + "_" + scale;
                defineSlot(slot, decode.slot(slot));
            }
        defineSlot("out0", decode.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_DETECTION_DECODER_H
