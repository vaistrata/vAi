/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         PaliGemma — answers one prompt about one image, profiles the
               image prefill and the decode.
  Assets       data/tokenizers/paligemma/tokenizer.json
                 from  huggingface.co/google/paligemma-3b-mix-224
               data/images/dog.jpg
                 ships with the repo
=============================================================================*/

#include "net.h"
#include "google-weights.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
namespace paligemma = model::production::paligemma;


/////////////////////////////////////////////////////////////////////////////////////////
// Tag nodes for profiling — group by kernel shape, so identical-shape ops across
// a tower collapse to one row per distinct kernel.
/////////////////////////////////////////////////////////////////////////////////////////
void setProfileTags(paligemma::Net& net)
{
    auto lin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    // Vision tower — SigLIP, one shape for every layer.
    const auto& modelCfg = net.modelCfg;
    const auto& vitCfg = modelCfg.visionCfg;
    const uint32_t Dv = vitCfg.layerCfg.embedDim;         // 1152
    const uint32_t Dvff = vitCfg.layerCfg.ffnHiddenDim;   // 4304

    for (auto& b : net.encoder.layers)
    {
        b.norm1.setProfileTag("Vision.LayerNorm");
        b.mhAttn.projQ.setProfileTag(lin("Vision", Dv, Dv));
        b.mhAttn.projK.setProfileTag(lin("Vision", Dv, Dv));
        b.mhAttn.projV.setProfileTag(lin("Vision", Dv, Dv));
        b.mhAttn.attn.setProfileTag("Vision.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Vision", Dv, Dv));
        b.norm2.setProfileTag("Vision.LayerNorm");
        b.ffn.fc1.setProfileTag(lin("Vision", Dv, Dvff));
        b.ffn.fc2.setProfileTag(lin("Vision", Dvff, Dv));
    }
    net.patchEmbed.perPatches.setProfileTag("Vision.Permute");
    net.patchEmbed.project.setProfileTag(lin("Vision", vitCfg.imageChannels * vitCfg.patchSize * vitCfg.patchSize, Dv));
    net.encoder.finalNorm.setProfileTag("Vision.LayerNorm");
    net.projector.setProfileTag(lin("Vision", Dv, modelCfg.languageCfg.layerCfg.embedDim));

    // Text tower — Gemma 2B.
    const auto& langCfg = modelCfg.languageCfg.layerCfg;
    const uint32_t D = langCfg.embedDim;                 // 2048
    const uint32_t Hq = langCfg.numQHeads;               //    8
    const uint32_t Hkv = langCfg.numKVHeads;             //    1 (GQA)
    const uint32_t Dh = langCfg.headDim;                 //  256
    const uint32_t Dq = Hq * Dh;                         // 2048  (Q projection)
    const uint32_t Dkv = Hkv * Dh;                       //  256  (K/V projection)
    const uint32_t Dff = langCfg.ffnHiddenDim;           // 16384
    const uint32_t V = modelCfg.languageCfg.vocabSize;   // 257152

    for (auto& b : net.decoder.layers)
    {
        b.norm1.setProfileTag("Text.RmsNorm");
        b.mhAttn.projQ.setProfileTag(lin("Text", D, Dq));
        b.mhAttn.projK.setProfileTag(lin("Text", D, Dkv));
        b.mhAttn.projV.setProfileTag(lin("Text", D, Dkv));
        b.mhAttn.ropeQ->setProfileTag("Text.Rope");
        b.mhAttn.ropeK->setProfileTag("Text.Rope");
        b.mhAttn.attn.setProfileTag("Text.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Text", Dq, D));
        b.norm2.setProfileTag("Text.RmsNorm");
        b.ffn.gateValue.setProfileTag(lin("Text", D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag("Text.GatedMul");
        b.ffn.out.setProfileTag(lin("Text", Dff, D));
    }
    net.embedding.setProfileTag("Text.Embedding");
    net.decoder.finalNorm.setProfileTag("Text.RmsNorm");
    net.lmHead.setProfileTag(lin("Text", D, V));
    net.topkSampling.setProfileTag("Text.Sampling");
}


int main()
{
    // Load model
    auto modelCfg = paligemma::ModelConfig::_3B_224();
    printf("Loading PaliGemma-%s model...\n", modelCfg.variant);
    paligemma::Net net(
        {
            .wDefaultDtype = Dtype::Fp16,
        },
        modelCfg,
        {
            .maxRuntimeContext = 2048,
        });

    // Configure the per-node GPU execution-time profiler
    setProfileTags(net);

    // Load weights
    printf("Loading weights...\n");
    {
        utils::TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"google">(net);
    }

    // Load tokenizer
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/paligemma/tokenizer.json");

    std::string imagePath = VAI_DATA_DIR "/images/dog.jpg";
    const uint32_t C = modelCfg.visionCfg.imageChannels;
    const uint32_t S = modelCfg.visionCfg.imageSize;

    // Preprocess the image
    printf("Loading image: %s\n", imagePath.c_str());
    std::vector<float> pixels =
        utils::ImageProcessor(imagePath)
            .resize(
                S,
                utils::Resize::
                    BilinearAlignCorners)   // SigLIP squashes the whole image to its trained square and scales it to [-1, 1].
            .normalize(-1.0f, 1.0f)
            .chw();

    // Prepare prompt and encode it to token ids
    std::string prompt = "caption en";
    printf("\nPrompt: \"%s\"\n", prompt.c_str());
    std::vector<int> ids = tokenizer.encode(prompt + "\n", true);
    const uint32_t promptLength = (uint32_t)ids.size();

    // Initialize the input tensors
    Tensor inImage = Tensor({ C, S, S }).setFloats(std::move(pixels));
    Tensor promptTokenIds = Tensor({ promptLength }).setData(ids.data());
    Tensor nextTokenId;

    // Warm-up
    {
        net.prefill(inImage, promptTokenIds, &nextTokenId);
        net.decode(&nextTokenId);
    }

    net.resetCache();
    net.resetProfileRecords();

    // Prefill
    {
        utils::TimeChecker totalTimer("\n"
                                      "────────────────────────────────────────\n"
                                      "▶ Prefill {} image + {} prompt tokens in {}ms \n"
                                      "────────────────────────────────────────",
                                      modelCfg.visionCfg.numPatches(),
                                      promptLength);

        net.prefill(inImage, promptTokenIds, &nextTokenId);
        ids.push_back(nextTokenId.getData<uint32_t>()[0]);
    }

    net.printProfile();

    // Decode
    {
        const int maxNewTokens = 128;

        utils::TimeChecker totalTimer("\n"
                                      "────────────────────────────────────────\n"
                                      "▶ Decode {} tokens in {}ms \n"
                                      "────────────────────────────────────────",
                                      maxNewTokens);

        for (int gen = 0; gen < maxNewTokens; gen++)
        {
            net.decode(&nextTokenId);
            ids.push_back(nextTokenId.getData<uint32_t>()[0]);
        }
    }

    net.printProfile();

    // Print the answer — what the model wrote after the prompt, up to <eos>.
    printf("\n=== Answer ===\n");
    const int eosId = tokenizer.pieceToId("<eos>");
    auto first = ids.begin() + promptLength;
    auto last = std::find(first, ids.end(), eosId);
    printf("%s\n", tokenizer.decode(std::vector<int>(first, last)).c_str());
    printf("\n--- Generation complete ---\n");

    return 0;
}
