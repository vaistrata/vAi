/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         Gemma 3 4B — completes one prompt over images, profiles prefill and decode.
  Assets       data/tokenizers/gemma3/tokenizer.json
                 from  huggingface.co/google/gemma-3-[1b|4b]-it
                 or  huggingface.co/unsloth/gemma-3-[1b|4b]-it
               data/images/paper_captured0.jpg, paper_captured1.jpg, paper_captured2.jpg, paper_captured3.jpg
=============================================================================*/

#include "net.h"
#include "google-weights.h"
#include "processing.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;
namespace gemma3 = model::production::gemma3;


void setProfileTags(gemma3::Net& net)
{
    auto towerLin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };
    const auto& decoderCfg = net.modelCfg.decoder;
    const uint32_t D = decoderCfg.embedDim;
    const uint32_t H = decoderCfg.numQHeads;
    const uint32_t Hkv = decoderCfg.numKVHeads;
    const uint32_t Dh = decoderCfg.headDim;
    const uint32_t Dq = H * Dh;
    const uint32_t Dkv = Hkv * Dh;
    const uint32_t Dff = decoderCfg.ffnHiddenDim;
    const uint32_t V = net.modelCfg.vocabSize;

    // Vision encoder
    if (net.visionEncoder)
    {
        const auto& v = *net.modelCfg.vision;
        const auto& visionLayer = v.layerCfg;
        auto& vision = net.visionEncoder.value();

        for (auto& b : vision.encoder.layers)
        {
            b.norm1.setProfileTag("Vision.LayerNorm");
            b.mhAttn.projQ.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.mhAttn.projK.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.mhAttn.projV.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.mhAttn.attn.setProfileTag("Vision.Attn");
            b.mhAttn.projOut.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.norm2.setProfileTag("Vision.LayerNorm");
            b.ffn.fc1.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.ffnHiddenDim));
            b.ffn.fc2.setProfileTag(towerLin("Vision", visionLayer.ffnHiddenDim, visionLayer.embedDim));
        }

        vision.patchEmbed.perPatches.setProfileTag("Vision.Permute");

        vision.patchEmbed.project.setProfileTag(
            towerLin("Vision", v.imageChannels * v.patchSize * v.patchSize, visionLayer.embedDim));

        vision.encoder.finalNorm.setProfileTag("Vision.LayerNorm");

        vision.pool.setProfileTag("Vision.Pool");
        vision.imageNorm.setProfileTag("Vision.RmsNorm");
        net.projector->setProfileTag(towerLin("Vision", visionLayer.embedDim, decoderCfg.embedDim));

#ifdef TEST_INDEX_COPY
        net.imageCopy.setProfileTag("Vision.IndexCopy");
#else
        net.imageScatter.setProfileTag("Vision.RowSliceScatter");
#endif
    }

    // Text decoder
    const std::string textPrefix = net.visionEncoder ? "Text." : "";

    auto textTag = [&textPrefix](const char* tag) { return textPrefix + tag; };

    auto textLin = [&textPrefix](uint32_t in, uint32_t out) {
        return textPrefix + "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < decoderCfg.nBlocks; ++i)
    {
        auto& b = net.decoder.blocks[i];
        b.norm1.setProfileTag(textTag("RmsNorm"));
        b.mhAttn.projQ.setProfileTag(textLin(D, Dq));
        b.mhAttn.projK.setProfileTag(textLin(D, Dkv));
        b.mhAttn.projV.setProfileTag(textLin(D, Dkv));
        b.mhAttn.normQ->setProfileTag(textTag("RmsNorm.QK"));
        b.mhAttn.normK->setProfileTag(textTag("RmsNorm.QK"));
        b.mhAttn.ropeQ->setProfileTag(textTag("Rope"));
        b.mhAttn.ropeK->setProfileTag(textTag("Rope"));

        b.mhAttn.attn.setProfileTag(textTag(gemma3::group::isGlobalBlock(i) ? "Attn.global" : "Attn.local"));

        b.mhAttn.projOut.setProfileTag(textLin(Dq, D));
        b.norm2.setProfileTag(textTag("RmsNorm"));
        b.norm3.setProfileTag(textTag("RmsNorm"));
        b.ffn.gateValue.setProfileTag(textLin(D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag(textTag("GatedMul"));
        b.ffn.out.setProfileTag(textLin(Dff, D));
        b.norm4.setProfileTag(textTag("RmsNorm"));
    }

    if (net.visionEncoder)
        net.embedding.setProfileTag("Text.Embedding");
    net.decoder.finalNorm.setProfileTag(textTag("RmsNorm"));
    net.lmHead.setProfileTag(textLin(D, V));
    net.topkSampling.setProfileTag(textTag("Sampling"));
}


int main()
{
    // Load model
    auto modelCfg = gemma3::ModelConfig::get("4B");
    printf("Loading Gemma-%s model...\n", modelCfg.variant);
    gemma3::Net net(
        {
            .wDefaultDtype = Dtype::Fp16,
        },
        modelCfg,
        {
            .maxRuntimeContext = 4096,
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
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/gemma3/tokenizer.json");

    const char* imagePaths[] = {
        VAI_DATA_DIR "/images/paper_captured0.jpg",
        // VAI_DATA_DIR "/images/paper_captured1.jpg",
        // VAI_DATA_DIR "/images/paper_captured2.jpg",
        // VAI_DATA_DIR "/images/paper_captured3.jpg",
    };
    const uint32_t B = (uint32_t)std::size(imagePaths);
    const uint32_t C = modelCfg.vision->imageChannels;
    const uint32_t S = modelCfg.vision->imageSize;

    // Preprocess the images and stack them into [B, C, S, S]
    for (uint32_t b = 0; b < B; ++b)
        printf("Loading image %u: %s\n", b, imagePaths[b]);
    std::vector<float> pixels = gemma3::loadImages({ std::begin(imagePaths), std::end(imagePaths) }, S);

    // Prepare prompt and encode it to token ids
    std::string prompt = "Summarize the image(s).";
    for (uint32_t b = 0; b < B; ++b)
        prompt = "<start_of_image>" + prompt;
    printf("\nPrompt: \"%s\"\n", prompt.c_str());
    prompt = gemma3::expandImagePlaceholders(prompt, modelCfg.vision->tokensPerImage);
    std::vector<int> ids = tokenizer.encode(prompt, true);
    const uint32_t promptLength = (uint32_t)ids.size();

    // Initialize the input tensors
    Tensor inImages = Tensor({ B, C, S, S }).setFloats(std::move(pixels));
    Tensor promptTokenIds = Tensor({ promptLength }).setData(ids.data());
    Tensor nextTokenId;

    // Warm-up
    {
        net.prefill(inImages, promptTokenIds, &nextTokenId);
        net.decode(&nextTokenId);
    }

    net.resetCache();
    net.resetProfileRecords();

    // Prefill
    {
        utils::TimeChecker totalTimer("\n"
                                      "────────────────────────────────────────\n"
                                      "▶ Prefill {} tokens in {}ms \n"
                                      "────────────────────────────────────────",
                                      promptLength);

        net.prefill(inImages, promptTokenIds, &nextTokenId);
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

    // Print final generated text
    printf("\n=== Final Generated Text ===\n");
    std::string generatedText = tokenizer.decode(ids);
    printf("%s\n", generatedText.c_str());
    printf("\n--- Generation complete ---\n");

    return 0;
}
