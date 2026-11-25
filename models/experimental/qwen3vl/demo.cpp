/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         Qwen3-VL — answers one prompt about one image, profiles the
               image prefill and the decode.
  Assets       data/tokenizers/qwen3vl/tokenizer.json
                 from  huggingface.co/Qwen/Qwen3-VL-2B-Instruct
               data/images/dog.jpg
                 ships with the repo
=============================================================================*/

#include "net.h"
#include "qwen-weights.h"
#include "processing.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
namespace qwen3vl = model::experimental::qwen3vl;


/////////////////////////////////////////////////////////////////////////////////////////
// Tag nodes for profiling — group by kernel shape, so identical-shape ops across
// a tower collapse to one row per distinct kernel.
/////////////////////////////////////////////////////////////////////////////////////////
void setProfileTags(qwen3vl::Net& net)
{
    auto lin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    // Vision tower
    const auto& vitCfg = net.modelCfg.vision.trf;
    const uint32_t Dv = vitCfg.embedDim;
    const uint32_t Dvff = vitCfg.ffnHiddenDim;
    const uint32_t D = net.modelCfg.trf.embedDim;

    auto& vision = net.visionEncoder;
    for (auto& b : vision.encoder.blocks)
    {
        b.norm1.setProfileTag("Vision.LayerNorm");
        b.mhAttn.projQKV.setProfileTag(lin("Vision", Dv, 3 * Dv));
        b.mhAttn.ropeQ->setProfileTag("Vision.Rope");
        b.mhAttn.ropeK->setProfileTag("Vision.Rope");
        b.mhAttn.attn.setProfileTag("Vision.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Vision", Dv, Dv));
        b.norm2.setProfileTag("Vision.LayerNorm");
        b.ffn.fc1.setProfileTag(lin("Vision", Dv, Dvff));
        b.ffn.fc2.setProfileTag(lin("Vision", Dvff, Dv));
    }
    vision.patchEmbed.perPatches.setProfileTag("Vision.Permute");
    vision.patchEmbed.project.setProfileTag(
        lin("Vision", vitCfg.imageChannels * vitCfg.patchSize * vitCfg.patchSize, Dv));

    vision.encoder.finalNorm.setProfileTag("Vision.LayerNorm");
    vision.connector.perBlocks.setProfileTag("Vision.Permute");
    vision.connector.projector.fc1.setProfileTag(lin("Vision", 4 * Dv, 4 * Dv));
    vision.connector.projector.fc2.setProfileTag(lin("Vision", 4 * Dv, D));
    for (auto& c : vision.deepConnectors)
    {
        c.perBlocks.setProfileTag("Vision.Permute");
        c.tokenNorm->setProfileTag("Vision.LayerNorm");
        c.projector.fc1.setProfileTag(lin("Vision", 4 * Dv, 4 * Dv));
        c.projector.fc2.setProfileTag(lin("Vision", 4 * Dv, D));
    }
    net.imageCopy.setProfileTag("Vision.IndexCopy");

    // Text tower
    const auto& langCfg = net.modelCfg.trf;
    const uint32_t Hq = langCfg.numQHeads;
    const uint32_t Hkv = langCfg.numKVHeads;
    const uint32_t Dh = langCfg.headDim;
    const uint32_t Dq = Hq * Dh;     // Q projection
    const uint32_t Dkv = Hkv * Dh;   // K/V projection
    const uint32_t Dff = langCfg.ffnHiddenDim;
    const uint32_t V = net.modelCfg.vocabSize;

    for (auto& b : net.decoder.blocks)
    {
        b.norm1.setProfileTag("Text.RmsNorm");
        b.mhAttn.projQ.setProfileTag(lin("Text", D, Dq));
        b.mhAttn.projK.setProfileTag(lin("Text", D, Dkv));
        b.mhAttn.projV.setProfileTag(lin("Text", D, Dkv));
        b.mhAttn.normQ->setProfileTag("Text.RmsNorm.QK");
        b.mhAttn.normK->setProfileTag("Text.RmsNorm.QK");
        b.mhAttn.ropeQ->setProfileTag("Text.Rope");
        b.mhAttn.ropeK->setProfileTag("Text.Rope");
        b.mhAttn.attn.setProfileTag("Text.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Text", Dq, D));
        b.norm2.setProfileTag("Text.RmsNorm");
        b.ffn.gateValue.setProfileTag(lin("Text", D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag("Text.GatedMul");
        b.ffn.out.setProfileTag(lin("Text", Dff, D));
    }
    for (auto& add : net.decoder.deepAdds)
        add.setProfileTag("Text.IndexAdd");
    net.embedding.setProfileTag("Text.Embedding");
    net.decoder.finalNorm.setProfileTag("Text.RmsNorm");
    net.lmHead.setProfileTag(lin("Text", D, V));
    net.topkSampling.setProfileTag("Text.Sampling");
}


int main()
{
    // Load model
    auto modelCfg = qwen3vl::ModelConfig::get("2B");
    printf("Loading Qwen3-VL-%s model...\n", modelCfg.variant);
    // clang-format off
    qwen3vl::Net net({
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg, {
        .maxRuntimeContext = 2048,
    });
    // clang-format on

    // Configure the per-node GPU execution-time profiler
    setProfileTags(net);

    // Load weights
    printf("Loading weights...\n");
    {
        utils::TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"qwen">(net);
    }

    // Load tokenizer
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/qwen3vl/tokenizer.json");

    const char* imagePaths[] = {
        VAI_DATA_DIR "/images/dog.jpg",
    };
    const uint32_t B = (uint32_t)std::size(imagePaths);
    const uint32_t C = net.modelCfg.vision.trf.imageChannels;
    const uint32_t S = net.runtimeCfg.imageSize;

    // Preprocess the images and stack them into [B, C, S, S]
    for (uint32_t b = 0; b < B; ++b)
        printf("Loading image %u: %s\n", b, imagePaths[b]);
    std::vector<float> pixels = qwen3vl::loadImages({ std::begin(imagePaths), std::end(imagePaths) }, S);

    // Prepare prompt and encode it to token ids
    std::string text = "Describe this image.";
    printf("\nPrompt: \"%s\"\n", text.c_str());
    std::string prompt =
        qwen3vl::expandImagePlaceholders(qwen3vl::chatPrompt(text, B), net.visionEncoder.tokensPerImage);
    std::vector<int> ids = tokenizer.encode(prompt);
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
        utils::TimeChecker totalTimer(
            "\n"
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

        utils::TimeChecker totalTimer(
            "\n"
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

    // Print the answer — what the model wrote after the prompt, up to <|im_end|>.
    printf("\n=== Answer ===\n");
    const int eosId = tokenizer.pieceToId("<|im_end|>");
    auto first = ids.begin() + promptLength;
    auto last = std::find(first, ids.end(), eosId);
    printf("%s\n", tokenizer.decode(std::vector<int>(first, last)).c_str());
    printf("\n--- Generation complete ---\n");

    return 0;
}
