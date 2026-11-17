/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         pi0.5 on two nets — turns three camera views, a task and the robot's
               state into one action chunk, profiles the prefill and the denoise,
               then times prefill - decode rounds: nothing re-records between them.
  Assets       data/tokenizers/paligemma/tokenizer.json
                 from  huggingface.co/google/paligemma-3b-mix-224
               data/images/cam_high.jpg, cam_left_wrist.jpg, cam_right_wrist.jpg
                 from  huggingface.co/datasets/suz22/RoboTwin_clean
=============================================================================*/

#include "net.h"
#include "lerobot-weights.h"
#include "processing.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
namespace pi05 = model::experimental::pi05;


/////////////////////////////////////////////////////////////////////////////////////////
// Tag nodes for profiling — group by kernel shape, so identical-shape ops across
// a tower collapse to one row per distinct kernel.
/////////////////////////////////////////////////////////////////////////////////////////
void setProfileTags(pi05::Net& net)
{
    auto lin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) +
               "]";
    };

    // Vision tower — SigLIP, one shape for every block.
    const auto& modelCfg = net.modelCfg;
    const auto& vitCfg = modelCfg.visionTrf;
    const uint32_t Dv = vitCfg.embedDim;         // 1152
    const uint32_t Dvff = vitCfg.ffnHiddenDim;   // 4304

    for (auto& b : net.visionEncoder.blocks)
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
    net.patchEmbed.project.setProfileTag(
        lin("Vision", vitCfg.imageChannels * vitCfg.patchSize * vitCfg.patchSize, Dv));
    net.visionEncoder.finalNorm.setProfileTag("Vision.LayerNorm");
    net.projector.setProfileTag(lin("Vision", Dv, modelCfg.languageTrf.embedDim));

    // The two Gemma stacks — same tag scheme, and their widths tell them apart.
    const auto& langCfg = modelCfg.languageTrf;
    const uint32_t Dq = langCfg.numQHeads * langCfg.headDim;     // 2048  (Q projection)
    const uint32_t Dkv = langCfg.numKVHeads * langCfg.headDim;   //  256  (K/V projection)

    auto tagGemma = [&](auto& gemma, const char* tower, uint32_t D, uint32_t Dff) {
        for (auto& b : gemma.blocks)
        {
            b.norm1.setProfileTag(std::string(tower) + ".RmsNorm");
            b.mhAttn.projQ.setProfileTag(lin(tower, D, Dq));
            b.mhAttn.projK.setProfileTag(lin(tower, D, Dkv));
            b.mhAttn.projV.setProfileTag(lin(tower, D, Dkv));
            b.mhAttn.ropeQ->setProfileTag(std::string(tower) + ".Rope");
            b.mhAttn.ropeK->setProfileTag(std::string(tower) + ".Rope");
            b.mhAttn.attn.setProfileTag(std::string(tower) + ".Attn");
            b.mhAttn.projOut.setProfileTag(lin(tower, Dq, D));
            b.norm2.setProfileTag(std::string(tower) + ".RmsNorm");
            b.ffn.gateValue.setProfileTag(lin(tower, D, 2 * Dff));
            b.ffn.gatedMul.setProfileTag(std::string(tower) + ".GatedMul");
            b.ffn.out.setProfileTag(lin(tower, Dff, D));
        }
        gemma.finalNorm.setProfileTag(std::string(tower) + ".RmsNorm");
    };

    tagGemma(net.languageEncoder, "Language", langCfg.embedDim, langCfg.ffnHiddenDim);
    tagGemma(
        net.actionDecoder, "Action", modelCfg.actionTrf.embedDim, modelCfg.actionTrf.ffnHiddenDim);

    // The expert's gated residuals, EleAffine in place of Gemma's folded adds.
    for (auto& b : net.actionDecoder.blocks)
    {
        b.fma1.setProfileTag("Action.EleAffine");
        b.fma2.setProfileTag("Action.EleAffine");
    }

    net.promptEmbed.setProfileTag("Language.Embedding");
    net.actionProj.setProfileTag("Action.ActionProj");
    net.condTable.setProfileTag("Action.CyclicTableReader");
    net.velocityHead.setProfileTag("Action.VelocityHead");
}


int main()
{
    // Construct model
    auto modelCfg = pi05::ModelConfig::_3B();
    printf("Loading pi0.5-%s model...\n", modelCfg.variant);
    // clang-format off
    pi05::Net net({
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg);
    // clang-format on

    // Configure the per-node GPU execution-time profiler
    setProfileTags(net);

    // Load weights
    printf("Loading weights...\n");
    {
        utils::TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"lerobot">(net);
    }

    // Load tokenizer
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/paligemma/tokenizer.json");

    const char* imagePaths[] = {
        VAI_DATA_DIR "/images/cam_high.jpg",
        VAI_DATA_DIR "/images/cam_left_wrist.jpg",
        VAI_DATA_DIR "/images/cam_right_wrist.jpg",
    };
    const uint32_t B = (uint32_t)std::size(imagePaths);
    const uint32_t C = modelCfg.visionTrf.imageChannels;
    const uint32_t S = modelCfg.visionTrf.imageSize;

    // Preprocess the images and stack them into [B, C, S, S]
    std::vector<float> pixels;
    pixels.reserve(size_t(B) * C * S * S);
    for (uint32_t k = 0; k < B; ++k)
    {
        printf("Loading image %u: %s\n", k, imagePaths[k]);
        // clang-format off
        std::vector<float> chw = utils::ImageProcessor(imagePaths[k])
            .letterbox(S, utils::Align::Center, 0.0f)  // LeRobot resize_with_pad_torch: no antialias, unlike openpi's JAX
            .normalize(-1.0f, 1.0f)
            .chw();
        // clang-format on
        pixels.insert(pixels.end(), chw.begin(), chw.end());
    }

    // The state is the robot's joint positions, padded to actionDim.
    std::vector<float> states(modelCfg.actionDim, 0.0f);
    for (uint32_t i = 0; i < 6; ++i)
        states[i] = 0.1f * (i + 1);

    // Prepare prompt and encode it to token ids
    std::string prompt =
        "Take the shoe with patterned bottom from the table and place it on the mat";
    printf("\nTask: \"%s\"\n", prompt.c_str());
    std::vector<int> ids = tokenizer.encode(pi05::makePrompt(prompt, states), true);
    const uint32_t promptLength = (uint32_t)ids.size();

    // Initialize the input tensors
    Tensor inImages = Tensor({ B, C, S, S }).setFloats(std::move(pixels));
    Tensor promptTokenIds = Tensor({ promptLength }).setData(ids.data());
    Tensor nextAction =
        Tensor({ modelCfg.actionChunk, modelCfg.actionDim })
            .setFloats(std::vector<float>(size_t(modelCfg.actionChunk) * modelCfg.actionDim));

    // Warm-up.
    {
        net.prefill(inImages, promptTokenIds);
        net.decode(&nextAction);
    }

    net.resetProfileRecords();

    // Prefill — every image and the prompt in one forward pass, filling the KV cache.
    {
        utils::TimeChecker totalTimer(
            "\n"
            "────────────────────────────────────────\n"
            "▶ Prefill {} image + {} prompt tokens in {}ms \n"
            "────────────────────────────────────────",
            B * modelCfg.visionTrf.numPatches(),
            promptLength);

        net.prefill(inImages, promptTokenIds);

        net.sync();   // this sync is not necessary, but it is here to measure the prefill time.
    }

    net.prefillNet.printProfile();

    // Denoise — Euler from noise at t = 1 down to the action chunk at t = 0.
    nextAction.setFloats(pi05::sampleNoise(size_t(modelCfg.actionChunk) * modelCfg.actionDim, 0));
    {
        const uint32_t steps = net.runtimeCfg.denoiseSteps;

        utils::TimeChecker totalTimer(
            "\n"
            "────────────────────────────────────────\n"
            "▶ Denoise {} steps over a {}-step chunk in {}ms \n"
            "────────────────────────────────────────",
            steps,
            modelCfg.actionChunk);

        net.decode(&nextAction);

        net.sync();   // this sync is not necessary, but it is here to measure the denoise time.
    }

    net.decodeNet.printProfile();

    // Rounds — prefill then denoise, as a control loop runs them. Each net keeps
    // its recording, so a round costs its GPU time alone.
    for (uint32_t round = 0; round < 3; ++round)
    {
        nextAction.setFloats(
            pi05::sampleNoise(size_t(modelCfg.actionChunk) * modelCfg.actionDim, 0));

        utils::TimeChecker roundTimer("▶ Round {}: prefill + denoise in {}ms", round);

        net.prefill(inImages, promptTokenIds);
        net.decode(&nextAction);

        net.sync();
    }

    // Print the chunk — the first few steps and dims.
    printf(
        "\n=== Action chunk (%u steps x %u dims) ===\n", modelCfg.actionChunk, modelCfg.actionDim);
    const uint32_t showSteps = std::min(modelCfg.actionChunk, 5u);
    const uint32_t showDims = std::min(modelCfg.actionDim, 8u);
    const float* a = nextAction.getData<float>();
    for (uint32_t s = 0; s < showSteps; ++s)
    {
        printf("  t%-2u ", s);
        for (uint32_t d = 0; d < showDims; ++d)
            printf("% .4f ", a[size_t(s) * modelCfg.actionDim + d]);
        printf("...\n");
    }
    printf("  ...\n");

    printf("\n--- Action chunk complete ---\n");

    return 0;
}


/*
  Reference chunk: models/production/pi05/demo.cpp. The two nets run the same
  graph on the same cache, so this demo prints the same chunk.
*/
