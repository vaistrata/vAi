/*=============================================================================
  Written by   ippx123@gmail.com
  Task         Z-Image DiT — runs the Turbo flow-matching sampler on the GPU
               from caption features and noise, writes the final latent.
  Assets       data/zimage/<name>[-q4][-split].safetensors, the prepared
               weights: written from the checkpoint on the first run (see
               tongyi-weights.h), read on every later one
               data/zimage/demo-input.safetensors
                 from  examples/zimage/tools.py prepare   (caption features of
                       one prompt via the Qwen3 encoder, plus the noise latent)
               data/zimage/demo-latent.bin  (written here; fp32 [16, H, W])
                 to    examples/zimage/tools.py decode    (VAE -> PNG)
  Options      --q4                  block projections as Q4 (~3.7 GB of weights)
               --nodes-per-submit N  split each forward into submissions of N
                                     nodes (mobile GPU watchdogs)
               --split-ffn-up        halve the longest node, for the same reason
=============================================================================*/

#include "models/zimage/model.h"
#include "models/zimage/tongyi-weights.h"
#include "vai/engine/safetensors-parser.h"
#include "vai/utils/time-checker.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using namespace vai;


// One fp32 tensor out of a safetensors file, as host floats.
static std::vector<float> readF32(const SafetensorsParser& parser, const char* key)
{
    VAI_ASSERT(parser.getDataType(key) == "F32");
    std::vector<float> v(parser.byteSize(key) / sizeof(float));
    parser.readInto(key, v.data());
    return v;
}


int main(int argc, char** argv)
{
    bool q4 = false, splitFfnUp = false;
    uint32_t nodesPerSubmit = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--q4"))                                  q4 = true;
        else if (!strcmp(argv[i], "--split-ffn-up"))                     splitFfnUp = true;
        else if (!strcmp(argv[i], "--nodes-per-submit") && i + 1 < argc) nodesPerSubmit = (uint32_t)atoi(argv[++i]);
        else
        {
            fprintf(stderr, "usage: %s [--q4] [--nodes-per-submit N] [--split-ffn-up]\n", argv[0]);
            return 1;
        }
    }

    eva::Device device = getVaiDevice();

    // Inputs: caption features [Nc0, 2560] and noise [16, H, W]
    SafetensorsParser input(VAI_DATA_DIR "/zimage/demo-input.safetensors");
    const auto capShape   = input.getShape("cap_feats");
    const auto noiseShape = input.getShape("noise");
    const uint32_t H = noiseShape[1], W = noiseShape[2];
    Tensor capFeats = Tensor(input["cap_feats"], Dtype::Fp32);
    std::vector<float> latent = readF32(input, "noise");
    printf("Caption tokens: %u, latent: %ux%ux%u\n", capShape[0], noiseShape[0], H, W);

    auto modelCfg = zimage::ModelConfig::Turbo();
    const uint32_t Nx = zimage::padTo((H / modelCfg.patchSize) * (W / modelCfg.patchSize), modelCfg.seqMultiple);
    const uint32_t Nc = zimage::padTo(capShape[0], modelCfg.seqMultiple);

    printf("Loading %s DiT%s...\n", modelCfg.name, q4 ? " (Q4 block weights)" : "");
    zimage::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = nodesPerSubmit,
    }, modelCfg, {
        .blockWeightDtype = q4 ? Dtype::Q4 : Dtype::None,
        .splitFfnUp = splitFfnUp,
    });

    {
        TimeChecker timer("Weights loaded in {}ms");
        if (!zimage::tongyi::loadWeightsPrepared(model))
            printf("Wrote %s; later runs load it instead of the checkpoint\n",
                   zimage::tongyi::preparedPath(model).c_str());
    }

    // FlowMatchEulerDiscreteScheduler, Turbo settings: sigmas linspace(1, 1/n)
    // under the static shift 3, then 0. The DiT sees t = 1 - sigma and
    // predicts -velocity.
    const uint32_t numSteps = 9;
    const float shift = 3.0f;
    std::vector<float> sigmas(numSteps + 1, 0.0f);
    for (uint32_t i = 0; i < numSteps; ++i)
    {
        float s = 1.0f + (1.0f / numSteps - 1.0f) * float(i) / float(numSteps - 1);
        sigmas[i] = shift * s / (1.0f + (shift - 1.0f) * s);
    }

    // First forward records the graph and uploads the weights.
    Tensor out;
    {
        TimeChecker timer("Warm-up forward in {}ms");
        model(zimage::makeInputs(modelCfg, latent.data(), H, W, capFeats, 0.0f), {{"outPatches", &out}});
        out.downloadToHost();
    }

    {
        TimeChecker total("\nSampled {} steps in {}ms", numSteps);
        for (uint32_t i = 0; i < numSteps; ++i)
        {
            TimeChecker step("  step {} (sigma {}): {}ms", i, sigmas[i]);   // TimeChecker takes bare {} only
            model(zimage::makeInputs(modelCfg, latent.data(), H, W, capFeats, 1.0f - sigmas[i]),
                  {{"outPatches", &out}});
            std::vector<float> v = zimage::unpatchify(modelCfg, out.getData<float>(), H, W);

            const float dt = sigmas[i + 1] - sigmas[i];
            for (size_t j = 0; j < latent.size(); ++j)
                latent[j] += dt * -v[j];
        }
    }

    const char* outPath = VAI_DATA_DIR "/zimage/demo-latent.bin";
    std::ofstream(outPath, std::ios::binary).write(
        reinterpret_cast<const char*>(latent.data()), latent.size() * sizeof(float));
    printf("Latent written to %s (decode: examples/zimage/tools.py decode)\n", outPath);
    return 0;
}
