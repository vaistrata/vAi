/*=============================================================================
  Written by   ippx123@gmail.com
  Task         Z-Image — runs the Turbo flow-matching sampler on the GPU from
               caption features and noise, then decodes the final latent with
               the VAE on the GPU and writes it as a PNG.
  Assets       data/zimage/<name>[-q4][-split].safetensors, the prepared
               weights: written from the checkpoint on the first run (see
               tongyi-weights.h), read on every later one
               data/zimage/demo-input.safetensors
                 from  examples/zimage/tools.py prepare   (caption features of
                       one prompt via the Qwen3 encoder, plus the noise latent)
               data/zimage/Z-Image-Turbo/vae/  (the VAE decoder)
               data/zimage/demo-latent.bin  (written here; fp32 [16, H, W])
               data/zimage/demo.png         (written here; the VAE decode)
  Options      --q4                  block projections as Q4 (~3.7 GB of weights)
               --nodes-per-submit N  split each forward into submissions of N
                                     nodes (mobile GPU watchdogs)
               --split-ffn-up        halve the longest node, for the same reason
               --steps N             sampler steps (default 8)
=============================================================================*/

#include "models/zimage/model.h"
#include "models/zimage/sampler.h"
#include "models/zimage/tongyi-weights.h"
#include "models/zimage/tongyi-vae-weights.h"
#include "vai/engine/safetensors-parser.h"
#include "vai/utils/time-checker.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vai/utils/stb_image_write.h"

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
    uint32_t nodesPerSubmit = 0, steps = 8;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--q4"))                                  q4 = true;
        else if (!strcmp(argv[i], "--split-ffn-up"))                     splitFfnUp = true;
        else if (!strcmp(argv[i], "--nodes-per-submit") && i + 1 < argc) nodesPerSubmit = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)            steps = (uint32_t)atoi(argv[++i]);
        else
        {
            fprintf(stderr, "usage: %s [--q4] [--nodes-per-submit N] [--split-ffn-up] [--steps N]\n", argv[0]);
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

    // Turbo: 8 steps by default; step 0 also uploads the weights and records the graph.
    const std::vector<float> sigmas = zimage::turboSigmas(steps);
    {
        TimeChecker total("\nSampled {} steps in {}ms", uint32_t(sigmas.size() - 1));
        auto t0 = std::chrono::steady_clock::now();
        zimage::sampleTurbo(model, capFeats, latent, H, W, sigmas, [&](uint32_t i, float sigma) {
            const auto t1 = std::chrono::steady_clock::now();
            printf("  step %u (sigma %.3f): %.0fms\n", i, sigma,
                   std::chrono::duration<double, std::milli>(t1 - t0).count());
            t0 = t1;
        });
    }

    const char* latentPath = VAI_DATA_DIR "/zimage/demo-latent.bin";
    std::ofstream(latentPath, std::ios::binary).write(
        reinterpret_cast<const char*>(latent.data()), latent.size() * sizeof(float));
    printf("Latent written to %s\n", latentPath);

    // VAE decode on the GPU: latent -> RGB.
    zimage::VaeDecoder vae({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = nodesPerSubmit,
    });
    {
        TimeChecker timer("VAE weights loaded in {}ms");
        loadWeights<"tongyi">(vae);
    }
    Tensor image;
    {
        TimeChecker timer("VAE decode (first, uploads the weights) in {}ms");
        vae({{"inLatent", zimage::decoderInput(vae.vaeCfg, latent.data(), H, W)}}, {{"outImage", &image}});
        image.downloadToHost();
    }
    const uint32_t imgH = image.shape()[1], imgW = image.shape()[2];
    const std::vector<uint8_t> rgb = zimage::toRgb8(image.getData<float>(), imgH, imgW);
    const char* pngPath = VAI_DATA_DIR "/zimage/demo.png";
    if (!stbi_write_png(pngPath, int(imgW), int(imgH), 3, rgb.data(), int(imgW) * 3))
    {
        fprintf(stderr, "Could not write %s\n", pngPath);
        return 1;
    }
    printf("Image written to %s (%ux%u)\n", pngPath, imgW, imgH);
    return 0;
}
