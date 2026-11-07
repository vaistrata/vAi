/*=============================================================================
  Written by   ippx123@gmail.com
  Requirements data/zimage/Z-Image-Turbo/vae/ (Tongyi-MAI/Z-Image-Turbo, see
               tongyi-vae-weights.h) and data/zimage/ref-vae-<H>x<W>.safetensors
               from benchmarks/zimage/torch/vae.py — the fp32 diffusers
               decode this bench checks against, of the latent it times.
               --stages compares the stage.* activations vae.py --stages
               stored; --profile prints the GPU time per node kind;
               --nodes-per-submit N splits each decode into submissions of N
               nodes, for Adreno's submission watchdog.
=============================================================================*/

#include "models/zimage/tongyi-vae-weights.h"
#include "vai/engine/safetensors-parser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;


static double nowSec()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::vector<float> readF32(const SafetensorsParser& parser, const char* key)
{
    VAI_ASSERT(parser.getDataType(key) == "F32");
    std::vector<float> v(parser.byteSize(key) / sizeof(float));
    parser.readInto(key, v.data());
    return v;
}

// Error of `got` against `ref`: relative L2, max abs, cosine.
static void report(const char* label, const float* got, const std::vector<float>& ref)
{
    double d2 = 0, r2 = 0, g2 = 0, dot = 0, maxAbs = 0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double d = double(got[i]) - ref[i];
        d2 += d * d; r2 += double(ref[i]) * ref[i]; g2 += double(got[i]) * got[i];
        dot += double(got[i]) * ref[i];
        maxAbs = std::max(maxAbs, std::abs(d));
    }
    printf("%-14s rel L2 %.3e  max abs %.3e  cosine %.6f\n",
           label, std::sqrt(d2 / r2), maxAbs, dot / std::sqrt(g2 * r2));
}

// PSNR of the 8-bit images both decodes round to.
static double psnr8(const float* got, const float* ref, uint32_t H, uint32_t W)
{
    const std::vector<uint8_t> a = zimage::toRgb8(got, H, W), b = zimage::toRgb8(ref, H, W);
    double se = 0;
    for (size_t i = 0; i < a.size(); ++i)
        se += double(int(a[i]) - int(b[i])) * double(int(a[i]) - int(b[i]));
    return se == 0 ? INFINITY : 10.0 * std::log10(255.0 * 255.0 * a.size() / se);
}


int main(int argc, char** argv)
{
    uint32_t height = 512, width = 512, iters = 3, nodesPerSubmit = 0;
    bool stages = false, profile = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--height" && i + 1 < argc)      { height = (uint32_t)atoi(argv[++i]); }
        else if (a == "--width" && i + 1 < argc)  { width = (uint32_t)atoi(argv[++i]); }
        else if (a == "--size" && i + 1 < argc)   { height = width = (uint32_t)atoi(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc)  { iters = (uint32_t)atoi(argv[++i]); }
        else if (a == "--stages")                 { stages = true; }
        else if (a == "--profile")                { profile = true; }
        else if (a == "--nodes-per-submit" && i + 1 < argc) { nodesPerSubmit = (uint32_t)atoi(argv[++i]); }
        else
        {
            fprintf(stderr, "usage: %s [--height H] [--width W] [--iters N] [--stages] [--profile] [--nodes-per-submit N]\n", argv[0]);
            return 1;
        }
    }

    const std::string refPath = std::string(VAI_DATA_DIR "/zimage/ref-vae-")
                              + std::to_string(height) + "x" + std::to_string(width) + ".safetensors";
    SafetensorsParser ref(refPath.c_str());
    const auto latentShape = ref.getShape("latent");
    const uint32_t h = latentShape[1], w = latentShape[2];
    const std::vector<float> latent = readF32(ref, "latent");
    const std::vector<float> expect = readF32(ref, "image");

    zimage::VaeDecoder vae({
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = nodesPerSubmit,
    });

    // --stages: the activations vae.py stored, tapped at the same places.
    std::vector<std::string> taps;
    if (stages)
    {
        const auto names = ref.getTensorNames();
        auto tap = [&](auto& node, const std::string& name) {
            if (std::find(names.begin(), names.end(), "stage." + name) == names.end()) return;
            node - vae.output("stage." + name);
            taps.push_back(name);
        };
        const uint32_t perBlock = vae.vaeCfg.layersPerBlock + 1;
        tap(vae.convIn, "conv_in");
        tap(vae.mid1, "mid");
        for (uint32_t b = 0; b < vae.upConv.size(); ++b) tap(vae.upConv[b], "up." + std::to_string(b));
        tap(vae.resnets.back(), "up." + std::to_string(vae.resnets.size() / perBlock - 1));
    }

    double t0 = nowSec();
    loadWeights<"tongyi">(vae);
    printf("weights read in %.2f s\n", nowSec() - t0);

    Tensor out;
    auto decode = [&](vai::TensorRefDict outs) {
        outs["outImage"] = &out;
        vae({{"inLatent", zimage::decoderInput(vae.vaeCfg, latent.data(), h, w)}}, outs);
        return out.getData<float>();
    };

    t0 = nowSec();
    std::vector<Tensor> got(taps.size());
    vai::TensorRefDict outs;
    for (size_t i = 0; i < taps.size(); ++i)
        outs["stage." + taps[i]] = &got[i];
    const float* image = decode(outs);
    printf("first decode (upload + record) %.2f s\n", nowSec() - t0);
    for (size_t i = 0; i < taps.size(); ++i)
        report(taps[i].c_str(), got[i].getData<float>(), readF32(ref, ("stage." + taps[i]).c_str()));
    printf("%ux%u from a %ux%u latent\n", width, height, w, h);
    report("vai vs fp32", image, expect);
    printf("8-bit PSNR vs fp32 %.1f dB\n", psnr8(image, expect.data(), height, width));

    for (int i = 0; i < 1; ++i) decode({});
    std::vector<double> times;
    for (uint32_t i = 0; i < iters; ++i)
    {
        t0 = nowSec();
        decode({});
        times.push_back(nowSec() - t0);
    }
    std::sort(times.begin(), times.end());
    printf("vai decode: median %.1f ms, min %.1f ms (%u iters)\n",
           times[times.size() / 2] * 1e3, times.front() * 1e3, iters);

    if (profile)
    {
        auto tagResnet = [](zimage::VaeResnet& r, const std::string& at) {
            r.norm1.setProfileTag(at + ".groupNorm");
            r.norm2.setProfileTag(at + ".groupNorm");
            r.conv1.setProfileTag(at + ".conv3x3");
            r.conv2.setProfileTag(at + ".conv3x3");
            r.shortcut.setProfileTag(at + ".conv1x1");
        };
        const uint32_t perBlock = vae.vaeCfg.layersPerBlock + 1;
        vae.convIn.setProfileTag("mid.conv3x3");
        tagResnet(vae.mid0, "mid");
        tagResnet(vae.mid1, "mid");
        vae.midAttn.norm.setProfileTag("mid.groupNorm");
        vae.midAttn.qkv.setProfileTag("mid.conv1x1");
        vae.midAttn.attn.setProfileTag("mid.attention");
        vae.midAttn.toOut.setProfileTag("mid.conv1x1");
        for (size_t i = 0; i < vae.resnets.size(); ++i)
            tagResnet(vae.resnets[i], "up" + std::to_string(i / perBlock));
        for (size_t b = 0; b < vae.upConv.size(); ++b)
            vae.upConv[b].setProfileTag("up" + std::to_string(b) + ".upConv3x3");
        vae.normOut.setProfileTag("out.groupNorm");
        vae.convOut.setProfileTag("out.conv3x3");

        decode({});
        vae.resetProfileRecords();
        for (uint32_t i = 0; i < iters; ++i) decode({});
        vae.printProfile();
    }
    return 0;
}
