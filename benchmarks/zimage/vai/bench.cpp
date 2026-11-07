/*=============================================================================
  Written by   ippx123@gmail.com
  Requirements data/zimage/Z-Image-Turbo/transformer/ (Tongyi-MAI/Z-Image-Turbo,
               see tongyi-weights.h) and data/zimage/ref-<H>x<W>.safetensors
               from benchmarks/zimage/torch/bench.py — the fp32 diffusers output
               this bench checks against, on the inputs it times.
               --stages compares the stage.* activations that bench.py
               --stages stored, to localize a mismatch; --profile prints the
               GPU time per node kind; --scalar-attention runs the
               non-coopmat attention kernel and --no-int8-attention
               keeps its Q·K in fp32; --q4 stores the block
               projections as Q4; --nodes-per-submit N splits each forward
               into submissions of N nodes and --split-ffn-up halves the
               longest node, for Adreno's submission watchdog.
=============================================================================*/

#include "models/zimage/model.h"
#include "models/zimage/tongyi-weights.h"
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
static void report(const char* label, const std::vector<float>& got, const std::vector<float>& ref)
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


int main(int argc, char** argv)
{
    uint32_t height = 1024, width = 1024, iters = 10, nodesPerSubmit = 0;
    bool stages = false, profile = false, scalarAttention = false, int8Attention = true, q4 = false;
    bool splitFfnUp = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--size" && i + 1 < argc)       { height = width = (uint32_t)atoi(argv[++i]); }
        else if (a == "--height" && i + 1 < argc) { height = (uint32_t)atoi(argv[++i]); }
        else if (a == "--width" && i + 1 < argc)  { width = (uint32_t)atoi(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc)  { iters = (uint32_t)atoi(argv[++i]); }
        else if (a == "--stages")                 { stages = true; }
        else if (a == "--profile")                { profile = true; }
        else if (a == "--scalar-attention")       { scalarAttention = true; }
        else if (a == "--no-int8-attention")      { int8Attention = false; }
        else if (a == "--q4")                     { q4 = true; }
        else if (a == "--split-ffn-up")           { splitFfnUp = true; }
        else if (a == "--nodes-per-submit" && i + 1 < argc) { nodesPerSubmit = (uint32_t)atoi(argv[++i]); }
        else
        {
            fprintf(stderr, "usage: %s [--height H] [--width W] [--iters N] [--stages] [--profile] [--scalar-attention] [--no-int8-attention] [--q4] [--nodes-per-submit N] [--split-ffn-up]\n", argv[0]);
            return 1;
        }
    }

    std::string refPath = std::string(VAI_DATA_DIR "/zimage/ref-")
                        + std::to_string(height) + "x" + std::to_string(width) + ".safetensors";
    SafetensorsParser ref(refPath.c_str());
    std::vector<float> latent = readF32(ref, "latent");
    std::vector<float> expect = readF32(ref, "out");
    std::vector<float> bf16   = readF32(ref, "out_bf16");
    const float t = readF32(ref, "t")[0];
    Tensor capFeats = Tensor(ref["cap_feats"], Dtype::Fp32);
    const uint32_t H = height / 8, W = width / 8;

    auto modelCfg = zimage::ModelConfig::Turbo();
    const uint32_t Nx = zimage::padTo((H / modelCfg.patchSize) * (W / modelCfg.patchSize), modelCfg.seqMultiple);
    const uint32_t Nc = zimage::padTo(capFeats.shape()[0], modelCfg.seqMultiple);

    eva::Device device = getVaiDevice();
    zimage::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = nodesPerSubmit,
    }, modelCfg, {
        .coopmatAttention = !scalarAttention,
        .int8Attention = int8Attention,
        .blockWeightDtype = q4 ? Dtype::Q4 : Dtype::None,
        .splitFfnUp = splitFfnUp,
    });

    // Stage taps: every stage.<name> in the reference gets an extra output
    // boundary on the matching node, before the graph is first built.
    std::vector<std::string> taps;
    if (stages)
    {
        const auto names = ref.getTensorNames();
        auto tap = [&](auto& node, const std::string& name) {
            if (std::find(names.begin(), names.end(), "stage." + name) == names.end()) return;
            node - model.output("stage." + name);
            taps.push_back(name);
        };
        tap(model.temb, "temb");
        tap(model.xEmbed, "x_embed");
        for (size_t i = 0; i < model.noiseRefiner.size(); ++i)   tap(model.noiseRefiner[i], "noise_refiner." + std::to_string(i));
        for (size_t i = 0; i < model.contextRefiner.size(); ++i) tap(model.contextRefiner[i], "context_refiner." + std::to_string(i));
        for (size_t i = 0; i < model.layers.size(); ++i)         tap(model.layers[i], "layers." + std::to_string(i));
    }

    double t0 = nowSec();
    loadWeights<"tongyi">(model);
    printf("weights read in %.1f s\n", nowSec() - t0);

    Tensor out;
    auto forward = [&]() {
        model(zimage::makeInputs(modelCfg, latent.data(), H, W, capFeats, t), {{"outPatches", &out}});
        return zimage::unpatchify(modelCfg, out.getData<float>(), H, W);
    };

    // Correctness: first forward (also records the graph and uploads weights)
    if (stages)
    {
        std::vector<vai::Tensor> got(taps.size());
        vai::TensorRefDict outs = {{"outPatches", &out}};
        for (size_t i = 0; i < taps.size(); ++i)
            outs["stage." + taps[i]] = &got[i];
        model(zimage::makeInputs(modelCfg, latent.data(), H, W, capFeats, t), outs);
        for (size_t i = 0; i < taps.size(); ++i)
        {
            const std::string& name = taps[i];
            std::vector<float> want = readF32(ref, ("stage." + name).c_str());
            const float* p = got[i].getData<float>();
            // Row counts can differ where one side pads and the other has not
            // yet (x_embed: torch embeds the padded rows, vAi pads after the
            // embedder) — compare the shared prefix.
            const size_t n = std::min(want.size(), size_t(got[i].numElements()));
            report(name.c_str(), std::vector<float>(p, p + n), std::vector<float>(want.begin(), want.begin() + n));
        }
    }
    t0 = nowSec();
    std::vector<float> got = forward();
    printf("first forward (upload + record) %.1f s\n", nowSec() - t0);
    printf("%ux%u, %u image + %u caption tokens (padded), t = %.3f, %s attention\n", width, height, Nx, Nc, t,
           model.layers[0].attn.attn.usingCoopmat() ? "coopmat" :
           model.layers[0].attn.attn.usingInt8() ? "int8 Q·K" : "scalar");
    report("vai vs fp32", got, expect);
    report("bf16 vs fp32", bf16, expect);

    // Timing: one DiT forward = one denoising step, host input build to readback.
    for (int i = 0; i < 2; ++i) forward();
    std::vector<double> times;
    for (uint32_t i = 0; i < iters; ++i)
    {
        t0 = nowSec();
        forward();
        times.push_back(nowSec() - t0);
    }
    std::sort(times.begin(), times.end());
    printf("vai %s: median %.1f ms, min %.1f ms (%u iters)\n", q4 ? "q4" : "fp16",
           times[times.size() / 2] * 1e3, times.front() * 1e3, iters);

    // GPU time per node kind (timestamps re-record the graph, so this runs
    // after the timing above).
    if (profile)
    {
        auto tagBlock = [](zimage::TransformerBlock& b, const std::string& seq) {
            b.adaLN.setProfileTag(seq + ".linear.adaLN");
            b.norm1.setProfileTag(seq + ".adaNorm");
            b.norm2.setProfileTag(seq + ".adaNorm");
            b.ffnNorm1.setProfileTag(seq + ".adaNorm");
            b.ffnNorm2.setProfileTag(seq + ".adaNorm");
            b.attn.toQ.setProfileTag(seq + ".linear.qkv");
            b.attn.toK.setProfileTag(seq + ".linear.qkv");
            b.attn.toV.setProfileTag(seq + ".linear.qkv");
            b.attn.normRopeQ.setProfileTag(seq + ".qkNormRope");
            b.attn.normRopeK.setProfileTag(seq + ".qkNormRope");
            b.attn.attn.setProfileTag(seq + ".attention");
            b.attn.toOut.setProfileTag(seq + ".linear.out");
            b.ffn.gateValue.setProfileTag(seq + ".linear.ffnUp");
            b.ffn.value.setProfileTag(seq + ".linear.ffnUp");
            b.ffn.join.setProfileTag(seq + ".gatedMul");
            b.ffn.gatedMul.setProfileTag(seq + ".gatedMul");
            b.ffn.out.setProfileTag(seq + ".linear.ffnDown");
        };
        for (auto& b : model.noiseRefiner)   tagBlock(b, "refiner");
        for (auto& b : model.contextRefiner) tagBlock(b, "refiner");
        for (auto& b : model.layers)         tagBlock(b, "layers");
        model.finalLinear.setProfileTag("final");

        for (int i = 0; i < 2; ++i) forward();
        model.resetProfileRecords();
        for (uint32_t i = 0; i < iters; ++i) forward();
        model.printProfile();
    }
    return 0;
}
