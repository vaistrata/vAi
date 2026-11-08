/*=============================================================================
  Written by   ippx123@gmail.com
  Requirements data/zimage/Z-Image-Turbo/tokenizer/tokenizer.json, the encoder's
               prepared file data/zimage/Z-Image-Turbo-text-encoder-q4.safetensors
               (examples/zimage/tools.py text-encoder, or written here from
               text_encoder/ on a first run), and data/zimage/ref-text-encoder.safetensors
               from benchmarks/zimage/torch/text_encoder.py — the fp32 features
               this bench checks against, of the prompt it stores.
               --layers N runs the first N layers and checks stage.<N-1>;
               --nodes-per-submit N splits the forward for Adreno's watchdog;
               --no-int8-input keeps the projections' input in fp16 / fp32;
               --tokens N times a prompt of N tokens (up to 512) instead, unchecked.
=============================================================================*/

#include "models/zimage/tongyi-text-encoder-weights.h"
#include "vai/engine/safetensors-parser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vai;


static double nowSec()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::vector<float> readF32(const SafetensorsParser& parser, const std::string& key)
{
    const Tensor t(parser[key], Dtype::Fp32);
    const float* p = static_cast<const float*>(t.hostData());
    return std::vector<float>(p, p + t.numElements());
}

static void compare(const float* got, const SafetensorsParser& ref, const std::string& key)
{
    const std::vector<float> expect = readF32(ref, key);
    double d2 = 0, r2 = 0, g2 = 0, dot = 0, maxAbs = 0;
    for (size_t i = 0; i < expect.size(); ++i)
    {
        const double d = double(got[i]) - expect[i];
        d2 += d * d; r2 += double(expect[i]) * expect[i]; g2 += double(got[i]) * got[i];
        dot += double(got[i]) * expect[i];
        maxAbs = std::max(maxAbs, std::abs(d));
    }
    printf("vai vs fp32 (%s): rel L2 %.3e  max abs %.3e  cosine %.6f\n",
           key.c_str(), std::sqrt(d2 / r2), maxAbs, dot / std::sqrt(g2 * r2));
}


int main(int argc, char** argv)
{
    uint32_t layers = 0, iters = 3, nodesPerSubmit = 0, tokens = 0;
    bool int8Input = true;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--layers" && i + 1 < argc)                 layers = (uint32_t)atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)             iters = (uint32_t)atoi(argv[++i]);
        else if (a == "--nodes-per-submit" && i + 1 < argc)  nodesPerSubmit = (uint32_t)atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc)            tokens = (uint32_t)atoi(argv[++i]);
        else if (a == "--no-int8-input")                     int8Input = false;
        else
        {
            fprintf(stderr, "usage: %s [--layers N] [--iters N] [--nodes-per-submit N] [--no-int8-input] [--tokens N]\n", argv[0]);
            return 1;
        }
    }

    SafetensorsParser ref(VAI_DATA_DIR "/zimage/ref-text-encoder.safetensors");

    // Tokens: the template through vAi's tokenizer against transformers'.
    const Tokenizer tok(zimage::tongyi::tokenizerPath(), /*inputIsNfc*/ true);   // an ASCII prompt
    zimage::TextEncoderConfig cfg;
    std::vector<int> refIds;
    for (float id : readF32(ref, "ids"))
        refIds.push_back(int(id));
    std::string prompt;
    for (float byte : readF32(ref, "prompt"))
        prompt.push_back(char(int(byte)));
    std::vector<int> ids = zimage::captionTokens(cfg, tok, prompt);
    printf("prompt \"%s\": %zu tokens, %s transformers'\n", prompt.c_str(), ids.size(),
           ids == refIds ? "the same as" : "DIFFERENT FROM");
    if (ids != refIds)
        return 1;

    if (layers)
        cfg.numLayers = layers;
    zimage::TextEncoder enc({
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = nodesPerSubmit,
    }, cfg);
    if (!int8Input)
        for (auto& layer : enc.layers)
            for (node::Linear* lin : {&layer.mhAttn.toQ, &layer.mhAttn.toK, &layer.mhAttn.toV,
                                      &layer.mhAttn.toOut, &layer.ffn.gateValue, &layer.ffn.out})
                lin->config_.int8Input = false;

    double t0 = nowSec();
    // A prepared file of more layers serves fewer: the keys are per layer.
    const std::string prepared = zimage::tongyi::preparedTextEncoderPath(enc);
    if (!zimage::tongyi::loadTextEncoderPrepared(enc))
        printf("wrote %s\n", prepared.c_str());
    printf("weights read in %.2f s\n", nowSec() - t0);

    if (tokens)
    {
        std::string longer;
        while (zimage::captionTokens(cfg, tok, longer).size() < std::min(tokens, cfg.maxTokens))
            longer += prompt + " ";
        ids = zimage::captionTokens(cfg, tok, longer);
        ids.resize(std::min<size_t>(ids.size(), tokens));
    }

    t0 = nowSec();
    Tensor feats = zimage::encodeCaption(enc, ids);
    printf("first encode (upload + record) %.2f s\n", nowSec() - t0);

    if (tokens)
        printf("%zu tokens: not checked\n", ids.size());
    else
        compare(static_cast<const float*>(feats.hostData()), ref,
                layers ? "stage." + std::to_string(layers - 1) : "cap_feats");

    std::vector<double> times;
    for (uint32_t i = 0; i < iters; ++i)
    {
        t0 = nowSec();
        zimage::encodeCaption(enc, ids);
        times.push_back(nowSec() - t0);
    }
    std::sort(times.begin(), times.end());
    if (!times.empty())
        printf("vai encode (%zu tokens): median %.1f ms, min %.1f ms (%u iters)\n",
               ids.size(), times[times.size() / 2] * 1e3, times.front() * 1e3, iters);
    return 0;
}
