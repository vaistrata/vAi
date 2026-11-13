#include "net.h"
#include "google-weights.h"
#include "vai/utils/tokenizer.h"
#include "eva-runtime.h"
#include "bench-common.h"
#include "vai/utils/image-processor.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::utils;
using namespace gemma3_bench;
namespace gemma3 = model::production::gemma3;

struct SizeOption { const char* name; };

static constexpr SizeOption modelSizes[] = {
    { "4B" },
    { "1B" },
};

inline std::vector<int> makePrompt(
    const Tokenizer& tokenizer,
    const gemma3::ModelConfig& modelCfg,
    const std::string& prompt,
    int nText)
{
    using namespace gemma3_bench;

    const std::vector<int> textTokens = tokenizer.encode(prompt);

    if (nText <= 0 || nText > static_cast<int>(textTokens.size()))
        throw std::runtime_error("requested text length exceeds source prompt");

    const std::vector<int> selected(textTokens.begin(), textTokens.begin() + nText);

    if (!modelCfg.vision)
        return selected;

    const int image = tokenizer.pieceToId(imageSoftToken);

    if (tokenizer.pieceToId(startOfImageToken) < 0 || tokenizer.pieceToId(endOfImageToken) < 0 || image < 0 ||
        static_cast<uint32_t>(image) != modelCfg.vision->imageTokenId)
        throw std::runtime_error("tokenizer image tokens do not match the model");

    std::vector<int> tokens = tokenizer.encode(std::string(userPromptPrefix) + startOfImageToken);

    tokens.insert(tokens.end(), modelCfg.vision->tokensPerImage, image);

    const auto afterImage = tokenizer.encode(std::string(endOfImageToken) + textAfterImage);
    tokens.insert(tokens.end(), afterImage.begin(), afterImage.end());

    tokens.insert(tokens.end(), selected.begin(), selected.end());

    const auto suffix = tokenizer.encode(modelPromptSuffix);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());

    return tokens;
}

int main(int argc, char** argv)
{
    constexpr int nGen = gemma3_bench::decodeTokens;
    constexpr int nReps = gemma3_bench::decodeRepetitions;
    constexpr int nPPReps = gemma3_bench::prefillRepetitions;
    constexpr int nWarmPre = gemma3_bench::warmupPrefills;
    constexpr int nWarmGen = gemma3_bench::warmupDecodeTokens;
    constexpr unsigned seed = gemma3_bench::randomSeed;

    gemma3_bench::BenchmarkArgumentParser args;

    if (!args.parse(argc, argv))
    {
        args.printErrorAndUsage();
        return 1;
    }

    const SizeOption* size = nullptr;

    for (const SizeOption& s : modelSizes)
        if (args.size == s.name)
        {
            size = &s;
            break;
        }

    if (!size)
    {
        fprintf(stderr, "error: unknown model size '%s'\n", args.size.c_str());
        return 1;
    }

    Dtype precision;

    if (args.precision == "fp16") precision = Dtype::Fp16;
    else if (args.precision == "fp32") precision = Dtype::Fp32;
    else {
        fprintf(stderr, "error: unsupported precision '%s'\n", args.precision.c_str());
        return 1;
    }

    eva::Device device = getVaiDevice();

    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    srand(seed);

    gemma3::ModelConfig modelCfg = gemma3::ModelConfig::get(size->name);

    const bool hasVision = modelCfg.vision.has_value();

    printf("model: gemma3 %s (%s)\n",
           modelCfg.variant, args.precision.c_str());

    gemma3::Net net(
        {
            .device = device,
            .wDefaultDtype = precision,
        },
        modelCfg,
        { .kvCacheDtype = precision, .maxRuntimeContext = gemma3_bench::runtimeContext });

    loadWeights<"google">(net);

    const std::string tokenizerPath = std::string(VAI_DATA_DIR "/gemma3/") + modelCfg.variant + "/tokenizer.json";

    Tokenizer tokenizer(tokenizerPath.c_str());

    const std::string vqaPrompt = gemma3_bench::vqaPrompt();

    Tensor inImage;

    if (hasVision) {
        const auto& v = modelCfg.vision.value();

        if (v.imageChannels != gemma3_bench::syntheticImageChannels ||
            v.imageSize != gemma3_bench::syntheticImageSize)
            throw std::runtime_error(
                "vision dimensions differ from the shared benchmark image"
            );

        std::vector<float> rgb;

        if(args.imagepath == "")
        {
            const auto random = gemma3_bench::makeRandom(v.imageSize, v.imageSize, v.imageChannels);
            rgb = ImageProcessor(random.data(), v.imageSize, v.imageSize, v.imageChannels)
                .normalize(-1.0f, 1.0f)
                .chw();
        } 
        else
        {
            rgb = ImageProcessor(args.imagepath)
                .resize(v.imageSize, Resize::BilinearAntialiasInt16)
                .normalize(-1.0f, 1.0f)
                .chw();
        }

        inImage = Tensor({v.imageChannels, v.imageSize, v.imageSize}).setFloats(std::move(rgb));
    }

    printHeader(nGen, nReps, nPPReps);

    std::vector<int> sanity;
    int sanityText = 0;
    int sanityTotal = 0;

    for (const int nText : args.promptLengths) {
        const std::vector<int> promptTokens =
            makePrompt(tokenizer, modelCfg, vqaPrompt, nText);

        const int nPrompt = static_cast<int>(promptTokens.size());

        if (nPrompt + nGen > static_cast<int>(net.runtimeCfg.maxRuntimeContext))
        {
            fprintf(stderr,
                    "error: prompt %d + generation exceeds the %u-token context\n",
                    nPrompt,
                    net.runtimeCfg.maxRuntimeContext);
            return 1;
        }

        Tensor inTokens({static_cast<uint32_t>(nPrompt)});
        Tensor outToken;

        auto prefill = [&]() -> int {
            net.resetCache();
            inTokens.setData(promptTokens.data());

            net.prefill(inImage, inTokens, &outToken);

            return static_cast<int>(
                outToken.getData<uint32_t>()[0]
            );
        };

        int tok = 0;

        for (int i = 0; i < nWarmPre; ++i) tok = prefill();

        for (int i = 0; i < nWarmGen; ++i) {
            net.decode(&outToken);
            tok = static_cast<int>(outToken.getData<uint32_t>()[0]);
        }

        std::vector<double> ppTps;
        for (int rep = 0; rep < nPPReps; ++rep)
        {
            const double t0 = nowSec();
            prefill();
            ppTps.push_back(nPrompt / (nowSec() - t0));
        }

        std::vector<double> tgTps;
        std::vector<int> generated;

        for (int rep = 0; rep < nReps; ++rep) {
            tok = prefill();

            const bool keep = args.showText && rep == 0;

            if (keep) {
                generated.clear();
                generated.push_back(tok);
            }

            const double t0 = nowSec();

            for (int gen = 0; gen < nGen; ++gen)
            {
                net.decode(&outToken);
                tok = static_cast<int>(outToken.getData<uint32_t>()[0]);

                if (keep) generated.push_back(tok);
            }

            tgTps.push_back(nGen / (nowSec() - t0));
        }

        printRow("vAi", nText, nPrompt, stats(ppTps), stats(tgTps));

        sanity = std::move(generated);
        sanityText = nText;
        sanityTotal = nPrompt;
    }

    if (args.showText)
        printf("\n=== Generated Text (text=%d, total=%d, rep 0) ===\n%s\n"
            ,sanityText, sanityTotal, tokenizer.decode(sanity).c_str());

    return 0;
}
