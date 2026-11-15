#include "net.h"
#include "google-weights.h"
#include "vai/utils/tokenizer.h"
#include "eva-runtime.h"
#include "bench-inputs.h"
#include "bench-utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;
namespace gemma3 = model::production::gemma3;


// --size names a folder under data/ckpts/gemma3/google/ and picks its config with it.
// 4B is a vision-language model: its vision weights load, but a prefill without images never
// runs them, so the bench measures its language model as it does the text-only sizes.
constexpr const char* modelSizes[] = { "1B", "270M", "4B" };


void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [N ...] [--size NAME] [--showText]\n", argv0);
    fprintf(stderr, "  N           prompt lengths in tokens, one table row each (default: 128 256 512 1024 2048)\n");
    fprintf(
        stderr,
        "  --size      checkpoint folder under data/ckpts/gemma3/google/ (default: %s), one of:",
        modelSizes[0]);
    for (const char* s : modelSizes)
        fprintf(stderr, " %s", s);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --showText  print rep 0's generated text for the last length after the table\n");
}


int main(int argc, char** argv)
{
    constexpr int nGen = 128;     // decode tokens per timed run
    constexpr int nReps = 5;      // timed decode runs per length
    constexpr int nPPReps = 5;    // timed prefill runs per length
    constexpr int nWarmPre = 1;   // untimed prefills per length after the decode warm-up
    constexpr int nWarmGen = 8;   // untimed decode steps per length
    constexpr int nCtx = 4096;    // = the llama.cpp bench's n_ctx
    constexpr uint32_t seed = 42;

    // N takes the first N tokens of llmParagraph in bench-inputs.h.
    bool showText = false;
    const char* sizeName = modelSizes[0];

    std::vector<int> lengths;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--showText"))
        {
            showText = true;
            continue;
        }
        if (!strcmp(argv[i], "--size"))
        {
            if (++i == argc)
            {
                fprintf(stderr, "error: --size needs a name\n");
                usage(argv[0]);
                return 1;
            }
            sizeName = argv[i];
            continue;
        }
        int n = 0;
        if (!parseInt(argv[i], n) || n <= 0)
        {
            usage(argv[0]);
            return 1;
        }
        lengths.push_back(n);
    }
    if (lengths.empty())
        lengths = { 128, 256, 512, 1024, 2048 };

    const char* size = nullptr;
    for (const char* s : modelSizes)
        if (!strcmp(s, sizeName))
        {
            size = s;
            break;
        }
    if (!size)
    {
        fprintf(stderr, "error: unknown --size '%s'\n", sizeName);
        usage(argv[0]);
        return 1;
    }

    eva::Device device = getVaiDevice();
    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    // TopKSampling draws via std::rand() each forward
    srand(seed);

    // Load model
    auto modelCfg = gemma3::ModelConfig::get(size);
    printf("model: gemma3 %s\n", modelCfg.variant);

    // clang-format off
    gemma3::Net net({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg, {
        .maxRuntimeContext = nCtx,
        .prefillChunk = nCtx,   // the whole prompt in one forward, as llama.cpp's n_ubatch = n_ctx
    });
    // clang-format on

    // Load weights
    loadWeights<"google">(net);

    // Load tokenizer
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/gemma3/tokenizer.json");

    // true: Gemma 3 prompts open with <bos>
    const std::vector<int> paragraphTokens = tokenizer.encode(llmParagraph, true);
    for (int n : lengths)
    {
        if (n > (int)paragraphTokens.size())
        {
            fprintf(stderr, "error: the prompt has %zu tokens, %d requested\n", paragraphTokens.size(), n);
            return 1;
        }
        if (n + nGen > (int)net.runtimeCfg.maxRuntimeContext)
        {
            fprintf(
                stderr,
                "error: prompt %d + generation exceeds the %u-token context\n",
                n,
                net.runtimeCfg.maxRuntimeContext);
            return 1;
        }
    }

    Tensor outToken;
    std::vector<int> sanity;   // --showText: rep 0 of the last length, printed at the end
    int sanityLen = 0;

    // Warm-up at the longest length: pipeline creation, buffer allocation and the GPU
    // clock ramp are paid here instead of in the first row.
    {
        const int nMax = *std::max_element(lengths.begin(), lengths.end());
        const std::vector<int> maxPrompt(paragraphTokens.begin(), paragraphTokens.begin() + nMax);

        net.resetCache();
        net.prefill({}, Tensor({ (uint32_t)nMax }).setData(maxPrompt.data()), &outToken);   // {}: no images

        for (int i = 0; i < nWarmGen; ++i)
            net.decode(&outToken);
    }

    printLLMHeader(nGen, nReps, nPPReps);

    for (const int nPrompt : lengths)
    {
        const std::vector<int> promptTokens(paragraphTokens.begin(), paragraphTokens.begin() + nPrompt);

        // Reused across prefills, as the demos do: a new Tensor would land on a different
        // device buffer.
        Tensor inTokenIds({ (uint32_t)nPrompt });

        // setData marks the tokens for upload again, so each prefill still copies its
        // prompt to the device, as llama.cpp's does.
        auto prefill = [&]() -> int {
            net.resetCache();
            inTokenIds.setData(promptTokens.data());
            net.prefill({}, inTokenIds, &outToken);
            return (int)outToken.getData<uint32_t>()[0];
        };

        // Warm-up for this length. The last prefill seeds the decode that opens the timed
        // prefills.
        prefill();   // seeds the cache the decode warm-up reads
        for (int i = 0; i < nWarmGen; ++i)
            net.decode(&outToken);

        for (int i = 0; i < nWarmPre; ++i)
            prefill();

        std::vector<double> ppTps, tgTps;
        std::vector<int> generated;

        // Each prefill follows a decode, as in generation, so it re-records its commands.
        // The readback keeps the decode out of the timed span.
        for (int rep = 0; rep < nPPReps; ++rep)
        {
            net.decode(&outToken);
            outToken.getData<uint32_t>();

            double t0 = nowSec();
            prefill();
            ppTps.push_back(nPrompt / (nowSec() - t0));
        }

        for (int rep = 0; rep < nReps; ++rep)
        {
            int tok = prefill();

            bool keep = showText && rep == 0;
            if (keep)
            {
                generated = promptTokens;
                generated.push_back(tok);
            }

            double t1 = nowSec();
            for (int gen = 0; gen < nGen; ++gen)
            {
                net.decode(&outToken);
                tok = (int)outToken.getData<uint32_t>()[0];
                if (keep)
                    generated.push_back(tok);
            }
            tgTps.push_back(nGen / (nowSec() - t1));
        }

        printLLMRow("vAi", nPrompt, ppTps, tgTps);
        sanity = generated;
        sanityLen = nPrompt;
    }

    if (showText)
    {
        printf("\n=== Generated Text (N=%d, rep 0) ===\n", sanityLen);
        printf("%s\n", tokenizer.decode(sanity).c_str());
    }

    return 0;
}
