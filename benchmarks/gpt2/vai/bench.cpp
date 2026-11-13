#include "net.h"
#include "openai-weights.h"
#include "vai/utils/tokenizer.h"
#include "eva-runtime.h"
#include "bench-prompts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::utils;
namespace gpt2 = model::production::gpt2;


// Checkpoint sizes this bench can run. The name is the folder under
// data/ckpts/gpt2/openai/, so --size picks the config and the weights together.
struct SizeOption { const char* name; gpt2::ModelConfig (*config)(); };

static constexpr SizeOption modelSizes[] = {
    { "124M", &gpt2::ModelConfig::_124M },
    { "355M", &gpt2::ModelConfig::_355M },
    { "774M", &gpt2::ModelConfig::_774M },
    { "1.5B", &gpt2::ModelConfig::_1_5B },
};

static void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [N ...] [--size NAME] [--showText]\n", argv0);
    fprintf(stderr, "  N       prompt lengths in tokens, one table row each (default: 146 256 512)\n");
    fprintf(stderr, "  --size  checkpoint folder under data/gpt2/ (default: %s), one of:", modelSizes[0].name);
    for (const SizeOption& s : modelSizes) fprintf(stderr, " %s", s.name);
    fprintf(stderr, "\n");
}


static double nowSec()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// median: robust to the occasional spike a shared cloud box throws in.
// se: standard error of the mean (sample stdev / sqrt(n)) — the uncertainty of
// this row's value, not the sample-to-sample spread; with 15 prefill samples
// the raw stdev read as several times the row's actual reproducibility.
struct Stats { double median, se; };

static Stats stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const double median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);

    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= n;

    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var = n > 1 ? var / (n - 1) : 0.0;

    return { median, std::sqrt(var / n) };
}

// Result table, one row per prompt length: request = the length asked for,
// tokens = what was actually prefilled; decode is always nGen tokens.
static void printHeader(int nGen, int nReps, int nPPReps)
{
    printf("\n decode %d tokens, %d reps · prefill %d reps · median ± SE\n", nGen, nReps, nPPReps);
    printf(" %7s  %-10s  %6s  %21s  %17s\n",
           "request", "framework", "tokens", "prefill (t/s)", "decode (t/s)");
    printf("────────────────────────────────────────────────────────────────\n");
}

static void printRow(const char* framework, int nReq, int nPrompt, Stats pp, Stats tg)
{
    printf(" %7d  %-10s  %6d  %11.2f ± %-7.2f  %9.2f ± %-6.2f\n",
           nReq, framework, nPrompt, pp.median, pp.se, tg.median, tg.se);
    fflush(stdout);
}


int main(int argc, char** argv)
{
    constexpr int      nGen    = 128;   // decode tokens per timed run
    constexpr int      nReps   = 5;     // timed decode runs per length
    constexpr int      nPPReps = 15;    // timed prefill runs per length: a prefill is one
                                        // forward, so it needs samples of its own, where a
                                        // decode run already averages nGen steps internally
    constexpr int      nWarmPre = 3;    // untimed prefills per length
    constexpr int      nWarmGen = 8;    // untimed decode steps per length
    constexpr unsigned seed    = 42;

    // Arguments are prompt lengths in tokens, one table row each: the first N
    // tokens of the essay in bench-prompts.h (146 = the prompt of
    // models/production/gpt2/demo.cpp). No argument runs the default set.
    // --showText: decode and print one generated sequence per run (off by default,
    // so the bench output stays just the table).
    // --size: which checkpoint to load; see modelSizes.
    bool        showText = false;
    const char* sizeName = modelSizes[0].name;

    std::vector<int> lengths;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--showText")) { showText = true; continue; }
        if (!strcmp(argv[i], "--size"))
        {
            if (++i == argc) { fprintf(stderr, "error: --size needs a name\n"); usage(argv[0]); return 1; }
            sizeName = argv[i];
            continue;
        }
        const int n = atoi(argv[i]);
        if (n <= 0)
        {
            usage(argv[0]);
            return 1;
        }
        lengths.push_back(n);
    }
    if (lengths.empty()) lengths = {146, 256, 512};

    const SizeOption* size = nullptr;
    for (const SizeOption& s : modelSizes)
        if (!strcmp(s.name, sizeName)) { size = &s; break; }
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

    auto modelCfg = size->config();
    printf("model: gpt2 %s\n", modelCfg.variant);

    gpt2::Net net(
        {
            .device = device,
            .wDefaultDtype = Dtype::Fp16,
        },
        modelCfg);

    loadWeights<"openai">(net);

    Tokenizer tokenizer(VAI_DATA_DIR "/gpt2/tokenizer.json");

    // Tokenize the whole essay once; each length takes a prefix of it.
    const std::vector<int> essayTokens = tokenizer.encode(prompts::kEssay);
    for (int n : lengths)
    {
        if (n > (int)essayTokens.size())
        {
            fprintf(stderr, "error: the prompt has %zu tokens, %d requested\n", essayTokens.size(), n);
            return 1;
        }
        if (n + nGen > (int)net.runtimeCfg.maxRuntimeContext)
        {
            fprintf(stderr,
                    "error: prompt %d + generation exceeds GPT-2's %u-token context\n",
                    n,
                    net.runtimeCfg.maxRuntimeContext);
            return 1;
        }
    }

    TensorDict inDict;
    Tensor outToken;
    std::vector<int> sanity;        // --showText: rep 0 of the last length, printed at the end
    int sanityLen = 0;

    // One-time warm-up at the longest length, before anything is timed: shader
    // compilation, buffer allocation and the GPU clock ramp are paid once here
    // instead of landing on whichever length happens to run first — which is what
    // left the first row of the table with several times the run-to-run spread of
    // the rest.
    {
        const int nMax = *std::max_element(lengths.begin(), lengths.end());
        const std::vector<int> maxPrompt(essayTokens.begin(), essayTokens.begin() + nMax);

        net.resetCache();
        inDict["inTokenIds"] = Tensor({(uint32_t)nMax}).setData(maxPrompt.data());
        net.prefill(inDict["inTokenIds"], &outToken);

        for (int i = 0; i < nWarmGen; ++i)
            net.decode(&outToken);
    }

    printHeader(nGen, nReps, nPPReps);

    for (const int nPrompt : lengths)
    {
        const std::vector<int> promptTokens(essayTokens.begin(), essayTokens.begin() + nPrompt);

        // Feeds the whole prompt as one forward and returns the sampled next token.
        auto prefill = [&]() -> int {
            net.resetCache();
            inDict["inTokenIds"] = Tensor({(uint32_t)nPrompt}).setData(promptTokens.data());
            net.prefill(inDict["inTokenIds"], &outToken);
            return (int)outToken.getData<uint32_t>()[0];
        };

        // Warm-up for this length's shapes. The prefill path runs more than once:
        // a single pass leaves the first timed sample carrying the cost of meeting
        // the new shape.
        {
            for (int i = 0; i < nWarmPre; ++i) prefill();

            for (int i = 0; i < nWarmGen; ++i)
                net.decode(&outToken);
        }

        std::vector<double> ppTps, tgTps;
        std::vector<int> generated;

        // Prefill, timed on its own: one forward per sample, so the count is set by
        // how many samples the mean needs, not by what a decode run costs.
        for (int rep = 0; rep < nPPReps; ++rep)
        {
            double t0 = nowSec();
            prefill();
            ppTps.push_back(nPrompt / (nowSec() - t0));
        }

        // Decode, timed on its own: the prefill that seeds each run is not counted.
        for (int rep = 0; rep < nReps; ++rep)
        {
            int tok = prefill();

            bool keep = showText && rep == 0;   // keep one run's output for --showText
            if (keep) { generated = promptTokens; generated.push_back(tok); }

            double t1 = nowSec();
            for (int gen = 0; gen < nGen; ++gen)
            {
                net.decode(&outToken);
                tok = (int)outToken.getData<uint32_t>()[0];
                if (keep) generated.push_back(tok);
            }
            tgTps.push_back(nGen / (nowSec() - t1));
        }

        printRow("vAi", nPrompt, nPrompt, stats(ppTps), stats(tgTps));
        sanity = generated; sanityLen = nPrompt;
    }

    if (showText)
    {
        printf("\n=== Generated Text (N=%d, rep 0) ===\n", sanityLen);
        printf("%s\n", tokenizer.decode(sanity).c_str());
    }

    return 0;
}
