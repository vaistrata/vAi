#include "llama.h"
#include "ggml-backend.h"
#include "bench-prompts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


// Checkpoint sizes this bench can run — the folder under data/llama3/, kept in
// step with the vai bench's --size so both read the same checkpoint.
static constexpr const char* modelSizes[] = { "3.2-1B", "3.2-3B" };

static void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [N ...] [--size NAME] [--showText] | --list-devices\n", argv0);
    fprintf(stderr, "  N       prompt lengths in tokens, one table row each (default: 146 512 1024 2048)\n");
    fprintf(stderr, "  --size  checkpoint folder under data/llama3/ (default: %s), one of:", modelSizes[0]);
    for (const char* s : modelSizes) fprintf(stderr, " %s", s);
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


static void fail(const char* msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}


// llama.cpp reports every loader and context detail at INFO, and its default
// callback prints regardless of level. Silence it the way llama-bench does;
// BENCH_VERBOSE=1 in the environment brings it back for debugging.
static void nullLog(ggml_log_level, const char*, void*) {}


int main(int argc, char** argv)
{
    if (!getenv("BENCH_VERBOSE"))
        llama_log_set(nullLog, nullptr);

    constexpr int      nGen    = 128;           // decode tokens per timed run
    constexpr int      nReps   = 5;             // timed decode runs per length
    constexpr int      nPPReps = 15;            // timed prefill runs per length
    constexpr int      nWarmPre = 3;            // untimed prefills per length
    constexpr int      nWarmGen = 8;            // untimed decode steps per length
    constexpr int      nCtx  = 4096;            // = the vai bench's maxRuntimeContext
    constexpr int      topK  = 40;              // = vai TopKSampling::Options::k
    constexpr uint32_t seed  = 42;

    // Enumerate and exit, so a caller can show the list before picking an index
    if (argc > 1 && !strcmp(argv[1], "--list-devices"))
    {
        ggml_backend_load_all();
        llama_backend_init();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            printf("[DEV %zu] %s — %s\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        }
        llama_backend_free();
        return 0;
    }

    // Arguments are prompt lengths in tokens, one table row each: the first N
    // tokens of the essay in bench-prompts.h (146 = the prompt of
    // models/production/llama3/demo.cpp). No argument runs the default set.
    // --showText: decode and print one generated sequence per run (off by default,
    // so the bench output stays just the table).
    bool        showText = false;
    const char* sizeName = modelSizes[0];

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
    if (lengths.empty()) lengths = {146, 512, 1024, 2048};

    bool known = false;
    for (const char* s : modelSizes) known = known || !strcmp(s, sizeName);
    if (!known)
    {
        fprintf(stderr, "error: unknown --size '%s'\n", sizeName);
        usage(argv[0]);
        return 1;
    }
    const std::string modelPath = std::string(VAI_DATA_DIR "/llama3/") + sizeName + "/model-f16.gguf";

    // No kernel path pinned: llama.cpp picks its own, NV_coopmat2 on NVIDIA
    // where the driver has it. vai is GL_KHR_cooperative_matrix only, so to
    // compare the same kernels there, run with GGML_VK_DISABLE_COOPMAT2=1.

    // No-op when the installed llama.cpp links its backends statically (the
    // default); loads the backend plugins when built with GGML_BACKEND_DL.
    ggml_backend_load_all();
    llama_backend_init();

    // Device list + interactive pick from stdin, mirroring eva's selector so
    // both benches can be driven the same way (e.g. `echo 1 |`).
    // Like eva, EOF or a non-number keeps the default index 0.
    const int nDev = (int)ggml_backend_dev_count();
    printf("**************************************************************\n");
    printf("Detected %d devices:\n", nDev);
    for (int i = 0; i < nDev; ++i)
    {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("[DEV %d] %s — %s\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
    }
    fflush(stdout);

    int selected = 0;
    if (nDev > 1)
    {
        while (true)
        {
            printf("Select a device (0-%d): ", nDev - 1);
            fflush(stdout);
            if (scanf("%d", &selected) != 1)
                break;
            if (selected >= 0 && selected < nDev)
                break;
            fprintf(stderr, "Invalid index %d! Please select again.\n", selected);
        }
    }
    if (selected < 0 || selected >= nDev)
        selected = 0;

    ggml_backend_dev_t dev = ggml_backend_dev_get(selected);
    printf("[DEV %d] is Selected.\n", selected);
    printf("**************************************************************\n\n");

    // Model, pinned to the one requested device
    ggml_backend_dev_t devices[2] = { dev, nullptr };

    llama_model_params mparams = llama_model_default_params();
    mparams.devices      = devices;
    mparams.n_gpu_layers = 999;

    printf("model: llama3 %s\n", sizeName);

    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!model) fail("failed to load model");

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize the whole essay once; each length takes a prefix of it.
    const char* kEssay = prompts::kEssay;
    std::vector<llama_token> essayTokens(strlen(kEssay) + 16);
    const int nEssay = llama_tokenize(vocab, kEssay, (int32_t)strlen(kEssay),
                                      essayTokens.data(), (int32_t)essayTokens.size(),
                                      /*add_special=*/true, /*parse_special=*/false);
    if (nEssay <= 0) fail("tokenization failed");
    essayTokens.resize(nEssay);
    for (int n : lengths)
    {
        if (n > nEssay)
        {
            fprintf(stderr, "error: the prompt has %d tokens, %d requested\n", nEssay, n);
            return 1;
        }
        if (n + nGen > nCtx) fail("prompt + generation exceeds the context");
    }

    // Backend (GPU) sampler chain: top-k then softmax draw, T=1.0 like vai.
    // The backend sampling API this rides on — llama_sampler_seq_config,
    // llama_context_params::samplers, llama_get_sampled_token_ith — is marked
    // [EXPERIMENTAL] in llama.h: another build may change its signature (build
    // break) or its meaning (silently different samples). Hence the b10426 pin.
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    llama_sampler_seq_config samplerCfg = { /*seq_id=*/0, chain };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = nCtx;
    cparams.n_batch    = nCtx;   // one llama_decode per prefill; default 2048 would assert above that
    cparams.n_ubatch   = nCtx;   // ... and one graph execution for it: the default physical batch (512)
                                 // splits the prompt into chunks, re-reading every weight per chunk.
    cparams.n_outputs_max = 1;   // only the last position's logits are read, and vai slices the last row
                                 // before its lm head; the default (= n_batch) reserves them for every
                                 // prompt position instead.
    cparams.samplers   = &samplerCfg;
    cparams.n_samplers = 1;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) fail("failed to create context");

    std::vector<llama_token> sanity;   // --showText: rep 0 of the last length, printed at the end
    int sanityLen = 0;

    // One-time warm-up at the longest length, before anything is timed: kernel
    // selection, buffer allocation and the GPU clock ramp are paid once here
    // instead of landing on whichever length happens to run first. Also proves the
    // sampler chain actually runs on the backend.
    {
        const int nMax = *std::max_element(lengths.begin(), lengths.end());
        std::vector<llama_token> maxPrompt(essayTokens.begin(), essayTokens.begin() + nMax);

        llama_memory_clear(llama_get_memory(ctx), /*data=*/false);
        llama_batch batch = llama_batch_get_one(maxPrompt.data(), nMax);
        if (llama_decode(ctx, batch)) fail("llama_decode failed (warm-up prefill)");
        llama_synchronize(ctx);

        llama_token tok = llama_get_sampled_token_ith(ctx, -1);
        if (tok == LLAMA_TOKEN_NULL)
            fail("backend sampling is inactive (chain fell back to CPU) — "
                 "the GPU comparison against vai would be apples to oranges");

        for (int i = 0; i < nWarmGen; ++i)
        {
            llama_batch one = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, one)) fail("llama_decode failed (warm-up decode)");
            llama_synchronize(ctx);
            tok = llama_get_sampled_token_ith(ctx, -1);
        }
    }

    printHeader(nGen, nReps, nPPReps);

    for (const int nPrompt : lengths)
    {
        std::vector<llama_token> promptTokens(essayTokens.begin(), essayTokens.begin() + nPrompt);

        // Feeds the whole prompt as one batch and returns the sampled next token.
        // data=false resets the cache metadata without memsetting the KV buffers,
        // matching what vai's Net::resetCache() does on its side of the bench.
        auto prefill = [&]() -> llama_token {
            llama_memory_clear(llama_get_memory(ctx), /*data=*/false);
            llama_batch batch = llama_batch_get_one(promptTokens.data(), nPrompt);
            if (llama_decode(ctx, batch)) fail("llama_decode failed (prefill)");
            llama_synchronize(ctx);
            return llama_get_sampled_token_ith(ctx, -1);
        };
        auto decodeOne = [&](llama_token tok) -> llama_token {
            llama_batch batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch)) fail("llama_decode failed (decode)");
            llama_synchronize(ctx);
            return llama_get_sampled_token_ith(ctx, -1);
        };

        // Warm-up for this length's shapes. The prefill path runs more than once:
        // a single pass leaves the first timed sample carrying the cost of meeting
        // the new shape.
        {
            llama_token tok = LLAMA_TOKEN_NULL;
            for (int i = 0; i < nWarmPre; ++i) tok = prefill();
            for (int i = 0; i < nWarmGen; ++i) tok = decodeOne(tok);
        }

        std::vector<double> ppTps, tgTps;
        std::vector<llama_token> generated;

        // Prefill, timed on its own: one llama_decode per sample, so the count is
        // set by how many samples the mean needs, not by what a decode run costs.
        for (int rep = 0; rep < nPPReps; ++rep)
        {
            double t0 = nowSec();
            prefill();
            ppTps.push_back(nPrompt / (nowSec() - t0));
        }

        // Decode, timed on its own: the prefill that seeds each run is not counted.
        for (int rep = 0; rep < nReps; ++rep)
        {
            llama_token tok = prefill();

            bool keep = showText && rep == 0;   // keep one run's output for --showText
            if (keep) { generated = promptTokens; generated.push_back(tok); }

            double t1 = nowSec();
            for (int gen = 0; gen < nGen; ++gen)
            {
                tok = decodeOne(tok);
                if (keep) generated.push_back(tok);
            }
            tgTps.push_back(nGen / (nowSec() - t1));
        }

        printRow("llama.cpp", nPrompt, nPrompt, stats(ppTps), stats(tgTps));
        sanity = generated; sanityLen = nPrompt;
    }

    if (showText)
    {
        printf("\n=== Generated Text (N=%d, rep 0) ===\n", sanityLen);
        for (llama_token t : sanity)
        {
            char buf[256];
            int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
            if (n > 0) printf("%.*s", n, buf);
        }
        printf("\n");
    }

    llama_free(ctx);
    llama_sampler_free(chain);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
