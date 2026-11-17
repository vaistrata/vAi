#include "llama.h"
#include "ggml-backend.h"
#include "bench-inputs.h"
#include "bench-utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>


// --size names a folder under data/benchmarks/gemma3/, the same names as the vAi bench.
constexpr const char* modelSizes[] = { "1B", "270M", "4B" };

void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s [N ...] [--size NAME] [--showText]\n", argv0);
    fprintf(stderr, "  N           prompt lengths in tokens, one table row each (default: 128 256 512 1024 2048)\n");
    fprintf(stderr, "  --size      model folder under data/benchmarks/gemma3/ (default: %s), one of:", modelSizes[0]);
    for (const char* s : modelSizes)
        fprintf(stderr, " %s", s);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --showText  print rep 0's generated text for the last length after the table\n");
}


void fail(const char* msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}


// Silences llama.cpp's logging, as llama-bench does; BENCH_VERBOSE=1 brings it back.
void nullLog(ggml_log_level, const char*, void*) {}


// The GPUs llama.cpp offloads to (dedicated and integrated), picked from stdin the way eva's
// selector does for the vAi bench: EOF keeps the first, anything else invalid asks again.
ggml_backend_dev_t pickGpu()
{
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
    {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)
            gpus.push_back(d);
    }
    if (gpus.empty())
        fail("no GPU found");

    printf("**************************************************************\n");
    printf("Detected %zu GPUs:\n", gpus.size());
    for (size_t i = 0; i < gpus.size(); ++i)
        printf("[GPU %zu] %s\n", i, ggml_backend_dev_description(gpus[i]));
    fflush(stdout);

    int selected = 0;
    std::string line;
    while (gpus.size() > 1)
    {
        printf("Select a GPU (0-%zu): ", gpus.size() - 1);
        fflush(stdout);
        if (!std::getline(std::cin, line))
        {
            selected = 0;
            printf("\n");
            break;
        }
        if (parseInt(line.c_str(), selected) && 0 <= selected && selected < (int)gpus.size())
            break;
        fprintf(stderr, "Invalid selection! Please select again.\n");
    }
    printf("[GPU %d] is Selected.\n", selected);
    printf("**************************************************************\n\n");
    fflush(stdout);
    return gpus[selected];
}


int main(int argc, char** argv)
{
    if (!getenv("BENCH_VERBOSE"))
        llama_log_set(nullLog, nullptr);

    constexpr int nGen = 128;     // decode tokens per timed run
    constexpr int nReps = 5;      // timed decode runs per length
    constexpr int nPPReps = 5;    // timed prefill runs per length
    constexpr int nWarmPre = 1;   // untimed prefills per length after the decode warm-up
    constexpr int nWarmGen = 8;   // untimed decode steps per length
    constexpr int nCtx = 4096;    // = the vAi bench's maxRuntimeContext
    constexpr int topK = 40;      // = vAi TopKSampling::Options::k
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
    const std::string modelPath = std::string(VAI_DATA_DIR "/benchmarks/gemma3/") + sizeName + "/model.gguf";

    // No kernel path pinned: llama.cpp picks its own, NV_coopmat2 on NVIDIA where the
    // driver has it. GGML_VK_DISABLE_COOPMAT2=1 turns that off.

    // No-op for a static llama.cpp; loads the backend plugins under GGML_BACKEND_DL.
    ggml_backend_load_all();
    llama_backend_init();

    ggml_backend_dev_t dev = pickGpu();

    // llama.cpp keeps its input layer on the CPU by default and copies the embeddings to
    // the GPU each call. Moved to the GPU, where vAi runs them, as this is faster.
    const llama_model_tensor_buft_override overrides[] = {
        { "token_embd", ggml_backend_dev_buffer_type(dev) },
        { nullptr, nullptr },
    };

    ggml_backend_dev_t devices[2] = { dev, nullptr };

    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devices;
    mparams.n_gpu_layers = 999;
    mparams.tensor_buft_overrides = overrides;

    // Load model
    printf("model: gemma3 %s\n", sizeName);

    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!model)
        fail("failed to load model");

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const char* paragraph = llmParagraph;
    std::vector<llama_token> paragraphTokens(strlen(paragraph) + 16);
    const int nParagraph = llama_tokenize(
        vocab,
        paragraph,
        (int32_t)strlen(paragraph),
        paragraphTokens.data(),
        (int32_t)paragraphTokens.size(),
        /*add_special=*/true,
        /*parse_special=*/false);
    if (nParagraph <= 0)
        fail("tokenization failed");
    paragraphTokens.resize(nParagraph);
    for (int n : lengths)
    {
        if (n > nParagraph)
        {
            fprintf(stderr, "error: the prompt has %d tokens, %d requested\n", nParagraph, n);
            return 1;
        }
        if (n + nGen > nCtx)
        {
            fprintf(stderr, "error: prompt %d + generation exceeds the %d-token context\n", n, nCtx);
            return 1;
        }
    }

    // GPU sampler chain: top-k then softmax draw, T=1.0 like vAi. This backend sampling
    // API is [EXPERIMENTAL] in llama.h and may change between builds, hence the v0.8.0 pin.
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    llama_sampler_seq_config samplerCfg = { /*seq_id=*/0, chain };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_batch = nCtx;      // one llama_process per prefill
    cparams.n_ubatch = nCtx;     // one graph execution per prefill, like vAi's single forward
    cparams.n_outputs_max = 1;   // last position only, as vAi slices the last row before its lm head
    cparams.samplers = &samplerCfg;
    cparams.n_samplers = 1;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx)
        fail("failed to create context");

    // Every call goes through llama_process() on one reused batch, as llama-bench does with
    // common_batch_get_one(): positions continue from the memory, the last token is the only
    // output. llama_decode() would convert a llama_batch into a new batch on every call.
    llama_batch_ext* batch = llama_batch_ext_init(ctx);

    auto process = [&](const llama_token* tokens, int n, const char* failure) -> llama_token {
        const llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; ++i)
        {
            const llama_pos pos = pos0 + i;
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos))
                fail("cannot add a token to the batch");
        }
        if (!llama_batch_ext_set_output_logits(batch, n - 1, true))
            fail("cannot mark the last token as the output");
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch))
            fail(failure);
        llama_synchronize(ctx);
        const llama_token next = llama_get_sampled_token_ith(ctx, -1);
        if (next == LLAMA_TOKEN_NULL)
            fail("backend sampling is inactive (chain fell back to CPU): "
                 "the GPU comparison against vAi would be apples to oranges");
        return next;
    };

    auto decodeOne = [&](llama_token tok) -> llama_token { return process(&tok, 1, "llama_process failed (decode)"); };

    std::vector<llama_token> sanity;   // --showText: rep 0 of the last length, printed at the end
    int sanityLen = 0;

    // Warm-up at the longest length: pipeline creation, buffer allocation and the GPU
    // clock ramp are paid here instead of in the first row.
    {
        const int nMax = *std::max_element(lengths.begin(), lengths.end());
        std::vector<llama_token> maxPrompt(paragraphTokens.begin(), paragraphTokens.begin() + nMax);

        llama_memory_clear(llama_get_memory(ctx), /*data=*/false);
        llama_token tok = process(maxPrompt.data(), nMax, "llama_process failed (warm-up prefill)");

        for (int i = 0; i < nWarmGen; ++i)
            tok = decodeOne(tok);
    }

    printLLMHeader(nGen, nReps, nPPReps);

    for (const int nPrompt : lengths)
    {
        std::vector<llama_token> promptTokens(paragraphTokens.begin(), paragraphTokens.begin() + nPrompt);

        // data=false resets the cache metadata without clearing the KV buffers, as vAi's
        // resetCache() does.
        auto prefill = [&]() -> llama_token {
            llama_memory_clear(llama_get_memory(ctx), /*data=*/false);
            return process(promptTokens.data(), nPrompt, "llama_process failed (prefill)");
        };

        // Warm-up for this length. The last prefill seeds the decode that opens the timed
        // prefills.
        llama_token tok = prefill();   // seeds the cache the decode warm-up reads
        for (int i = 0; i < nWarmGen; ++i)
            tok = decodeOne(tok);

        for (int i = 0; i < nWarmPre; ++i)
            tok = prefill();

        std::vector<double> ppTps, tgTps;
        std::vector<llama_token> generated;

        // Each prefill follows a decode, as in generation, so it rebuilds its graph.
        // decodeOne() waits for the decode, keeping it out of the timed span.
        for (int rep = 0; rep < nPPReps; ++rep)
        {
            decodeOne(tok);

            double t0 = nowSec();
            tok = prefill();
            ppTps.push_back(nPrompt / (nowSec() - t0));
        }

        for (int rep = 0; rep < nReps; ++rep)
        {
            tok = prefill();

            bool keep = showText && rep == 0;
            if (keep)
            {
                generated = promptTokens;
                generated.push_back(tok);
            }

            double t1 = nowSec();
            for (int gen = 0; gen < nGen; ++gen)
            {
                tok = decodeOne(tok);
                if (keep)
                    generated.push_back(tok);
            }
            tgTps.push_back(nGen / (nowSec() - t1));
        }

        printLLMRow("llama.cpp", nPrompt, ppTps, tgTps);
        sanity = generated;
        sanityLen = nPrompt;
    }

    if (showText)
    {
        printf("\n=== Generated Text (N=%d, rep 0) ===\n", sanityLen);
        for (llama_token t : sanity)
        {
            char buf[256];
            int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
            if (n > 0)
                printf("%.*s", n, buf);
        }
        printf("\n");
    }

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_sampler_free(chain);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
