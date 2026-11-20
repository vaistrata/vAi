#ifndef BENCHMARKS_COMMON_BENCH_UTILS_H
#define BENCHMARKS_COMMON_BENCH_UTILS_H

#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>


/////////////////////////////////////////////////////////////////////////////////////////
// Common
/////////////////////////////////////////////////////////////////////////////////////////
inline double nowSec()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


// A whole decimal integer, blanks around it allowed: "12" and " 12 " parse, "12x" and "" do not.
// value is left as it was when the text does not parse.
inline bool parseInt(const char* text, int& value)
{
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX)
        return false;
    while (std::isspace((unsigned char)*end))
        ++end;
    if (*end != '\0')
        return false;
    value = (int)parsed;
    return true;
}


// A whole finite decimal number, blanks around it allowed, as parseInt.
inline bool parseNumber(const char* text, double& value)
{
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text, &end);
    if (end == text || errno == ERANGE || !std::isfinite(parsed))
        return false;
    while (std::isspace((unsigned char)*end))
        ++end;
    if (*end != '\0')
        return false;
    value = parsed;
    return true;
}


// One of the GPUs a framework lists, picked from stdin the way eva's selector does for the
// vAi benches: EOF keeps the first, anything else invalid asks again.
inline int pickGpu(const std::vector<std::string>& names)
{
    if (names.empty())
        return 0;

    printf("**************************************************************\n");
    printf("Detected %zu GPUs:\n", names.size());
    for (size_t i = 0; i < names.size(); ++i)
        printf("[GPU %zu] %s\n", i, names[i].c_str());
    fflush(stdout);

    int selected = 0;
    std::string line;
    while (names.size() > 1)
    {
        printf("Select a GPU (0-%zu): ", names.size() - 1);
        fflush(stdout);
        if (!std::getline(std::cin, line))
        {
            selected = 0;
            printf("\n");
            break;
        }
        if (parseInt(line.c_str(), selected) && 0 <= selected && selected < (int)names.size())
            break;
        fprintf(stderr, "Invalid selection! Please select again.\n");
    }
    printf("[GPU %d] is Selected.\n", selected);
    printf("**************************************************************\n\n");
    fflush(stdout);
    return selected;
}


// Mean and sample standard deviation of the per-sample values, as llama-bench reports them.
struct Stats
{
    double mean, sd;
};

inline Stats stats(const std::vector<double>& v)
{
    const size_t n = v.size();

    double mean = 0.0;
    for (double x : v)
        mean += x;
    mean /= n;

    double var = 0.0;
    for (double x : v)
        var += (x - mean) * (x - mean);
    var = n > 1 ? var / (n - 1) : 0.0;

    return { mean, std::sqrt(var) };
}


// A row whose SD exceeds this share of its mean is flagged: a sample ran slow, from a
// clock or power state change or another process on the GPU, so the mean is not the
// GPU's steady rate.
inline constexpr double spreadLimit = 0.10;


// Runs fn nWarm times untimed, then nReps times timed, into latencyMs (milliseconds per call).
// fn returns void, or bool where false stops the measurement and makes this return false.
template <typename Fn>
bool measureLatency(int nWarm, int nReps, Fn&& fn, std::vector<double>& latencyMs)
{
    auto call = [&] {
        if constexpr (std::is_void_v<std::invoke_result_t<Fn&>>)
        {
            fn();
            return true;
        }
        else
            return bool(fn());
    };

    for (int i = 0; i < nWarm; ++i)
        if (!call())
            return false;

    latencyMs.clear();
    latencyMs.reserve(nReps);
    for (int rep = 0; rep < nReps; ++rep)
    {
        const double t0 = nowSec();
        if (!call())
            return false;
        latencyMs.push_back((nowSec() - t0) * 1000.0);
    }
    return true;
}


// A reference output: count little-endian FP32 values. Empty when the file is absent; a
// file of another size ends the bench, since it belongs to another export.
inline std::vector<float> readReference(const std::string& path, size_t count)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return {};
    if (size_t(file.tellg()) != count * sizeof(float))
    {
        fprintf(
            stderr,
            "error: %s holds %lld bytes, expected %zu\n",
            path.c_str(),
            (long long)file.tellg(),
            count * sizeof(float));
        exit(1);
    }
    std::vector<float> values(count);
    file.seekg(0);
    file.read((char*)values.data(), std::streamsize(count * sizeof(float)));
    if (!file)
    {
        fprintf(stderr, "error: cannot read %s\n", path.c_str());
        exit(1);
    }
    return values;
}


// numpy.allclose: |out - ref| <= atol + rtol * |ref| for every value. Prints one line either way.
// With rows of dim values, only the first checkedDims of each row are compared (0: all).
inline bool checkReference(
    const float* out, const std::vector<float>& ref, double atol, double rtol, size_t dim = 0, size_t checkedDims = 0)
{
    double maxAbs = 0.0;
    size_t mismatches = 0;
    size_t checked = 0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        if (dim && checkedDims && i % dim >= checkedDims)
            continue;
        ++checked;
        const double diff = std::fabs(double(out[i]) - double(ref[i]));
        if (!(diff <= atol + rtol * std::fabs(double(ref[i]))))   // NaN fails too
            ++mismatches;
        if (!(diff <= maxAbs))
            maxAbs = diff;
    }
    printf(
        "accuracy: %s, max abs error %.3g, %zu of %zu values outside atol %g + rtol %g\n",
        mismatches ? "FAIL" : "pass",
        maxAbs,
        mismatches,
        checked,
        atol,
        rtol);
    fflush(stdout);
    return mismatches == 0;
}


inline bool allFinite(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i]))
            return false;
    return true;
}


/////////////////////////////////////////////////////////////////////////////////////////
// LLM
/////////////////////////////////////////////////////////////////////////////////////////
// LLM: one row per prompt length, in tokens.
inline void printLLMHeader(int nGen, int nReps, int nPPReps)
{
    printf("\n decode %d tokens, %d reps · prefill %d reps · mean ± SD\n", nGen, nReps, nPPReps);
    printf(" %7s  %-10s  %21s  %18s\n", "prompt", "framework", "prefill (t/s)", "decode (t/s)");
    printf("───────────────────────────────────────────────────────────────\n");
}


inline constexpr double llmSpreadLimit = spreadLimit;


inline void printLLMRow(
    const char* framework, int nPrompt, const std::vector<double>& ppTps, const std::vector<double>& tgTps)
{
    const Stats pp = stats(ppTps);
    const Stats tg = stats(tgTps);

    printf(" %7d  %-10s  %11.2f ± %-7.2f  %9.2f ± %-6.2f\n", nPrompt, framework, pp.mean, pp.sd, tg.mean, tg.sd);

    auto flag = [](const char* phase, const Stats& s) {
        if (s.sd > llmSpreadLimit * s.mean)
            printf(
                "          warning: %s SD is %.0f%% of the mean; a sample ran slow, rerun this row\n",
                phase,
                100.0 * s.sd / s.mean);
    };
    flag("prefill", pp);
    flag("decode", tg);
    fflush(stdout);
}


/////////////////////////////////////////////////////////////////////////////////////////
// VLA
/////////////////////////////////////////////////////////////////////////////////////////
// VLA: one row per prompt, its image and text tokens apart ("512 + 18"), in milliseconds per
// prediction (one action chunk of actionChunk actions, after denoiseSteps steps).
inline void printVLAHeader(int nWarm, int nReps, unsigned denoiseSteps, unsigned actionChunk)
{
    printf(
        "\n %d warmup, %d reps, %u denoise steps, action chunk %u, ms per prediction, mean ± SD\n",
        nWarm,
        nReps,
        denoiseSteps,
        actionChunk);
    printf(" %12s  %-18s  %18s  %9s\n", "image + text", "framework", "latency (ms)", "min (ms)");
    printf("─────────────────────────────────────────────────────────────────\n");
    fflush(stdout);
}


inline void printVLARow(const char* framework, int nImage, int nText, const std::vector<double>& latencyMs)
{
    const Stats s = stats(latencyMs);
    double minMs = latencyMs.empty() ? 0.0 : latencyMs[0];
    for (double x : latencyMs)
        minMs = std::fmin(minMs, x);

    char prompt[32];
    snprintf(prompt, sizeof(prompt), "%d + %d", nImage, nText);
    printf(" %12s  %-18s  %9.2f ± %-6.2f  %9.2f\n", prompt, framework, s.mean, s.sd, minMs);
    if (s.sd > spreadLimit * s.mean)
        printf(
            "          warning: SD is %.0f%% of the mean; a sample ran slow, rerun this row\n", 100.0 * s.sd / s.mean);
    fflush(stdout);
}


// The first five actions, eight dimensions each, of a [horizon, dim] chunk.
inline void printActionChunk(const float* actions, unsigned horizon, unsigned dim)
{
    printf("\n=== Action chunk (%u steps x %u dims) ===\n", horizon, dim);
    for (unsigned s = 0; s < horizon && s < 5; ++s)
    {
        printf("  t%-2u ", s);
        for (unsigned d = 0; d < dim && d < 8; ++d)
            printf("% .4f ", actions[size_t(s) * dim + d]);
        printf("...\n");
    }
    printf("  ...\n");
}


struct VLABenchOptions
{
    int nWarm = 3;
    int nReps = 20;
    double atol = 2e-3;
    double rtol = 1e-3;
    bool showActions = false;
    std::string referencePath;   // checked when the file is there; the bench runs unverified without it
    unsigned denoiseSteps = 0;   // the workload's, shown in the table header
    unsigned checkedDims = 0;    // leading action dims compared with the reference (0: all)
};


// The callables the VLA helpers below take, the same for every framework:
//   predict  one prediction, from host inputs to the host action chunk (void, or bool for success)
//   actions  the chunk the last prediction left, [horizon, dim] FP32

// One untimed prediction checked against options.referencePath. True when it matches or
// when there is no reference to check.
template <typename Predict, typename Actions>
bool checkVLAReference(
    const VLABenchOptions& options, unsigned horizon, unsigned dim, Predict&& predict, Actions&& actions)
{
    const std::vector<float> reference = readReference(options.referencePath, size_t(horizon) * dim);
    if (reference.empty())
    {
        printf("accuracy: unverified, no %s\n", options.referencePath.c_str());
        fflush(stdout);
        return true;
    }

    printf("reference: %s\n", options.referencePath.c_str());
    if (options.checkedDims && options.checkedDims < dim)
        printf("compared: the first %u of %u action dims\n", options.checkedDims, dim);

    std::vector<double> once;
    if (!measureLatency(0, 1, predict, once))
        return false;

    if (!checkReference(actions(), reference, options.atol, options.rtol, dim, options.checkedDims))
    {
        fprintf(stderr, "error: reference mismatch; latency results suppressed\n");
        return false;
    }

    return true;
}


// Warmup, timed reps and one table row for a prompt of nImage image and nText text tokens.
template <typename Predict, typename Actions>
bool runVLARow(
    const char* framework, int nImage, int nText, const VLABenchOptions& options, unsigned horizon, unsigned dim,
    Predict&& predict, Actions&& actions)
{
    std::vector<double> latencyMs;
    if (!measureLatency(options.nWarm, options.nReps, predict, latencyMs))
        return false;
    if (!allFinite(actions(), size_t(horizon) * dim))
    {
        fprintf(stderr, "error: the action chunk contains non-finite values\n");
        return false;
    }
    printVLARow(framework, nImage, nText, latencyMs);
    return true;
}


// A bench of one prompt: the reference check, the table of one row and optionally the
// actions. Returns the process exit code.
template <typename Predict, typename Actions>
int runVLABench(
    const char* framework, int nImage, int nText, const VLABenchOptions& options, unsigned horizon, unsigned dim,
    Predict&& predict, Actions&& actions)
{
    if (!checkVLAReference(options, horizon, dim, predict, actions))
        return 1;

    printVLAHeader(options.nWarm, options.nReps, options.denoiseSteps, horizon);

    if (!runVLARow(framework, nImage, nText, options, horizon, dim, predict, actions))
        return 1;

    if (options.showActions)
        printActionChunk(actions(), horizon, dim);

    return 0;
}

#endif   // BENCHMARKS_COMMON_BENCH_UTILS_H
