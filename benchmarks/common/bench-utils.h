#ifndef BENCHMARKS_COMMON_BENCH_UTILS_H
#define BENCHMARKS_COMMON_BENCH_UTILS_H

#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


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


// LLM: one row per prompt length, in tokens.
inline void printLLMHeader(int nGen, int nReps, int nPPReps)
{
    printf("\n decode %d tokens, %d reps · prefill %d reps · mean ± SD\n", nGen, nReps, nPPReps);
    printf(" %7s  %-10s  %21s  %18s\n", "prompt", "framework", "prefill (t/s)", "decode (t/s)");
    printf("───────────────────────────────────────────────────────────────\n");
}


// A row whose SD exceeds this share of its mean is flagged: a sample ran slow, from a
// clock or power state change or another process on the GPU, so the mean is not the
// GPU's steady rate.
inline constexpr double llmSpreadLimit = 0.10;


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

#endif   // BENCHMARKS_COMMON_BENCH_UTILS_H
