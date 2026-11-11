#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef YOLOX_WEIGHTS_DIR
#define YOLOX_WEIGHTS_DIR ""
#endif

enum class Variant {
    SMALL,
    MEDIUM,
    LARGE,
    XLARGE,
};

inline Variant stringToVariant(const std::string& value)
{
    if (value == "s" || value == "small") {
        return Variant::SMALL;
    }
    if (value == "m" || value == "medium") {
        return Variant::MEDIUM;
    }
    if (value == "l" || value == "large") {
        return Variant::LARGE;
    }
    if (value == "x" || value == "xlarge") {
        return Variant::XLARGE;
    }
    throw std::invalid_argument("Unsupported YOLOX variant: " + value);
}

inline std::string variantToString(Variant variant)
{
    switch (variant) {
        case Variant::SMALL:
            return "s";
        case Variant::MEDIUM:
            return "m";
        case Variant::LARGE:
            return "l";
        case Variant::XLARGE:
            return "x";
        default:
            throw std::invalid_argument("Unsupported YOLOX variant.");
    }
}

struct BenchmarkConfig
{
    static constexpr int kDefaultWarmup = 1000;
    static constexpr int kDefaultIterations = 100;

    int warmup;
    int iterations;
    std::string runtime_name;

    BenchmarkConfig(int warmup_count, int iteration_count, std::string runtime)
        : warmup(warmup_count)
        , iterations(iteration_count)
        , runtime_name(std::move(runtime))
    {}
};

struct BenchmarkResult
{
    double latency_avg_ms = 0.0;
    double latency_min_ms = 0.0;
    double latency_max_ms = 0.0;
    double latency_stddev = 0.0;
    double throughput_ips = 0.0;
    size_t processed_count = 0;
    std::string runtime_name;

    void print() const
    {
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "[" << runtime_name << "] Result" << std::endl;
        std::cout << " - Processed Items : " << processed_count << std::endl;
        std::cout << " - Avg Latency     : " << latency_avg_ms << " ms" << std::endl;
        std::cout << " - Min Latency     : " << latency_min_ms << " ms" << std::endl;
        std::cout << " - Max Latency     : " << latency_max_ms << " ms" << std::endl;
        std::cout << " - Std Latency     : " << latency_stddev << " ms" << std::endl;
        std::cout << " - Throughput      : " << throughput_ips << " items/sec (FPS or TPS)" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
};

class Timer
{
public:
    using Clock = std::chrono::steady_clock;

    void start() { t0_ = Clock::now(); }

    double stop_ms() const
    {
        const auto t1 = Clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    Clock::time_point t0_;
};

namespace yolox_benchmark {

inline constexpr int kDefaultWarmup = BenchmarkConfig::kDefaultWarmup;
inline constexpr int kDefaultIterations = BenchmarkConfig::kDefaultIterations;

enum class Precision
{
    FP32,
    FP16,
};

inline std::optional<Precision> parsePrecision(const std::string& value)
{
    if (value == "fp32" || value == "float32" || value == "f32") {
        return Precision::FP32;
    }
    if (value == "fp16" || value == "float16" || value == "half" || value == "f16") {
        return Precision::FP16;
    }
    return std::nullopt;
}

inline std::string toCliName(Precision precision)
{
    return precision == Precision::FP16 ? "fp16" : "fp32";
}

inline std::string precisionDisplayName(Precision precision)
{
    return precision == Precision::FP16 ? "FP16" : "FP32";
}

inline std::string modelArtifactPath(
    Variant variant,
    const std::string& filename)
{
    return std::string(YOLOX_WEIGHTS_DIR) + "/yolox_" + variantToString(variant)
        + filename;
}

inline size_t precisionElementSize(Precision precision)
{
    return precision == Precision::FP16 ? sizeof(uint16_t) : sizeof(float);
}

inline std::string defaultImagePath()
{
    return VAI_DATA_DIR "/images/dog.jpg";
}

struct CommonCliOptions
{
    Variant variant = Variant::LARGE;
    int warmup = kDefaultWarmup;
    int iterations = kDefaultIterations;
    Precision precision = Precision::FP32;
    std::string imagePath = defaultImagePath();
};

inline std::string takeOptionValue(int& index, int argc, char** argv, const std::string& option)
{
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + option + ".");
    }
    return argv[++index];
}

inline int parseIntegerOption(const std::string& value, const std::string& option)
{
    int parsed = 0;
    const char* const begin = value.data();
    const char* const end = begin + value.size();
    const auto [position, error] = std::from_chars(begin, end, parsed);
    if (error != std::errc() || position != end) {
        throw std::invalid_argument("Invalid integer for " + option + ": " + value + ".");
    }
    return parsed;
}

inline bool parseCommonOption(
    const std::string& argument,
    int& index,
    int argc,
    char** argv,
    CommonCliOptions& options)
{
    if (argument == "--size") {
        options.variant = ::stringToVariant(takeOptionValue(index, argc, argv, argument));
        return true;
    }
    if (argument == "--warmup") {
        options.warmup = parseIntegerOption(
            takeOptionValue(index, argc, argv, argument), argument);
        return true;
    }
    if (argument == "--iterations") {
        options.iterations = parseIntegerOption(
            takeOptionValue(index, argc, argv, argument), argument);
        return true;
    }
    if (argument == "--precision") {
        const auto parsedPrecision = parsePrecision(takeOptionValue(index, argc, argv, argument));
        if (!parsedPrecision) {
            throw std::invalid_argument("Unsupported --precision value. Expected fp32 or fp16.");
        }
        options.precision = *parsedPrecision;
        return true;
    }
    if (argument == "--image") {
        options.imagePath = takeOptionValue(index, argc, argv, argument);
        return true;
    }
    return false;
}

inline void validateCommonCliOptions(const CommonCliOptions& options)
{
    if (options.warmup < 0 || options.iterations <= 0) {
        throw std::invalid_argument(
            "--warmup must be non-negative and --iterations must be positive.");
    }
}

inline BenchmarkResult makeBenchmarkResult(
    const std::string& runtimeName,
    const std::vector<double>& elapsedTimesMs)
{
    if (elapsedTimesMs.empty()) {
        throw std::invalid_argument("Benchmark requires at least one measured latency sample.");
    }

    BenchmarkResult result;
    result.runtime_name = runtimeName;
    result.processed_count = elapsedTimesMs.size();
    const double sum = std::accumulate(elapsedTimesMs.begin(), elapsedTimesMs.end(), 0.0);
    result.latency_avg_ms = sum / static_cast<double>(elapsedTimesMs.size());
    result.throughput_ips = result.latency_avg_ms > 0.0 ? 1000.0 / result.latency_avg_ms : 0.0;

    const auto [minIt, maxIt] = std::minmax_element(elapsedTimesMs.begin(), elapsedTimesMs.end());
    result.latency_min_ms = *minIt;
    result.latency_max_ms = *maxIt;

    double variance = 0.0;
    for (double elapsedMs : elapsedTimesMs) {
        const double diff = elapsedMs - result.latency_avg_ms;
        variance += diff * diff;
    }
    variance /= static_cast<double>(elapsedTimesMs.size());
    result.latency_stddev = std::sqrt(variance);
    return result;
}

} // namespace yolox_benchmark
