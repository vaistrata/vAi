#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image-processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>


namespace vai::utils {

namespace {

// Element offsets of one step along each axis.
struct Strides
{
    size_t c;
    size_t y;
    size_t x;
};

Strides strides(bool chw, uint32_t width, uint32_t height, uint32_t channels)
{
    if (chw)
        return {size_t(width) * height, width, 1};
    return {1, size_t(width) * channels, channels};
}

// Writes one output value, normalized and moved to its output channel.
struct Emit
{
    float* base;
    Strides d;
    bool normalize = false;
    std::array<float, 4> mean{};   // by output channel
    std::array<float, 4> std{};
    bool swapRB = false;

    float value(uint32_t c, float v) const { return normalize ? (v - mean[c]) / std[c] : v; }

    void operator()(uint32_t c, uint32_t y, uint32_t x, float v) const
    {
        if (swapRB && c != 1)
            c = 2 - c;
        base[c * d.c + y * d.y + x * d.x] = value(c, v);
    }
};

// Calls f(c, v) on every value of a held image, with its channel.
template <class F>
void forEachValue(std::vector<float>& pixels, bool chw, uint32_t channels, F f)
{
    const size_t plane = pixels.size() / channels;
    if (chw)
        for (uint32_t c = 0; c < channels; ++c)
            for (size_t i = 0; i < plane; ++i)
                f(c, pixels[c * plane + i]);
    else
        for (size_t i = 0; i < plane; ++i)
            for (uint32_t c = 0; c < channels; ++c)
                f(c, pixels[i * channels + c]);
}

inline int64_t asInt(uint8_t v) { return v; }
inline int64_t asInt(float v) { return std::lround(v); }

// Calls f(lo, hi) on disjoint row ranges covering [0, rows), one per hardware thread.
template<class F>
void parallelRows(uint32_t rows, F f)
{
    const uint32_t count = std::min(std::max(std::thread::hardware_concurrency(), 1u), rows);
    const uint32_t chunk = (rows + count - 1) / count;
    std::vector<std::thread> threads;
    for (uint32_t lo = chunk; lo < rows; lo += chunk)
        threads.emplace_back(f, lo, std::min(rows, lo + chunk));
    f(0u, std::min(rows, chunk));
    for (auto& t : threads)
        t.join();
}


/////////////////////////////////////////////////////////////////////////////////////////
// Bilinear - two taps per output pixel, computed once per axis.
/////////////////////////////////////////////////////////////////////////////////////////
struct Tap
{
    uint32_t i0;
    uint32_t i1;
    float frac;
};

std::vector<Tap> bilinearTaps(uint32_t src, uint32_t dst, bool alignCorners)
{
    const float ratio = alignCorners
        ? float(src - 1) / std::max(dst - 1, 1u)
        : float(src) / dst;

    std::vector<Tap> taps(dst);
    for (uint32_t i = 0; i < dst; ++i)
    {
        const float s = alignCorners ? i * ratio : std::max((i + 0.5f) * ratio - 0.5f, 0.0f);
        const uint32_t i0 = std::min(uint32_t(s), src - 1);
        taps[i] = {i0, std::min(i0 + 1, src - 1), s - i0};
    }
    return taps;
}

template<class T>
void resizeBilinear(
    const T* src, Strides s, uint32_t srcW, uint32_t srcH,
    uint32_t dstW, uint32_t dstH, uint32_t channels, bool alignCorners, const Emit& emit)
{
    const auto xTaps = bilinearTaps(srcW, dstW, alignCorners);
    const auto yTaps = bilinearTaps(srcH, dstH, alignCorners);

    parallelRows(dstH, [&](uint32_t lo, uint32_t hi) {
        for (uint32_t c = 0; c < channels; ++c)
            for (uint32_t y = lo; y < hi; ++y)
            {
                const Tap& ty = yTaps[y];
                const T* row0 = src + c * s.c + ty.i0 * s.y;
                const T* row1 = src + c * s.c + ty.i1 * s.y;
                for (uint32_t x = 0; x < dstW; ++x)
                {
                    const Tap& tx = xTaps[x];
                    const float top    = float(row0[tx.i0 * s.x]) * (1.0f - tx.frac) + float(row0[tx.i1 * s.x]) * tx.frac;
                    const float bottom = float(row1[tx.i0 * s.x]) * (1.0f - tx.frac) + float(row1[tx.i1 * s.x]) * tx.frac;
                    emit(c, y, x, top * (1.0f - ty.frac) + bottom * ty.frac);
                }
            }
    });
}


/////////////////////////////////////////////////////////////////////////////////////////
// Antialias - Pillow's separable triangle filter, widened by the downscale factor,
// in fixed-point integer arithmetic.
/////////////////////////////////////////////////////////////////////////////////////////
struct Filter
{
    uint32_t first;
    std::vector<int32_t> coefficients;
};

struct Axis
{
    std::vector<Filter> filters;
    unsigned precision = 0;
};

Axis antialiasAxis(uint32_t src, uint32_t dst, bool int16)
{
    Axis axis;
    axis.filters.resize(dst);
    std::vector<std::vector<double>> weights(dst);

    const double scale = double(src) / dst;
    const double support = std::max(1.0, scale);
    const double invSupport = 1.0 / support;
    double maxWeight = 0.0;
    for (uint32_t i = 0; i < dst; ++i)
    {
        const double center = scale * (i + 0.5);
        const int64_t first = std::max<int64_t>(0, int64_t(center - support + 0.5));
        const int64_t end = std::min<int64_t>(src, int64_t(center + support + 0.5));
        axis.filters[i].first = uint32_t(first);

        double total = 0.0;
        for (int64_t j = first; j < end; ++j)
        {
            const double w = std::max(0.0, 1.0 - std::abs((j - center + 0.5) * invSupport));
            weights[i].push_back(w);
            total += w;
        }
        for (double& w : weights[i])
        {
            w /= total;
            maxWeight = std::max(maxWeight, w);
        }
    }

    if (int16)
        while (axis.precision < 22 &&
               int64_t(0.5 + maxWeight * (int64_t{1} << (axis.precision + 1))) < (1 << 15))
            ++axis.precision;
    else
        axis.precision = 22;

    for (uint32_t i = 0; i < dst; ++i)
        for (double w : weights[i])
            axis.filters[i].coefficients.push_back(int32_t(0.5 + w * (int64_t{1} << axis.precision)));
    return axis;
}

template<class T>
void resizeAntialias(
    const T* src, Strides s, uint32_t srcW, uint32_t srcH,
    uint32_t dstW, uint32_t dstH, uint32_t channels, bool int16, const Emit& emit)
{
    const Axis horizontal = antialiasAxis(srcW, dstW, int16);
    const Axis vertical = antialiasAxis(srcH, dstH, int16);
    const int64_t hHalf = int64_t{1} << (horizontal.precision - 1);
    const int64_t vHalf = int64_t{1} << (vertical.precision - 1);

    const size_t plane = size_t(dstW) * srcH;
    std::vector<int32_t> tmp(channels * plane);     // [channels, srcH, dstW]

    parallelRows(srcH, [&](uint32_t lo, uint32_t hi) {
        for (uint32_t c = 0; c < channels; ++c)
            for (uint32_t y = lo; y < hi; ++y)
                for (uint32_t x = 0; x < dstW; ++x)
                {
                    const Filter& f = horizontal.filters[x];
                    const T* in = src + c * s.c + y * s.y + f.first * s.x;
                    int64_t sum = hHalf;
                    for (size_t k = 0; k < f.coefficients.size(); ++k)
                        sum += int64_t(f.coefficients[k]) * asInt(in[k * s.x]);
                    tmp[c * plane + size_t(y) * dstW + x] =
                        int32_t(std::clamp<int64_t>(sum >> horizontal.precision, 0, 255));
                }
    });

    // Each tap adds a whole tmp row, so x runs contiguously.
    parallelRows(dstH, [&](uint32_t lo, uint32_t hi) {
        std::vector<int64_t> acc(dstW);
        for (uint32_t c = 0; c < channels; ++c)
            for (uint32_t y = lo; y < hi; ++y)
            {
                const Filter& f = vertical.filters[y];
                std::fill(acc.begin(), acc.end(), vHalf);
                for (size_t k = 0; k < f.coefficients.size(); ++k)
                {
                    const int64_t w = f.coefficients[k];
                    const int32_t* row = tmp.data() + c * plane + size_t(f.first + k) * dstW;
                    for (uint32_t x = 0; x < dstW; ++x)
                        acc[x] += w * row[x];
                }
                for (uint32_t x = 0; x < dstW; ++x)
                    emit(c, y, x, float(std::clamp<int64_t>(acc[x] >> vertical.precision, 0, 255)));
            }
    });
}

template<class T>
void resampleImage(
    const T* src, Strides s, uint32_t srcW, uint32_t srcH,
    uint32_t dstW, uint32_t dstH, uint32_t channels, Resize mode, const Emit& emit)
{
    switch (mode)
    {
    case Resize::Bilinear:
    case Resize::BilinearAlignCorners:
        resizeBilinear(src, s, srcW, srcH, dstW, dstH, channels, mode == Resize::BilinearAlignCorners, emit);
        break;
    case Resize::BilinearAntialias:
    case Resize::BilinearAntialiasInt16:
        resizeAntialias(src, s, srcW, srcH, dstW, dstH, channels, mode == Resize::BilinearAntialiasInt16, emit);
        break;
    }
}

template<class T>
void convert(const T* src, Strides s, uint32_t width, uint32_t height, uint32_t channels, const Emit& emit)
{
    for (uint32_t c = 0; c < channels; ++c)
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                emit(c, y, x, float(src[c * s.c + y * s.y + x * s.x]));
}

} // namespace


/////////////////////////////////////////////////////////////////////////////////////////
// ImageProcessor - recording
/////////////////////////////////////////////////////////////////////////////////////////
ImageProcessor::ImageProcessor(const std::string& path, uint32_t channels)
{
    int w, h, c0;
    uint8_t* hwc = stbi_load(path.c_str(), &w, &h, &c0, int(channels));
    if (!hwc)
        throw std::runtime_error("failed to load image " + path + ": " + stbi_failure_reason());

    *this = ImageProcessor(hwc, uint32_t(w), uint32_t(h), channels);
    stbi_image_free(hwc);
}

ImageProcessor::ImageProcessor(const uint8_t* pixels, uint32_t width, uint32_t height, uint32_t channels)
    : source(pixels, pixels + size_t(width) * height * channels),
      bufWidth(width), bufHeight(height),
      width(width), height(height), channels(channels),
      imageGeometry{.sourceWidth = width, .sourceHeight = height}
{
    if (width == 0 || height == 0 || channels == 0 || channels > 4)
        throw std::invalid_argument("image dimensions must be non-zero, with at most 4 channels");
}

void ImageProcessor::recordResize(uint32_t size, Resize mode)
{
    if (size == 0)
        throw std::invalid_argument("resize size must be non-zero");

    const float sx = float(size) / width;
    const float sy = float(size) / height;
    imageGeometry.scaleX *= sx;
    imageGeometry.scaleY *= sy;
    imageGeometry.padX *= sx;
    imageGeometry.padY *= sy;

    steps.push_back({size, size, 0, 0, 0, 0.0f, mode});
    width = size;
    height = size;
}

void ImageProcessor::recordLetterbox(uint32_t size, Align align, float padValue, Resize mode)
{
    if (size == 0)
        throw std::invalid_argument("letterbox size must be non-zero");

    const float scale = std::min(float(size) / width, float(size) / height);
    const uint32_t dstW = std::max(1u, uint32_t(width * scale));
    const uint32_t dstH = std::max(1u, uint32_t(height * scale));
    const uint32_t padX = align == Align::Center ? (size - dstW) / 2 : 0;
    const uint32_t padY = align == Align::Center ? (size - dstH) / 2 : 0;
    imageGeometry.scaleX *= scale;
    imageGeometry.scaleY *= scale;
    imageGeometry.padX = imageGeometry.padX * scale + padX;
    imageGeometry.padY = imageGeometry.padY * scale + padY;

    steps.push_back({dstW, dstH, size, padX, padY, padValue, mode});
    width = size;
    height = size;
}

void ImageProcessor::recordNormalize(float min, float max)
{
    if (!(min < max))
        throw std::invalid_argument("normalize needs min < max");

    normalized = true;
    rangeMin = min;
    rangeMax = max;
}

void ImageProcessor::recordStandardize(const std::array<float, 3>& mean, const std::array<float, 3>& std)
{
    if (channels != 3)
        throw std::logic_error("standardize() needs 3 channels");

    normalized = true;
    std::copy(mean.begin(), mean.end(), statMean.begin());
    std::copy(std.begin(), std.end(), statStd.begin());
}

void ImageProcessor::recordBgr()
{
    if (channels != 3)
        throw std::logic_error("bgr() needs 3 channels");

    outputBgr = true;
}


/////////////////////////////////////////////////////////////////////////////////////////
// ImageProcessor - running
/////////////////////////////////////////////////////////////////////////////////////////

// Only the last pass normalizes and turns the channels into the output order.
void ImageProcessor::resample(const Resample& step, bool last)
{
    const uint32_t outW = step.canvas ? step.canvas : step.width;
    const uint32_t outH = step.canvas ? step.canvas : step.height;
    const Strides d = strides(outputChw, outW, outH, channels);

    std::vector<float> dst(size_t(channels) * outW * outH);
    Emit emit{dst.data() + step.padY * d.y + step.padX * d.x, d};
    if (last)
    {
        emit.normalize = normalized;
        emit.mean = normMean;
        emit.std = normStd;
        emit.swapRB = storedBgr != outputBgr;
    }
    if (step.canvas)
    {
        std::array<float, 4> pad;
        for (uint32_t c = 0; c < channels; ++c)
            pad[c] = emit.value(c, step.padValue);
        forEachValue(dst, outputChw, channels, [&](uint32_t c, float& v) { v = pad[c]; });
    }

    if (isFloat)
        resampleImage(pixels.data(), strides(storedChw, bufWidth, bufHeight, channels), bufWidth, bufHeight,
                      step.width, step.height, channels, step.mode, emit);
    else
        resampleImage(source.data(), strides(false, bufWidth, bufHeight, channels), bufWidth, bufHeight,
                      step.width, step.height, channels, step.mode, emit);

    pixels = std::move(dst);
    source = {};
    isFloat = true;
    storedChw = outputChw;
    bufWidth = outW;
    bufHeight = outH;
    if (last)
    {
        storedBgr = outputBgr;
        storedNormalized = normalized;
        storedMean = normMean;
        storedStd = normStd;
    }
}

// The last pass when nothing resamples: the decoded bytes, or the previous output, to the output.
void ImageProcessor::finish()
{
    Emit emit{nullptr, strides(outputChw, bufWidth, bufHeight, channels)};
    emit.normalize = normalized && !storedNormalized;
    emit.mean = normMean;
    emit.std = normStd;
    emit.swapRB = storedBgr != outputBgr;

    if (isFloat && storedChw == outputChw)
    {
        if (!emit.normalize && !emit.swapRB)
            return;
        const Strides s = emit.d;
        if (emit.swapRB)
            for (uint32_t y = 0; y < bufHeight; ++y)
                for (uint32_t x = 0; x < bufWidth; ++x)
                    std::swap(pixels[y * s.y + x * s.x], pixels[y * s.y + x * s.x + 2 * s.c]);
        if (emit.normalize)
            forEachValue(pixels, storedChw, channels, [&](uint32_t c, float& v) { v = emit.value(c, v); });
    }
    else
    {
        std::vector<float> dst(size_t(channels) * bufWidth * bufHeight);
        emit.base = dst.data();
        if (isFloat)
            convert(pixels.data(), strides(storedChw, bufWidth, bufHeight, channels), bufWidth, bufHeight, channels, emit);
        else
            convert(source.data(), strides(false, bufWidth, bufHeight, channels), bufWidth, bufHeight, channels, emit);
        pixels = std::move(dst);
        source = {};
        isFloat = true;
        storedChw = outputChw;
    }
    storedBgr = outputBgr;
    storedNormalized = normalized;
    storedMean = normMean;
    storedStd = normStd;
}

void ImageProcessor::run()
{
    // normalize() and standardize() fold into one (v - mean) / std, so [-1, 1] is
    // (v - 127.5) / 127.5 and [0, 1] is v / 255, bit for bit what the reference
    // processors compute.
    for (uint32_t c = 0; c < 4; ++c)
    {
        normMean[c] = 255.0f * (statMean[c] - rangeMin) / (rangeMax - rangeMin);
        normStd[c] = 255.0f * statStd[c] / (rangeMax - rangeMin);
    }

    // A run after an earlier one starts from that output; back to 0~255 when it
    // resamples or normalizes differently, so every pass sees pixel values.
    if (storedNormalized &&
        (!steps.empty() || !normalized || storedMean != normMean || storedStd != normStd))
    {
        forEachValue(pixels, storedChw, channels, [&](uint32_t c, float& v) { v = v * storedStd[c] + storedMean[c]; });
        storedNormalized = false;
    }

    for (size_t i = 0; i < steps.size(); ++i)
        resample(steps[i], i + 1 == steps.size());
    if (steps.empty())
        finish();
    steps.clear();
}

ImageProcessor::operator const std::vector<float>&() &
{
    run();
    return pixels;
}

ImageProcessor::operator std::vector<float>() &&
{
    run();
    return std::move(pixels);
}

}   // namespace vai::utils
