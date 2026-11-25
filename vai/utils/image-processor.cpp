/*=============================================================================
  Image preprocessing - how the pieces fit

  Recording and running
    Every public step only records; converting to std::vector<float> runs them.
    A resize or a letterbox is one pass over the image. A crop is no pass of its
    own: it narrows the window the pass before it writes, or, ahead of any pass,
    the window the first pass reads, so a resize then a crop computes only the
    pixels it keeps. normalize(), standardize(), bgr() and chw() describe the
    output and ride on the last pass's writes through Emit; with no pass,
    finish() is the one conversion.

  Images in memory
    Source is a view of the held image, the decoded bytes or the last pass's
    floats, HWC or CHW, through Strides. A pass reads either layout, and any
    window of it, in place. Target is the size a pass resizes to and the window
    of that it writes.

  Resampling - two engines, after the two families of reference resamplers
    interpolate<N>   torch F.interpolate and cv2 on float images: N taps per
                     axis at fixed offsets, indices clamped at the border, float
                     arithmetic. Nearest (1), Bilinear (2) and Bicubic (4) differ
                     only in how their taps are built.
    antialias        Pillow, and torchvision on uint8: a Kernel widened by the
                     downscale factor, fixed-point coefficients, the result
                     rounded to 0~255 after each axis. Bilinear, Bicubic and
                     Lanczos differ only in the Kernel.
    resampleImage() gives each Resize mode its engine and taps or Kernel, one
    case per mode. A mode matching a further reference tends to be one tap
    builder or one Kernel, plus its case there.

  Speed
    Taps and filters are built once per axis, rows are split across threads by
    parallelRows(), and both engines touch only the source rows and output
    pixels the written window needs.
=============================================================================*/

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image-processor.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
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

// A window of a held image, read in place.
template<class T>
struct Source
{
    const T* data;   // the window's top-left value
    Strides s;
    uint32_t width;
    uint32_t height;
    uint32_t channels;

    const T* row(uint32_t c, uint32_t y) const { return data + c * s.c + y * s.y; }
};

template<class T>
Source<T> sourceOf(const T* data, bool chw, uint32_t bufWidth, uint32_t bufHeight, uint32_t channels, Window view)
{
    const Strides s = strides(chw, bufWidth, bufHeight, channels);
    return {data + view.y * s.y + view.x * s.x, s, view.width, view.height, channels};
}

// The size a pass resizes to, and the window of it the pass writes.
struct Target
{
    uint32_t width;
    uint32_t height;
    Window keep;
};

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
// Interpolate - N taps per axis at fixed offsets, float arithmetic, as torch's
// F.interpolate and cv2 on float images. Each output pixel is the N x N taps,
// each source row first reduced along x, then the rows along y.
/////////////////////////////////////////////////////////////////////////////////////////
// One output pixel's taps along an axis.
template<uint32_t N>
struct Taps
{
    uint32_t index[N];   // source pixels, clamped to the image
    float weight[N];
};

// One axis: the taps of every output pixel along it.
template<uint32_t N>
using TapAxis = std::vector<Taps<N>>;

// floor(i * scale), torch's nearest and cv2 INTER_NEAREST.
TapAxis<1> nearestTaps(uint32_t src, uint32_t dst)
{
    const float scale = float(src) / dst;

    TapAxis<1> axis(dst);
    for (uint32_t i = 0; i < dst; ++i)
        axis[i] = {{std::min(uint32_t(std::floor(i * scale)), src - 1)}, {1.0f}};
    return axis;
}

// Half-pixel centres clamped at 0, or corners aligned.
TapAxis<2> linearTaps(uint32_t src, uint32_t dst, bool alignCorners)
{
    const float ratio = alignCorners
        ? float(src - 1) / std::max(dst - 1, 1u)
        : float(src) / dst;

    TapAxis<2> axis(dst);
    for (uint32_t i = 0; i < dst; ++i)
    {
        const float s = alignCorners ? i * ratio : std::max((i + 0.5f) * ratio - 0.5f, 0.0f);
        const uint32_t i0 = std::min(uint32_t(s), src - 1);
        const float frac = s - i0;
        axis[i] = {{i0, std::min(i0 + 1, src - 1)}, {1.0f - frac, frac}};
    }
    return axis;
}

// Keys a = -0.75, torch's two branches, in float as torch evaluates them.
float cubicInner(float x)   // |x| < 1
{
    constexpr float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

float cubicOuter(float x)   // 1 <= |x| < 2
{
    constexpr float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

// Half-pixel centres, not clamped: the four taps straddle the centre and clamp
// one by one, as torch's do.
TapAxis<4> cubicTaps(uint32_t src, uint32_t dst)
{
    const float scale = float(src) / dst;

    TapAxis<4> axis(dst);
    for (uint32_t i = 0; i < dst; ++i)
    {
        const float s = scale * (i + 0.5f) - 0.5f;
        const float base = std::floor(s);
        const float t = s - base;
        const float u = 1.0f - t;
        for (int64_t k = 0; k < 4; ++k)
            axis[i].index[k] = uint32_t(std::clamp<int64_t>(int64_t(base) - 1 + k, 0, src - 1));
        axis[i].weight[0] = cubicOuter(t + 1.0f);
        axis[i].weight[1] = cubicInner(t);
        axis[i].weight[2] = cubicInner(u);
        axis[i].weight[3] = cubicOuter(u + 1.0f);
    }
    return axis;
}

template<uint32_t N, class T>
void interpolate(
    const Source<T>& src, const Target& dst, const TapAxis<N>& xAxis, const TapAxis<N>& yAxis, const Emit& emit)
{
    const Window& keep = dst.keep;

    parallelRows(keep.height, [&](uint32_t lo, uint32_t hi) {
        for (uint32_t c = 0; c < src.channels; ++c)
            for (uint32_t y = lo; y < hi; ++y)
            {
                const Taps<N>& ty = yAxis[keep.y + y];
                const T* rows[N];
                for (uint32_t j = 0; j < N; ++j)
                    rows[j] = src.row(c, ty.index[j]);

                for (uint32_t x = 0; x < keep.width; ++x)
                {
                    const Taps<N>& tx = xAxis[keep.x + x];
                    float v = 0.0f;
                    for (uint32_t j = 0; j < N; ++j)
                    {
                        float alongX = 0.0f;
                        for (uint32_t i = 0; i < N; ++i)
                            alongX += float(rows[j][tx.index[i] * src.s.x]) * tx.weight[i];
                        v += alongX * ty.weight[j];
                    }
                    emit(c, y, x, v);
                }
            }
    });
}


/////////////////////////////////////////////////////////////////////////////////////////
// Antialias - Pillow's separable filters, the kernel widened by the downscale factor,
// in fixed-point integer arithmetic; torchvision's uint8 antialias is the same with
// coefficients capped to fit int16. A Kernel is Pillow's filter description: its
// half-width and its weight at distance x.
/////////////////////////////////////////////////////////////////////////////////////////
double triangleWeight(double x)
{
    return std::max(0.0, 1.0 - std::abs(x));
}

// Keys with a = -0.5, Pillow's and torchvision's.
double bicubicWeight(double x)
{
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

double sinc(double x)
{
    if (x == 0.0)
        return 1.0;
    x *= std::numbers::pi;
    return std::sin(x) / x;
}

double lanczosWeight(double x)
{
    return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0;
}

struct Kernel
{
    double support;   // half-width before the downscale widens it
    double (*weight)(double x);
};

constexpr Kernel triangle = {1.0, triangleWeight};
constexpr Kernel bicubic = {2.0, bicubicWeight};
constexpr Kernel lanczos = {3.0, lanczosWeight};

// One axis: the filter of every output pixel along it. Output i reads count[i]
// source pixels from first[i], with the fixed-point coefficients at(i).
struct FilterAxis
{
    uint32_t stride = 0;                 // coefficients per output, the widest filter's
    std::vector<uint32_t> first;
    std::vector<uint32_t> count;
    std::vector<int32_t> coefficients;   // [dst, stride]
    unsigned precision = 0;

    const int32_t* at(uint32_t i) const { return coefficients.data() + size_t(i) * stride; }
};

FilterAxis filterAxis(uint32_t src, uint32_t dst, const Kernel& kernel, bool int16)
{
    const double scale = double(src) / dst;
    const double widen = std::max(1.0, scale);
    const double support = kernel.support * widen;
    const double invWiden = 1.0 / widen;

    FilterAxis axis;
    axis.stride = uint32_t(std::ceil(support)) * 2 + 1;
    axis.first.resize(dst);
    axis.count.resize(dst);
    axis.coefficients.resize(size_t(dst) * axis.stride);
    std::vector<double> weights(size_t(dst) * axis.stride);

    double maxWeight = 0.0;
    for (uint32_t i = 0; i < dst; ++i)
    {
        const double center = scale * (i + 0.5);
        const int64_t first = std::max<int64_t>(0, int64_t(center - support + 0.5));
        const int64_t end = std::min<int64_t>(src, int64_t(center + support + 0.5));
        axis.first[i] = uint32_t(first);
        axis.count[i] = uint32_t(end - first);

        double* w = weights.data() + size_t(i) * axis.stride;
        double total = 0.0;
        for (uint32_t k = 0; k < axis.count[i]; ++k)
        {
            w[k] = kernel.weight((first + k - center + 0.5) * invWiden);
            total += w[k];
        }
        for (uint32_t k = 0; k < axis.count[i]; ++k)
        {
            if (total != 0.0)
                w[k] /= total;
            maxWeight = std::max(maxWeight, w[k]);
        }
    }

    if (int16)
        while (axis.precision < 22 &&
               int64_t(0.5 + maxWeight * (int64_t{1} << (axis.precision + 1))) < (1 << 15))
            ++axis.precision;
    else
        axis.precision = 22;

    // Rounded half away from zero: a negative lobe rounds as Pillow and torchvision do.
    const double one = double(int64_t{1} << axis.precision);
    for (size_t k = 0; k < weights.size(); ++k)
        axis.coefficients[k] = int32_t(weights[k] < 0.0 ? -0.5 + weights[k] * one : 0.5 + weights[k] * one);
    return axis;
}

// Horizontal pass into an int32 [channels, rows, keep.width] buffer, then vertical,
// each clipped to 0~255. Only the source rows the kept output rows reach go through
// the horizontal pass.
template<class T>
void antialias(
    const Source<T>& src, const Target& dst, const FilterAxis& xAxis, const FilterAxis& yAxis, const Emit& emit)
{
    const Window& keep = dst.keep;
    const int64_t xHalf = int64_t{1} << (xAxis.precision - 1);
    const int64_t yHalf = int64_t{1} << (yAxis.precision - 1);

    const uint32_t lastY = keep.y + keep.height - 1;
    const uint32_t rowLo = yAxis.first[keep.y];
    const uint32_t rowHi = yAxis.first[lastY] + yAxis.count[lastY];
    const size_t plane = size_t(keep.width) * (rowHi - rowLo);
    const auto tmp = std::make_unique_for_overwrite<int32_t[]>(src.channels * plane);

    parallelRows(rowHi - rowLo, [&](uint32_t lo, uint32_t hi) {
        for (uint32_t c = 0; c < src.channels; ++c)
            for (uint32_t r = lo; r < hi; ++r)
            {
                const T* in = src.row(c, rowLo + r);
                int32_t* out = tmp.get() + c * plane + size_t(r) * keep.width;
                for (uint32_t x = 0; x < keep.width; ++x)
                {
                    const uint32_t i = keep.x + x;
                    const int32_t* k = xAxis.at(i);
                    const T* p = in + xAxis.first[i] * src.s.x;
                    int64_t sum = xHalf;
                    for (uint32_t t = 0; t < xAxis.count[i]; ++t)
                        sum += int64_t(k[t]) * asInt(p[t * src.s.x]);
                    out[x] = int32_t(std::clamp<int64_t>(sum >> xAxis.precision, 0, 255));
                }
            }
    });

    // Each tap adds a whole tmp row, so x runs contiguously.
    parallelRows(keep.height, [&](uint32_t lo, uint32_t hi) {
        std::vector<int64_t> acc(keep.width);
        for (uint32_t c = 0; c < src.channels; ++c)
            for (uint32_t y = lo; y < hi; ++y)
            {
                const uint32_t i = keep.y + y;
                const int32_t* k = yAxis.at(i);
                std::fill(acc.begin(), acc.end(), yHalf);
                for (uint32_t t = 0; t < yAxis.count[i]; ++t)
                {
                    const int64_t w = k[t];
                    const int32_t* row = tmp.get() + c * plane + size_t(yAxis.first[i] + t - rowLo) * keep.width;
                    for (uint32_t x = 0; x < keep.width; ++x)
                        acc[x] += w * row[x];
                }
                for (uint32_t x = 0; x < keep.width; ++x)
                    emit(c, y, x, float(std::clamp<int64_t>(acc[x] >> yAxis.precision, 0, 255)));
            }
    });
}


/////////////////////////////////////////////////////////////////////////////////////////
// Dispatch - each Resize mode to its engine, one case per mode. Both engines take
// one axis per direction, built here from the same description.
/////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void resampleImage(const Source<T>& src, const Target& dst, Resize mode, const Emit& emit)
{
    auto interpolateWith = [&](auto taps, auto... args) {
        interpolate(src, dst, taps(src.width, dst.width, args...), taps(src.height, dst.height, args...), emit);
    };
    auto antialiasWith = [&](const Kernel& kernel, bool int16) {
        antialias(src, dst, filterAxis(src.width, dst.width, kernel, int16),
                  filterAxis(src.height, dst.height, kernel, int16), emit);
    };

    switch (mode)
    {
    case Resize::Nearest:
        interpolateWith(nearestTaps);
        break;
    case Resize::Bilinear:
        interpolateWith(linearTaps, false);
        break;
    case Resize::BilinearAlignCorners:
        interpolateWith(linearTaps, true);
        break;
    case Resize::Bicubic:
        interpolateWith(cubicTaps);
        break;
    case Resize::BilinearAntialias:
        antialiasWith(triangle, false);
        break;
    case Resize::BilinearAntialiasInt16:
        antialiasWith(triangle, true);
        break;
    case Resize::BicubicAntialias:
        antialiasWith(bicubic, false);
        break;
    case Resize::BicubicAntialiasInt16:
        antialiasWith(bicubic, true);
        break;
    case Resize::LanczosAntialias:
        antialiasWith(lanczos, false);
        break;
    }
}

// The held window to the output as it is, when nothing resamples.
template<class T>
void convert(const Source<T>& src, const Emit& emit)
{
    for (uint32_t c = 0; c < src.channels; ++c)
        for (uint32_t y = 0; y < src.height; ++y)
        {
            const T* in = src.row(c, y);
            for (uint32_t x = 0; x < src.width; ++x)
                emit(c, y, x, float(in[x * src.s.x]));
        }
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
      bufWidth(width), bufHeight(height), view{0, 0, width, height},
      width(width), height(height), channels(channels),
      imageGeometry{.sourceWidth = width, .sourceHeight = height}
{
    if (width == 0 || height == 0 || channels == 0 || channels > 4)
        throw std::invalid_argument("image dimensions must be non-zero, with at most 4 channels");
}

void ImageProcessor::recordResize(uint32_t newWidth, uint32_t newHeight, Resize mode)
{
    if (newWidth == 0 || newHeight == 0)
        throw std::invalid_argument("resize size must be non-zero");

    const float sx = float(newWidth) / width;
    const float sy = float(newHeight) / height;
    imageGeometry.scaleX *= sx;
    imageGeometry.scaleY *= sy;
    imageGeometry.padX *= sx;
    imageGeometry.padY *= sy;

    steps.push_back({newWidth, newHeight, {0, 0, newWidth, newHeight}, 0, 0, 0, 0.0f, mode});
    width = newWidth;
    height = newHeight;
}

void ImageProcessor::recordResizeShorter(uint32_t size, Resize mode)
{
    if (size == 0)
        throw std::invalid_argument("resizeShorter size must be non-zero");

    // int(size * long / short), the quotient rounded to a double first as Python does.
    const bool wide = width >= height;
    const uint32_t shorter = wide ? height : width;
    const uint32_t longer = wide ? width : height;
    const uint32_t scaled = std::max(1u, uint32_t(double(uint64_t(size) * longer) / shorter));
    recordResize(wide ? scaled : size, wide ? size : scaled, mode);
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

    steps.push_back({dstW, dstH, {0, 0, dstW, dstH}, size, padX, padY, padValue, mode});
    width = size;
    height = size;
}

// No pass of its own: the crop narrows what the last resize writes, or what the
// first pass reads when it comes before any.
void ImageProcessor::recordCenterCrop(uint32_t cropWidth, uint32_t cropHeight)
{
    if (cropWidth == 0 || cropHeight == 0 || cropWidth > width || cropHeight > height)
        throw std::invalid_argument("centerCrop size must be non-zero and fit the image");
    if (!steps.empty() && steps.back().canvas)
        throw std::logic_error("centerCrop() cannot follow letterbox()");

    const uint32_t x = (width - cropWidth) / 2;
    const uint32_t y = (height - cropHeight) / 2;
    imageGeometry.padX -= x;
    imageGeometry.padY -= y;

    Window& window = steps.empty() ? view : steps.back().keep;
    window = {window.x + x, window.y + y, cropWidth, cropHeight};
    width = cropWidth;
    height = cropHeight;
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
    const uint32_t outW = step.canvas ? step.canvas : step.keep.width;
    const uint32_t outH = step.canvas ? step.canvas : step.keep.height;
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

    const Target target{step.width, step.height, step.keep};
    if (isFloat)
        resampleImage(sourceOf(pixels.data(), storedChw, bufWidth, bufHeight, channels, view), target, step.mode, emit);
    else
        resampleImage(sourceOf(source.data(), false, bufWidth, bufHeight, channels, view), target, step.mode, emit);

    pixels = std::move(dst);
    source = {};
    isFloat = true;
    storedChw = outputChw;
    bufWidth = outW;
    bufHeight = outH;
    view = {0, 0, outW, outH};
    if (last)
    {
        storedBgr = outputBgr;
        storedNormalized = normalized;
        storedMean = normMean;
        storedStd = normStd;
    }
}

// The last pass when nothing resamples: the decoded bytes, or the previous output, to
// the output. A whole float image already in the output layout is adjusted in place.
void ImageProcessor::finish()
{
    Emit emit{nullptr, strides(outputChw, view.width, view.height, channels)};
    emit.normalize = normalized && !storedNormalized;
    emit.mean = normMean;
    emit.std = normStd;
    emit.swapRB = storedBgr != outputBgr;

    const bool whole = view.width == bufWidth && view.height == bufHeight;
    if (isFloat && storedChw == outputChw && whole)
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
        std::vector<float> dst(size_t(channels) * view.width * view.height);
        emit.base = dst.data();
        if (isFloat)
            convert(sourceOf(pixels.data(), storedChw, bufWidth, bufHeight, channels, view), emit);
        else
            convert(sourceOf(source.data(), false, bufWidth, bufHeight, channels, view), emit);
        pixels = std::move(dst);
        source = {};
        isFloat = true;
        storedChw = outputChw;
        bufWidth = view.width;
        bufHeight = view.height;
        view = {0, 0, bufWidth, bufHeight};
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
