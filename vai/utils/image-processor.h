/*=============================================================================
  Written by   oneway@vaistrata.com
  Image preprocessing - decodes an image, resizes it, normalizes it and hands
  the float pixels over without a copy.

  Knows nothing about Tensor. The caller wraps the result:
    std::vector<float> pixels = ImageProcessor(path)
                                    .resize(896, Resize::BilinearAntialiasInt16)
                                    .normalize(-1.0f, 1.0f)
                                    .chw();
    Tensor image = Tensor({3, 896, 896}).setFloats(pixels);

  resize() and letterbox() run in call order. normalize(), standardize(), bgr()
  and chw() describe the output, so where and how often they are called does not
  matter; they ride on the writes of the last resize or letterbox, and
  without one a single pass turns the decoded bytes into the output.

  The steps only record; converting to std::vector<float> runs them. Pixels are RGB,
  HWC and 0~255 unless bgr(), chw(), normalize() or standardize() says otherwise,
  and padValue is a 0~255 pixel value normalized with the image.
=============================================================================*/

#ifndef VAI_UTILS_IMAGE_PROCESSOR_H
#define VAI_UTILS_IMAGE_PROCESSOR_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>


namespace vai::utils {

enum class Resize
{
    Bilinear,                   // half-pixel centres, as cv2 and torchvision without antialias
    BilinearAlignCorners,       // corner pixels map to corner pixels
    BilinearAntialias,          // Pillow: 22-bit integer coefficients, rounded to 0~255 after each axis
    BilinearAntialiasInt16,     // torchvision uint8 antialias: as above, coefficients capped to fit int16
};

enum class Align
{
    Center,
    TopLeft,
};

// outputXY = sourceXY * scale + pad, per axis. Undo it to put boxes back on the source image.
struct ImageGeometry
{
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};


class ImageProcessor
{
public:
    explicit ImageProcessor(const std::string& path, uint32_t channels = 3);

    // pixels is HWC, width * height * channels bytes.
    ImageProcessor(const uint8_t* pixels, uint32_t width, uint32_t height, uint32_t channels = 3);

    // Each step returns the processor as it was called, a named one as an
    // l-value and a temporary chain as an r-value, so the conversion below
    // knows whether it may take the buffer.

    // Squashes the image to size x size, ignoring the aspect ratio.
    auto&& resize(this auto&& self, uint32_t size, Resize mode = Resize::Bilinear)
    {
        self.recordResize(size, mode);
        return std::forward<decltype(self)>(self);
    }

    // Scales the longer side to size and fills the rest of the square with padValue.
    auto&& letterbox(this auto&& self, uint32_t size, Align align, float padValue, Resize mode = Resize::Bilinear)
    {
        self.recordLetterbox(size, align, padValue, mode);
        return std::forward<decltype(self)>(self);
    }

    // Maps 0~255 linearly onto [min, max].
    auto&& normalize(this auto&& self, float min, float max)
    {
        self.recordNormalize(min, max);
        return std::forward<decltype(self)>(self);
    }

    // Then (v - mean[c]) / std[c] per output channel, in normalize()'s range.
    auto&& standardize(this auto&& self, const std::array<float, 3>& mean, const std::array<float, 3>& std)
    {
        self.recordStandardize(mean, std);
        return std::forward<decltype(self)>(self);
    }

    // Hands the channels over as BGR instead of RGB.
    auto&& bgr(this auto&& self)
    {
        self.recordBgr();
        return std::forward<decltype(self)>(self);
    }

    // Hands the pixels over as CHW instead of HWC.
    auto&& chw(this auto&& self)
    {
        self.outputChw = true;
        return std::forward<decltype(self)>(self);
    }

    // A named processor lends its buffer, which a later run replaces; a
    // temporary hands it over without a copy.
    operator const std::vector<float>&() &;
    operator std::vector<float>() &&;

    const ImageGeometry& geometry() const { return imageGeometry; }

private:
    struct Resample
    {
        uint32_t width;             // the resized image
        uint32_t height;
        uint32_t canvas;            // square it sits in, 0 for none
        uint32_t padX;
        uint32_t padY;
        float padValue;
        Resize mode;
    };

    void recordResize(uint32_t size, Resize mode);
    void recordLetterbox(uint32_t size, Align align, float padValue, Resize mode);
    void recordNormalize(float min, float max);
    void recordStandardize(const std::array<float, 3>& mean, const std::array<float, 3>& std);
    void recordBgr();
    void run();
    void resample(const Resample& step, bool last);
    void finish();

    std::vector<uint8_t> source;    // decoded HWC bytes until the first pass
    std::vector<float> pixels;      // output of the last pass
    bool isFloat = false;           // pixels holds the image rather than source
    bool storedChw = false;         // layout of pixels
    bool storedBgr = false;         // channel order of pixels
    bool storedNormalized = false;  // pixels went through storedMean and storedStd
    std::array<float, 4> storedMean{};
    std::array<float, 4> storedStd{};
    bool outputChw = false;         // what the conversion hands over
    bool outputBgr = false;
    bool normalized = false;
    float rangeMin = 0.0f;   // normalize()
    float rangeMax = 255.0f;
    std::array<float, 4> statMean{0.0f, 0.0f, 0.0f, 0.0f};   // standardize()
    std::array<float, 4> statStd{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> normMean{};   // both as (v - normMean[c]) / normStd[c], set by run()
    std::array<float, 4> normStd{};
    uint32_t bufWidth;              // size of the image held now
    uint32_t bufHeight;
    uint32_t width;                 // size after every recorded step
    uint32_t height;
    uint32_t channels;
    std::vector<Resample> steps;
    ImageGeometry imageGeometry;
};


}   // namespace vai::utils

#endif   // VAI_UTILS_IMAGE_PROCESSOR_H
