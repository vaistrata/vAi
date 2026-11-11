#include "python/bindings/module.h"
#include "python/bindings/checks.h"
#include "python/bindings/convert.h"
#include "vai/utils/image-processor.h"
#include <nanobind/stl/filesystem.h>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>


namespace pyvai {

using vai::utils::Align;
using vai::utils::ImageGeometry;
using vai::utils::ImageProcessor;
using vai::utils::Resize;

namespace {

// The C++ class hands its pixels over flat and keeps their layout private, so
// what shapes the ndarray is tracked beside it.
struct PyImageProcessor
{
    ImageProcessor proc;
    uint32_t width;       // of the output, after every recorded step
    uint32_t height;
    uint32_t channels;
    bool chw = false;
};

}   // namespace


void bindImageProcessor(nb::module_& m)
{
    nb::enum_<Resize>(m, "Resize", "How ImageProcessor resamples.")
        .value("BILINEAR", Resize::Bilinear,
               "Half-pixel centres, as cv2 and torchvision without antialias.")
        .value("BILINEAR_ALIGN_CORNERS", Resize::BilinearAlignCorners,
               "Corner pixels map to corner pixels.")
        .value("BILINEAR_ANTIALIAS", Resize::BilinearAntialias,
               "Pillow: 22-bit integer coefficients, rounded to 0..255 after each axis.")
        .value("BILINEAR_ANTIALIAS_INT16", Resize::BilinearAntialiasInt16,
               "torchvision uint8 antialias: as above, coefficients capped to fit int16.");

    nb::enum_<Align>(m, "Align", "Where a letterbox puts the image in its square.")
        .value("CENTER",   Align::Center)
        .value("TOP_LEFT", Align::TopLeft);

    nb::class_<ImageGeometry>(m, "ImageGeometry",
        "How an ImageProcessor's output maps onto the source image: "
        "output_xy = source_xy * scale + pad, per axis. Undo it to put boxes "
        "back on the source.")
        .def(
            "__init__",
            [](ImageGeometry* self, uint32_t source_width, uint32_t source_height,
               float scale_x, float scale_y, float pad_x, float pad_y)
            {
                new (self) ImageGeometry{source_width, source_height,
                                         scale_x, scale_y, pad_x, pad_y};
            },
            nb::kw_only(),
            nb::arg("source_width")  = 0u,
            nb::arg("source_height") = 0u,
            nb::arg("scale_x")       = 1.0f,
            nb::arg("scale_y")       = 1.0f,
            nb::arg("pad_x")         = 0.0f,
            nb::arg("pad_y")         = 0.0f)
        .def_rw("source_width",  &ImageGeometry::sourceWidth)
        .def_rw("source_height", &ImageGeometry::sourceHeight)
        .def_rw("scale_x",       &ImageGeometry::scaleX)
        .def_rw("scale_y",       &ImageGeometry::scaleY)
        .def_rw("pad_x",         &ImageGeometry::padX)
        .def_rw("pad_y",         &ImageGeometry::padY)
        .def("__repr__", [](const ImageGeometry& g) {
            return "ImageGeometry(source=" + std::to_string(g.sourceWidth) + "x"
                 + std::to_string(g.sourceHeight) + ", scale=(" + std::to_string(g.scaleX)
                 + ", " + std::to_string(g.scaleY) + "), pad=(" + std::to_string(g.padX)
                 + ", " + std::to_string(g.padY) + "))";
        });

    nb::class_<PyImageProcessor>(m, "ImageProcessor",
        "Image preprocessing: decodes an image, resizes it, normalizes it and "
        "hands the float pixels over.\n\n"
        "    chw = (pyvai.ImageProcessor(\"dog.jpg\")\n"
        "           .letterbox(640, pyvai.Align.CENTER, 114.0)\n"
        "           .normalize(0.0, 1.0)\n"
        "           .chw()\n"
        "           .to_numpy())                    # [3, 640, 640] float32\n\n"
        "resize() and letterbox() run in call order; normalize(), bgr() and chw() "
        "describe the output, wherever they are called. The steps only record and "
        "to_numpy() runs them. Pixels are RGB, HWC and 0..255 by default.")
        .def(
            "__init__",
            [](PyImageProcessor* self, const std::filesystem::path& path, uint32_t channels)
            {
                const std::string p = path.string();
                ensureFileExists("ImageProcessor", p);
                if (channels == 0)
                    throw nb::value_error("ImageProcessor: channels must be >= 1");
                ImageProcessor proc(p, channels);   // RuntimeError when it does not decode
                const ImageGeometry g = proc.geometry();
                new (self) PyImageProcessor{
                    std::move(proc), g.sourceWidth, g.sourceHeight, channels};
            },
            nb::arg("path"),
            nb::arg("channels") = 3u,
            "Decode the image file at `path` (JPEG, PNG, ...) to `channels` channels: "
            "3 for RGB, 1 for gray."
        )
        .def(
            "__init__",
            [](PyImageProcessor* self,
               nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu> pixels)
            {
                if (pixels.ndim() != 2 && pixels.ndim() != 3)
                    throw nb::value_error(
                        "ImageProcessor: pixels must be an [H, W, C] or [H, W] uint8 array");
                const uint32_t height   = static_cast<uint32_t>(pixels.shape(0));
                const uint32_t width    = static_cast<uint32_t>(pixels.shape(1));
                const uint32_t channels = pixels.ndim() == 3
                    ? static_cast<uint32_t>(pixels.shape(2))
                    : 1u;
                if (width == 0 || height == 0 || channels == 0)
                    throw nb::value_error("ImageProcessor: image dimensions must be non-zero");
                new (self) PyImageProcessor{ImageProcessor(pixels.data(), width, height, channels),
                                            width, height, channels};
            },
            nb::arg("pixels").noconvert(),   // no silent cast of a float image to bytes
            "Take already decoded pixels: an [H, W, C] uint8 array (C-contiguous), "
            "or [H, W] for one channel. The bytes are copied."
        )
        .def(
            "resize",
            [](nb::handle_t<PyImageProcessor> self, uint32_t size, Resize mode) -> nb::object
            {
                auto& p = nb::cast<PyImageProcessor&>(self);
                p.proc.resize(size, mode);   // ValueError on size 0
                p.width = p.height = size;
                return nb::borrow(self);
            },
            nb::arg("size"),
            nb::arg("mode") = Resize::Bilinear,
            "Squash the image to size x size, ignoring the aspect ratio."
        )
        .def(
            "letterbox",
            [](nb::handle_t<PyImageProcessor> self, uint32_t size, Align align,
               float pad_value, Resize mode) -> nb::object
            {
                auto& p = nb::cast<PyImageProcessor&>(self);
                p.proc.letterbox(size, align, pad_value, mode);   // ValueError on size 0
                p.width = p.height = size;
                return nb::borrow(self);
            },
            nb::arg("size"),
            nb::arg("align"),
            nb::arg("pad_value"),
            nb::arg("mode") = Resize::Bilinear,
            "Scale the longer side to `size` and fill the rest of the square with "
            "`pad_value`, a 0..255 pixel value normalized with the image."
        )
        .def(
            "normalize",
            [](nb::handle_t<PyImageProcessor> self, float min, float max) -> nb::object
            {
                // This is a ValueError unless min < max.
                nb::cast<PyImageProcessor&>(self).proc.normalize(min, max);
                return nb::borrow(self);
            },
            nb::arg("min"),
            nb::arg("max"),
            "Map 0..255 linearly onto [min, max]."
        )
        .def(
            "bgr",
            [](nb::handle_t<PyImageProcessor> self) -> nb::object
            {
                auto& p = nb::cast<PyImageProcessor&>(self);
                if (p.channels != 3)   // a std::logic_error in the core
                    throw nb::value_error("ImageProcessor.bgr: needs 3 channels");
                p.proc.bgr();
                return nb::borrow(self);
            },
            "Hand the channels over as BGR instead of RGB."
        )
        .def(
            "chw",
            [](nb::handle_t<PyImageProcessor> self) -> nb::object
            {
                auto& p = nb::cast<PyImageProcessor&>(self);
                p.proc.chw();
                p.chw = true;
                return nb::borrow(self);
            },
            "Hand the pixels over as CHW instead of HWC."
        )
        .def(
            "to_numpy",
            [](PyImageProcessor& self)
            {
                // The processor lends its buffer and a later run replaces it,
                // so the ndarray gets a copy of its own.
                std::vector<float> pixels;
                {
                    nb::gil_scoped_release rel;   // the resampling runs on every core
                    const std::vector<float>& lent = self.proc;
                    pixels = lent;
                }
                const std::vector<size_t> shape = self.chw
                    ? std::vector<size_t>{self.channels, self.height, self.width}
                    : std::vector<size_t>{self.height, self.width, self.channels};
                return ownedNdarray(std::move(pixels), shape);
            },
            "Run the recorded steps and return the pixels as a float32 array, "
            "[H, W, C] or, after chw(), [C, H, W]. The processor can go on from "
            "here: a later run starts from this output."
        )
        .def_prop_ro(
            "geometry",
            [](const PyImageProcessor& self) { return self.proc.geometry(); },
            "How the output maps onto the source image after the steps recorded "
            "so far (a copy)."
        );
}

}   // namespace pyvai
