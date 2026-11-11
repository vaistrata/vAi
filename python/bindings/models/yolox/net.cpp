#include "python/bindings/models/models.h"
#include "python/bindings/models/yolo.h"
#include "models/experimental/yolox/processing.h"
#include "models/experimental/yolox/net.h"
#include "models/experimental/yolox/megvii-weights.h"
#include <string>
#include <utility>

namespace yolox = model::experimental::yolox;


namespace pyvai {

void bindYOLOX(nb::module_& m)
{
    // Members are the C++ enumerator names; from_letter() maps the size letters.
    auto tierEnum = nb::enum_<yolox::Tier>(m, "YOLOXTier")
                        .value("SMALL", yolox::Tier::SMALL)
                        .value("MEDIUM", yolox::Tier::MEDIUM)
                        .value("LARGE", yolox::Tier::LARGE)
                        .value("XLARGE", yolox::Tier::XLARGE);
    bindTierLetters(tierEnum, "YOLOXTier", "smlx",
                    "The tier for a Megvii size letter: 's' SMALL, 'm' MEDIUM, "
                    "'l' LARGE, 'x' XLARGE.",
                    "The Megvii size letter of this tier (yolox_<letter>).");

    using yolox::Detection;
    nb::class_<Detection>(m, "YOLOXDetection")
        .def_ro("x1", &Detection::x1)
        .def_ro("y1", &Detection::y1)
        .def_ro("x2", &Detection::x2)
        .def_ro("y2", &Detection::y2)
        .def_ro("confidence", &Detection::confidence)
        .def_ro("class_id", &Detection::classId)
        .def_ro("class_name", &Detection::className)
        .def("__repr__", [](const Detection& d) {
            return "<YOLOXDetection " + d.className + " " + std::to_string(d.confidence) + ">";
        });

    auto cls =
        nb::class_<yolox::Net, vai::NeuralNet>(m, "YOLOXNet")
            .def(
                "__init__",
                [](yolox::Net* self, yolox::Tier tier, uint32_t image_size, float conf_threshold, float iou_threshold,
                   vai::Dtype w_default_dtype, vai::Dtype o_default_dtype, std::optional<eva::Device> device) {
                    // The core ctor asserts on an illegal size.
                    if (image_size < 32 || image_size % 32 != 0)
                        throw nb::value_error("YOLOXNet: image_size must be a multiple of 32, at least 32");
                    yolox::ModelConfig mcfg = yolox::getConfigCOCO(tier);
                    yolox::RuntimeConfig run{
                        .imageSize = image_size,
                        .confThreshold = conf_threshold,
                        .iouThreshold = iou_threshold,
                    };
                    const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                    new (self) yolox::Net(cfg, mcfg, run);
                },
                nb::arg("tier") = yolox::Tier::LARGE, nb::arg("image_size") = 640u, nb::arg("conf_threshold") = 0.25f,
                nb::arg("iou_threshold") = 0.45f, nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                nb::arg("o_default_dtype") = vai::Dtype::Fp32, nb::arg("device") = nb::none(),
                "Build a YOLOX model (COCO config). image_size must be a multiple "
                "of 32. out0 is [N, 5 + 80]: one row per anchor of cx, cy, w, h, "
                "objectness and the class scores, in network pixels.");
    bindLoadWeights<yolox::Net>(
        cls, "YOLOXNet", yolox::megvii::loadWeights,
        "Load weights from a checkpoint folder (data/ckpts/yolox/megvii/<tier name>; the "
        "Megvii .pth converted with models/experimental/yolox/megvii-weights-convertor.py): every "
        "*.safetensors in it is read, and it must match this model's tier.");
    cls.def_prop_ro(
        "num_classes", [](yolox::Net& self) { return self.cfg.numClasses; }, "Class count of the head (80, COCO).");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"in0"}, {"out0"});

    // The input ndarray is read in place, with the GIL released.
    m.def(
        "yolox_postprocess",
        [](nb::ndarray<const float, nb::c_contig, nb::device::cpu> data, const vai::utils::ImageGeometry& geometry,
           float conf_threshold, float iou_threshold, uint32_t num_classes) {
            // postprocess() derives the anchor count as size / (5 + classes)
            // and would silently misread anything else.
            const size_t channels = 5 + size_t(num_classes);
            if (num_classes == 0 || data.size() == 0 || data.size() % channels != 0)
                throw nb::value_error(("yolox_postprocess: expected an [N, " + std::to_string(channels) +
                                       "] output (a multiple of " + std::to_string(channels) + " floats), got " +
                                       std::to_string(data.size()))
                                          .c_str());
            // The boxes are divided by them.
            if (!(geometry.scaleX > 0.0f) || !(geometry.scaleY > 0.0f))
                throw nb::value_error("yolox_postprocess: geometry.scale_x and scale_y must be > 0");
            std::vector<Detection> dets;
            {
                nb::gil_scoped_release rel;
                dets = yolox::postprocess(std::vector<float>(data.data(), data.data() + data.size()), geometry,
                                          conf_threshold, iou_threshold, num_classes);
            }
            return dets;
        },
        nb::arg("data"), nb::arg("geometry"), nb::arg("conf_threshold"), nb::arg("iou_threshold"),
        nb::arg("num_classes") = yolox::NUM_COCO_CLASSES,
        "Objectness * class score filter + NMS over the network output: a float32 "
        "array of [N, 5 + num_classes] rows (cx, cy, w, h, objectness, class "
        "scores), flat or 2-D. Boxes are mapped back through `geometry`, the "
        "ImageProcessor's that letterboxed the image, and clipped to the source "
        "image. ValueError when the size is not a multiple of 5 + num_classes.");
}

}   // namespace pyvai
