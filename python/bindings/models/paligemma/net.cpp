#include "python/bindings/models/models.h"
#include "python/bindings/models/lm.h"
#include "python/bindings/models/config-base.h"
#include "models/production/paligemma/net.h"
#include "models/production/paligemma/google-weights.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace paligemma = model::production::paligemma;


namespace pyvai {

namespace {

const Presets<paligemma::ModelConfig>& paligemmaPresets()
{
    static const Presets<paligemma::ModelConfig> presets = {
        {"3B-224", &paligemma::ModelConfig::_3B_224},
        {"3B-448", &paligemma::ModelConfig::_3B_448},
    };
    return presets;
}

}   // namespace


void bindPaliGemma(nb::module_& m)
{
    auto cfgCls = nb::class_<paligemma::ModelConfig>(
        m, "PaliGemmaConfig",
        "A PaliGemma checkpoint (models/production/paligemma/net.h): SigLIP So400m/14 at the "
        "variant's resolution and the Gemma 2B decoder, built by preset(variant).");
    bindPresets(cfgCls, "PaliGemmaConfig", paligemmaPresets());
    cfgCls
        .def_rw("vision_cfg", &paligemma::ModelConfig::visionCfg,
                "The SigLIP tower (VisionModelConfig); its image_size is the variant's.")
        .def_rw("language_cfg", &paligemma::ModelConfig::languageCfg,
                "The Gemma decoder and 257152-row token table (LanguageModelConfig).");

    auto cls =
        nb::class_<paligemma::Net, vai::NeuralNet>(m, "PaliGemmaNet")
            .def(
                "__init__",
                [](paligemma::Net* self, const paligemma::ModelConfig& config, vai::Dtype w_default_dtype,
                   vai::Dtype o_default_dtype, std::optional<eva::Device> device,
                   std::optional<uint32_t> max_runtime_context, vai::Dtype kv_cache_dtype) {
                    checkVisionConfig(config.visionCfg, "PaliGemmaNet", "config.vision_cfg");
                    checkLanguageConfig(config.languageCfg, "PaliGemmaNet", "config.language_cfg",
                                        /*clampsContext*/ true);
                    const uint32_t context =
                        runtimeContext<paligemma::RuntimeConfig>(max_runtime_context, "PaliGemmaNet");
                    const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                    new (self) paligemma::Net(cfg, config,
                                              {
                                                  .kvCacheDtype = kv_cache_dtype,
                                                  .maxRuntimeContext = context,
                                              });
                },
                nb::arg("config") = paligemma::ModelConfig::_3B_224(), nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                nb::arg("o_default_dtype") = vai::Dtype::Fp32, nb::arg("device") = nb::none(),
                nb::arg("max_runtime_context") = nb::none(), nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                "Build a PaliGemma model (default config: PaliGemmaConfig.preset('3B-224')). "
                "`max_runtime_context` bounds the KV cache, clamped to the checkpoint's "
                "max_trained_context (None: all 8K of it); an image costs num_patches of it.");
    bindLoadWeights<paligemma::Net>(
        cls, "PaliGemmaNet", paligemma::google::loadWeights,
        "Load the upstream HuggingFace PaliGemma checkpoint (google/paligemma-3b-mix-224 "
        "or -448) from its folder (data/ckpts/paligemma/google/<variant>): every *.safetensors "
        "shard in it is read.");

    cls.def(
        "prefill",
        [](nb::handle_t<paligemma::Net> self, nb::handle_t<vai::Tensor> image,
           nb::handle_t<vai::Tensor> prompt_token_ids) -> nb::object {
            auto& model = nb::cast<paligemma::Net&>(self);
            const char* who = "PaliGemmaNet.prefill";
            PrefillArgs a = prefillArgs(self, prompt_token_ids, who);
            vai::Tensor img = stageInput(self, "inImage", image, who);
            const auto& v = model.modelCfg.visionCfg;
            if (!img.isShapeOf(v.imageChannels, v.imageSize, v.imageSize))
                throw nb::value_error(("PaliGemmaNet.prefill: image must be [" + std::to_string(v.imageChannels) +
                                       ", " + std::to_string(v.imageSize) + ", " + std::to_string(v.imageSize) + "]")
                                          .c_str());
            {
                nb::gil_scoped_release rel;   // waits the previous forward
                model.prefill(img, a.tokens, a.token);
            }
            return a.out;
        },
        nb::arg("image"), nb::arg("prompt_token_ids"),
        "Open a sequence with a picture: `image` ([3, S, S] floats in [-1, 1]) goes "
        "into the KV cache as num_patches soft tokens ahead of the prompt ([T] uint32 "
        "ids, Tensor.from_u32). Returns the next token: a [1] Tensor to read with "
        "to_u32() and hand to decode(), the same one every time. Appends num_patches "
        "+ T tokens; nothing checks the budget.");
    bindDecodeLoop<paligemma::Net>(cls, "PaliGemmaNet");
    cls.def_prop_ro(
           "num_patches", [](paligemma::Net& self) { return self.modelCfg.visionCfg.numPatches(); },
           "Soft tokens one image adds to the context (vision_cfg.num_patches).")
        .def_prop_ro(
            "image_size", [](paligemma::Net& self) { return self.modelCfg.visionCfg.imageSize; },
            "Side of the square image prefill() takes (vision_cfg.image_size).");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inImage", "inTokenIds"}, {"nextTokenId"});
}

}   // namespace pyvai
