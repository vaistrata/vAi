#include "python/bindings/models/lm.h"
#include "python/bindings/models/models.h"
#include "python/bindings/convert.h"
#include "python/bindings/models/config-base.h"
#include "models/production/gemma3/net.h"
#include "models/production/gemma3/google-weights.h"
#include "models/production/gemma3/processing.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gemma3 = model::production::gemma3;


namespace pyvai {

namespace {

const Presets<gemma3::ModelConfig>& gemma3Presets()
{
    static const Presets<gemma3::ModelConfig> presets = {
        {"270M", &gemma3::ModelConfig::_270M},
        {"1B", &gemma3::ModelConfig::_1B},
        {"4B", &gemma3::ModelConfig::_4B},
        {"12B", &gemma3::ModelConfig::_12B},
    };
    return presets;
}


// The vision tower's ctor asserts its pooling, and the image tokens are
// written into rows of the decoder's width.
void checkVisionTower(const gemma3::group::VisionTowerConfig& v, uint32_t textEmbedDim)
{
    checkVisionConfig(v, "Gemma3Net", "config.vision_cfg");
    if (v.patchesPerSide() != 4 * v.tokensPerSide)
        configError("Gemma3Net", "config.vision_cfg: image_size / patch_size must be " +
                                     std::to_string(4 * v.tokensPerSide) +
                                     ": the tower pools 4 x 4 patches into each of its " +
                                     std::to_string(v.tokensPerImage) + " tokens");
    if (v.outDim != textEmbedDim)
        configError("Gemma3Net", "config.vision_cfg.out_dim must be the decoder's embed_dim "
                                 "(config.language_cfg.layer_cfg.embed_dim)");
}

}   // namespace


void bindGemma3(nb::module_& m)
{
    nb::class_<gemma3::group::VisionTowerConfig, VisionModelConfig>(
        m, "Gemma3VisionTowerConfig",
        "The vision half of a Gemma 3 VLM checkpoint (Gemma3Config.vision_cfg): "
        "SigLIP So400m/14 at 896 pixels a side (a VisionModelConfig), average-pooled "
        "to tokens_per_image soft tokens and projected to the decoder's width.")
        .def(nb::init<uint32_t>(), nb::arg("out_dim"),
             "The checkpoints' tower, projecting to `out_dim`: the decoder's embed_dim.")
        .def_rw("out_dim", &gemma3::group::VisionTowerConfig::outDim,
                "Width the image tokens are projected to: the decoder's embed_dim.")
        .def_rw("image_token_id", &gemma3::group::VisionTowerConfig::imageTokenId,
                "Id of <image_soft_token>, the placeholder an image's tokens replace.")
        .def_prop_ro_static("tokens_per_side",
                            [](nb::handle) { return gemma3::group::VisionTowerConfig::tokensPerSide; })
        .def_prop_ro_static(
            "tokens_per_image", [](nb::handle) { return gemma3::group::VisionTowerConfig::tokensPerImage; },
            "Soft tokens per image: the length of its placeholder run in a prompt.");

    auto cfgCls = nb::class_<gemma3::ModelConfig>(
        m, "Gemma3Config",
        "A Gemma 3 checkpoint (models/production/gemma3/net.h), built by preset(variant); "
        "the 4B and 12B VLMs carry a vision half. Every sixth layer is global (whole "
        "prefix, global_rope) and the rest local (a sliding window of sliding_window "
        "positions in a ring cache, local_rope).");
    bindPresets(cfgCls, "Gemma3Config", gemma3Presets());
    cfgCls
        .def_rw("language_cfg", &gemma3::ModelConfig::languageCfg,
                "The decoder stack and its token table (LanguageModelConfig).")
        .def_rw("sliding_window", &gemma3::ModelConfig::slidingWindow,
                "A local layer's span: the last positions a query attends.")
        .def_rw("global_rope", &gemma3::ModelConfig::globalRope, "RoPE of the global layers (RopeConfig).")
        .def_rw("local_rope", &gemma3::ModelConfig::localRope, "RoPE of the local layers (RopeConfig).")
        .def_rw("vision_cfg", &gemma3::ModelConfig::visionCfg,
                "The vision half (Gemma3VisionTowerConfig), or None for a text-only "
                "checkpoint. Edits through the property land in place.")
        .def_static("is_global_layer", &gemma3::ModelConfig::isGlobalLayer, nb::arg("layer"),
                    "Whether `layer` attends the whole prefix rather than the sliding window.");

    auto cls =
        nb::class_<gemma3::Net, vai::NeuralNet>(m, "Gemma3Net")
            .def(
                "__init__",
                [](gemma3::Net* self, const gemma3::ModelConfig& config, vai::Dtype w_default_dtype,
                   vai::Dtype o_default_dtype, std::optional<eva::Device> device,
                   std::optional<uint32_t> max_runtime_context, uint32_t prefill_chunk, vai::Dtype kv_cache_dtype) {
                    checkLanguageConfig(config.languageCfg, "Gemma3Net", "config.language_cfg",
                                        /*clampsContext*/ true);
                    if (config.visionCfg)
                    {
                        checkVisionTower(*config.visionCfg, config.languageCfg.layerCfg.embedDim);
                        if (prefill_chunk > 0 && prefill_chunk < config.visionCfg->tokensPerImage)
                            configError("Gemma3Net", "prefill_chunk must hold an image's tokens (0 or >= " +
                                                         std::to_string(config.visionCfg->tokensPerImage) + ")");
                    }
                    const uint32_t context = runtimeContext<gemma3::RuntimeConfig>(max_runtime_context, "Gemma3Net");
                    const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                    new (self) gemma3::Net(cfg, config,
                                           {
                                               .kvCacheDtype = kv_cache_dtype,
                                               .maxRuntimeContext = context,
                                               .prefillChunk = prefill_chunk,
                                           });
                },
                nb::arg("config") = gemma3::ModelConfig::_1B(), nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                nb::arg("o_default_dtype") = vai::Dtype::Fp32, nb::arg("device") = nb::none(),
                nb::arg("max_runtime_context") = nb::none(),
                nb::arg("prefill_chunk") = gemma3::RuntimeConfig{}.prefillChunk,
                nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                "Build a Gemma 3 model (default config: Gemma3Config.preset('1B')). "
                "`max_runtime_context` bounds the KV cache of the global layers, clamped "
                "to the checkpoint's max_trained_context (None: all of it); a local layer "
                "keeps sliding_window + prefill_chunk - 1 positions, rounded up to 256. A "
                "prompt longer than `prefill_chunk` runs as several forwards, cut between "
                "images (0: as one).");
    bindLoadWeights<gemma3::Net>(cls, "Gemma3Net", gemma3::google::loadWeights,
                                 "Load the upstream HuggingFace Gemma 3 checkpoint (google/gemma-3-1b-it, "
                                 "-4b-it, ...) from its folder (data/ckpts/gemma3/google/<variant>): every "
                                 "*.safetensors shard in it is read.");

    cls.def(
        "prefill",
        [](nb::handle_t<gemma3::Net> self, nb::handle images,
           nb::handle_t<vai::Tensor> prompt_token_ids) -> nb::object {
            auto& model = nb::cast<gemma3::Net&>(self);
            const char* who = "Gemma3Net.prefill";
            PrefillArgs a = prefillArgs(self, prompt_token_ids, who);
            vai::Tensor batch;   // empty: text only
            if (!images.is_none())
            {
                if (!nb::isinstance<vai::Tensor>(images))
                    throw nb::type_error("Gemma3Net.prefill: images must be a Tensor or None");
                if (!model.modelCfg.visionCfg)
                    throw nb::value_error("Gemma3Net.prefill: this checkpoint has no vision tower "
                                          "(config.vision_cfg is None)");
                batch = stageInput(self, "inImages", images, who);
                // The model asserts the shape.
                const auto& v = *model.modelCfg.visionCfg;
                const auto& shape = batch.shape();
                const size_t r = shape.size();
                if ((r != 3 && r != 4) || shape[r - 3] != v.imageChannels || shape[r - 2] != v.imageSize ||
                    shape[r - 1] != v.imageSize)
                    throw nb::value_error(("Gemma3Net.prefill: images must be [B, " + std::to_string(v.imageChannels) +
                                           ", " + std::to_string(v.imageSize) + ", " + std::to_string(v.imageSize) +
                                           "], or one image without the B")
                                              .c_str());
            }
            {
                nb::gil_scoped_release rel;   // waits the previous forward
                model.prefill(batch, a.tokens, a.token);
            }
            return a.out;
        },
        nb::arg("images").none(), nb::arg("prompt_token_ids"),
        "Run the prompt ([T] uint32 ids, Tensor.from_u32) on top of the KV cache and "
        "return the next token: a [1] Tensor to read with to_u32() and hand to decode(), "
        "the same one every time. Appends T tokens; nothing checks the budget.\n\n"
        "`images` is None for text only; a VLM takes [B, 3, S, S] floats in [-1, 1] (or "
        "[3, S, S] for one image). The prompt must hold one run of "
        "vision_cfg.tokens_per_image image_token_id placeholders per image, in order "
        "(gemma3_expand_image_placeholders writes them); RuntimeError on a mismatch.");
    bindDecodeLoop<gemma3::Net>(cls, "Gemma3Net");
    cls.def_prop_ro(
        "prefill_chunk", [](gemma3::Net& self) { return self.runtimeCfg.prefillChunk; },
        "Prefill chunk length; 0 runs the whole prompt in one forward.");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inImages", "inTokenIds"}, {"nextTokenId"});

    // models/production/gemma3/processing.h: what the demo and the chat build a prompt with.
    m.def(
        "gemma3_expand_image_placeholders",
        [](const std::string& prompt, uint32_t tokens_per_image) {
            return gemma3::expandImagePlaceholders(prompt, tokens_per_image);
        },
        nb::arg("prompt"), nb::arg("tokens_per_image"),
        "Replace every <start_of_image> in `prompt` with the HuggingFace processor's "
        "block for one image: two newlines, <start_of_image>, `tokens_per_image` "
        "<image_soft_token> placeholders, <end_of_image>, two newlines.");

    m.def(
        "gemma3_take_image_tags",
        [](const std::string& message) {
            std::vector<std::string> paths;
            std::string text = gemma3::takeImageTags(message, paths);
            return nb::make_tuple(text, paths);
        },
        nb::arg("message"),
        "Turn every <image:file> tag in `message` into <start_of_image>. Returns "
        "(message, files): the files in the order their tags stood.");

    m.def(
        "gemma3_load_images",
        [](const std::vector<std::string>& paths, uint32_t image_size) {
            if (image_size == 0)
                throw nb::value_error("gemma3_load_images: image_size must be >= 1");
            for (const std::string& path : paths)
                ensureFileExists("gemma3_load_images", path);
            std::vector<float> pixels;
            {
                nb::gil_scoped_release rel;   // decode + resample, on every core
                pixels = gemma3::loadImages(paths, image_size);
            }
            return ownedNdarray(std::move(pixels), {paths.size(), 3, image_size, image_size});
        },
        nb::arg("paths"), nb::arg("image_size"),
        "Decode the image files and squash each to image_size x image_size as "
        "Gemma3ImageProcessor does (torchvision's uint8 antialias), scaled to "
        "[-1, 1]: a [B, 3, image_size, image_size] float32 array for prefill().");
}

}   // namespace pyvai
