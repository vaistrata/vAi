#include "python/bindings/models/lm.h"
#include "python/bindings/models/models.h"
#include "python/bindings/convert.h"
#include "python/bindings/models/config-base.h"
#include "python/bindings/models/gemma3/profile-tags.h"
#include "models/production/gemma3/net.h"
#include "models/production/gemma3/google-weights.h"
#include "models/production/gemma3/processing.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>


namespace pyvai {

namespace {

// ModelConfig::get() by name: the presets table holds plain functions.
const Presets<gemma3::ModelConfig>& gemma3Presets()
{
    static const Presets<gemma3::ModelConfig> presets = {
        { "270M", +[] { return gemma3::ModelConfig::get("270M"); } },
        { "1B", +[] { return gemma3::ModelConfig::get("1B"); } },
        { "4B", +[] { return gemma3::ModelConfig::get("4B"); } },
        { "12B", +[] { return gemma3::ModelConfig::get("12B"); } },
    };
    return presets;
}


// What the net's constructor divides by, sizes buffers with, or asserts.
void checkConfig(const gemma3::ModelConfig& c, uint32_t prefillChunk)
{
    const char* who = "Gemma3Model";
    const gemma3::TransformerConfig& d = c.decoder;
    if (d.nBlocks == 0)
        configError(who, "config.decoder.n_blocks must be >= 1");
    if (d.embedDim == 0 || d.ffnHiddenDim == 0)
        configError(who, "config.decoder: embed_dim and ffn_hidden_dim must be >= 1");
    if (d.numQHeads == 0 || d.numKVHeads == 0)
        configError(who, "config.decoder: num_q_heads and num_kv_heads must be >= 1");
    if (d.numQHeads % d.numKVHeads != 0)
        configError(who, "config.decoder.num_kv_heads must divide num_q_heads");
    if (d.slidingWindow == 0)
        configError(who, "config.decoder.sliding_window must be >= 1");
    if (!(d.globalRopeFactor > 0.0f))
        configError(who, "config.decoder.global_rope_factor must be > 0");
    if (c.vocabSize == 0)
        configError(who, "config.vocab_size must be >= 1");
    if (c.maxTrainedContext == 0)
        configError(who, "config.max_trained_context must be >= 1: the KV cache is clamped to it");
    if (!c.vision)
        return;

    // The encoder pools 4 x 4 patches into each of its tokens, and a prefill
    // chunk holds an image whole.
    const gemma3::VisionEncoderConfig& v = *c.vision;
    checkVisionConfig(v, who, "config.vision");
    if (v.patchesPerSide() != 4 * v.tokensPerSide)
        configError(who,
                    "config.vision: image_size / patch_size must be " + std::to_string(4 * v.tokensPerSide) +
                        ": the encoder pools 4 x 4 patches into each of its " + std::to_string(v.tokensPerImage) +
                        " tokens");
    if (prefillChunk != 0 && prefillChunk < v.tokensPerImage)
        configError(who,
                    "prefill_chunk must be 0 or at least " + std::to_string(v.tokensPerImage) +
                        ", the tokens of one image");
}

}   // namespace


void bindGemma3(nb::module_& m)
{
    nb::class_<gemma3::TransformerConfig>(m,
                                          "Gemma3TransformerConfig",
                                          "The decoder stack of a Gemma 3 checkpoint (Gemma3Config.decoder). Every "
                                          "sixth block is global (the whole prefix); the others are local, a sliding "
                                          "window of sliding_window tokens.")
        .def_rw("n_blocks", &gemma3::TransformerConfig::nBlocks)
        .def_rw("embed_dim", &gemma3::TransformerConfig::embedDim)
        .def_rw("num_q_heads", &gemma3::TransformerConfig::numQHeads)
        .def_rw("num_kv_heads",
                &gemma3::TransformerConfig::numKVHeads,
                "Fewer than num_q_heads for GQA; it must divide them.")
        .def_rw("ffn_hidden_dim", &gemma3::TransformerConfig::ffnHiddenDim)
        .def_rw(
            "sliding_window", &gemma3::TransformerConfig::slidingWindow, "A local block's attention window, in tokens.")
        .def_rw("global_rope_factor",
                &gemma3::TransformerConfig::globalRopeFactor,
                "A global block's linear RoPE scale when > 1; 1 is plain RoPE.")
        .def_prop_ro_static("head_dim", [](nb::handle) { return gemma3::TransformerConfig::headDim; })
        .def_prop_ro_static("norm_eps", [](nb::handle) { return gemma3::TransformerConfig::normEps; });

    nb::class_<gemma3::VisionEncoderConfig, VisionModelConfig>(
        m,
        "Gemma3VisionEncoderConfig",
        "The vision half of a Gemma 3 VLM checkpoint (Gemma3Config.vision): SigLIP "
        "So400m/14 at 896 pixels a side (a VisionModelConfig), average-pooled to "
        "tokens_per_image soft tokens and projected to the decoder's width.")
        .def(nb::init<>(), "The checkpoints' encoder.")
        .def_rw("image_token_id",
                &gemma3::VisionEncoderConfig::imageTokenId,
                "Id of <image_soft_token>, the placeholder an image's tokens replace.")
        .def_prop_ro_static("tokens_per_side", [](nb::handle) { return gemma3::VisionEncoderConfig::tokensPerSide; })
        .def_prop_ro_static(
            "tokens_per_image",
            [](nb::handle) { return gemma3::VisionEncoderConfig::tokensPerImage; },
            "Soft tokens per image: the length of its placeholder run in a prompt.");

    auto cfgCls = nb::class_<gemma3::ModelConfig>(
        m,
        "Gemma3Config",
        "A Gemma 3 checkpoint (models/production/gemma3/net.h), built by preset(variant); "
        "the 4B and 12B VLMs carry a vision half.");
    bindPresets(cfgCls, "Gemma3Config", gemma3Presets());
    cfgCls.def_rw("decoder", &gemma3::ModelConfig::decoder, "The decoder stack (Gemma3TransformerConfig).")
        .def_rw("vocab_size", &gemma3::ModelConfig::vocabSize)
        .def_rw("max_trained_context",
                &gemma3::ModelConfig::maxTrainedContext,
                "The trained context, which caps the KV cache.")
        .def_rw("vision",
                &gemma3::ModelConfig::vision,
                "The vision half (Gemma3VisionEncoderConfig), or None for a text-only "
                "checkpoint. Edits through the property land in place.")
        .def_static("is_global_block",
                    &gemma3::group::isGlobalBlock,
                    nb::arg("block"),
                    "Whether `block` attends the whole prefix rather than the sliding window.");

    auto cls = nb::class_<gemma3::Net, vai::NeuralNet>(m, "Gemma3Model")
                   .def(
                       "__init__",
                       [](gemma3::Net* self,
                          const gemma3::ModelConfig& config,
                          vai::Dtype w_default_dtype,
                          vai::Dtype o_default_dtype,
                          std::optional<eva::Device>
                              device,
                          std::optional<uint32_t>
                              max_runtime_context,
                          vai::Dtype kv_cache_dtype,
                          uint32_t prefill_chunk) {
                           checkConfig(config, prefill_chunk);
                           const uint32_t context =
                               runtimeContext<gemma3::RuntimeConfig>(max_runtime_context, "Gemma3Model");
                           const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                           new (self) gemma3::Net(cfg,
                                                  config,
                                                  {
                                                      .kvCacheDtype = kv_cache_dtype,
                                                      .maxRuntimeContext = context,
                                                      .prefillChunk = prefill_chunk,
                                                  });
                       },
                       nb::arg("config") = gemma3::ModelConfig::get("1B"),
                       nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                       nb::arg("o_default_dtype") = vai::Dtype::Fp32,
                       nb::arg("device") = nb::none(),
                       nb::arg("max_runtime_context") = nb::none(),
                       nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                       nb::arg("prefill_chunk") = gemma3::RuntimeConfig{}.prefillChunk,
                       "Build a Gemma 3 model (default config: Gemma3Config.preset('1B')). "
                       "`max_runtime_context` bounds the KV cache of every block, local ones "
                       "included, clamped to the checkpoint's max_trained_context (None: all of "
                       "it). prefill() runs the prompt `prefill_chunk` tokens at a time (0: in "
                       "one forward); a chunk never splits an image.");
    bindWeightsAndTags<gemma3::Net>(cls,
                                    "Gemma3Model",
                                    gemma3::google::loadWeights,
                                    "Load the upstream HuggingFace Gemma 3 checkpoint (google/gemma-3-1b-it, "
                                    "-4b-it, ...) from its folder (data/gemma3/ckpts/<variant>): every "
                                    "*.safetensors shard in it is read.",
                                    examples::setProfileTags);

    cls.def(
        "prefill",
        [](nb::handle_t<gemma3::Net> self, nb::handle images, nb::handle_t<vai::Tensor> prompt_token_ids)
            -> nb::object {
            auto& model = nb::cast<gemma3::Net&>(self);
            const char* who = "Gemma3Model.prefill";
            PrefillArgs a = prefillArgs(self, prompt_token_ids, who);
            vai::Tensor batch;   // empty: text only
            if (!images.is_none())
            {
                if (!nb::isinstance<vai::Tensor>(images))
                    throw nb::type_error("Gemma3Model.prefill: images must be a Tensor or None");
                if (!model.modelCfg.vision)
                    throw nb::value_error("Gemma3Model.prefill: this checkpoint has no vision encoder "
                                          "(config.vision is None)");
                batch = stageInput(self, "inImages", images, who);
                // The model asserts the shape.
                const auto& v = *model.modelCfg.vision;
                const auto& shape = batch.shape();
                const size_t r = shape.size();
                if ((r != 3 && r != 4) || shape[r - 3] != v.imageChannels
                    || shape[r - 2] != v.imageSize || shape[r - 1] != v.imageSize)
                    throw nb::value_error(("Gemma3Model.prefill: images must be [B, "
                        + std::to_string(v.imageChannels) + ", "
                        + std::to_string(v.imageSize) + ", "
                        + std::to_string(v.imageSize) + "], or one image without the B").c_str());
            }
            {
                nb::gil_scoped_release rel;   // waits the previous forward
                model.prefill(batch, a.tokens, a.token);
            }
            return a.out;
        },
        nb::arg("images").none(),
        nb::arg("prompt_token_ids"),
        "Run the prompt ([T] uint32 ids, Tensor.from_u32) on top of the KV cache and "
        "return the next token: a [1] Tensor to read with to_u32() and hand to decode(), "
        "the same one every time. Appends T tokens; nothing checks the budget.\n\n"
        "`images` is None for text only; a VLM takes [B, 3, S, S] floats in [-1, 1] (or "
        "[3, S, S] for one image). The prompt must hold one run of "
        "vision.tokens_per_image image_token_id placeholders per image, in order "
        "(gemma3_expand_image_placeholders writes them); RuntimeError on a mismatch.");
    bindDecodeLoop<gemma3::Net>(cls, "Gemma3Model");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inImages", "inTokenIds"}, {"nextTokenId"});

    // models/production/gemma3/processing.h: what the demo and the chat build a prompt with.
    m.def("gemma3_expand_image_placeholders",
          [](const std::string& prompt, uint32_t tokens_per_image) {
              return gemma3::expandImagePlaceholders(prompt, tokens_per_image);
          },
          nb::arg("prompt"), nb::arg("tokens_per_image"),
          "Replace every <start_of_image> in `prompt` with the HuggingFace processor's "
          "block for one image: two newlines, <start_of_image>, `tokens_per_image` "
          "<image_soft_token> placeholders, <end_of_image>, two newlines.");

    m.def("gemma3_take_image_tags",
          [](const std::string& message) {
              std::vector<std::string> paths;
              std::string text = gemma3::takeImageTags(message, paths);
              return nb::make_tuple(text, paths);
          },
          nb::arg("message"),
          "Turn every <image:file> tag in `message` into <start_of_image>. Returns "
          "(message, files): the files in the order their tags stood.");

    m.def("gemma3_load_images",
          [](const std::vector<std::string>& paths, uint32_t image_size)
          {
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
