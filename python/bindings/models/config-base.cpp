#include "python/bindings/models/models.h"
#include "python/bindings/models/config-base.h"
#include <cstdint>
#include <optional>
#include <string>


namespace pyvai {

// The parts of every family's config: a transformer layer and its stack
// (vai/groups/transformer.h), and models/config-base.h's vision and language bases.
void bindModelConfigs(nb::module_& m)
{
    nb::class_<vai::TransformerLayerConfig>(
        m,
        "TransformerLayerConfig",
        "One transformer layer as a checkpoint states it. Run-time choices (K/V "
        "dtype, cache depth, a batch axis) are the model's, set from its arguments.")
        .def(
            "__init__",
            [](vai::TransformerLayerConfig* self,
               uint32_t embed_dim,
               uint32_t head_dim,
               uint32_t num_q_heads,
               uint32_t num_kv_heads,
               uint32_t ffn_hidden_dim,
               vai::Epilogue ffn_activation,
               bool with_qk_norm,
               std::optional<vai::RopeConfig>
                   rope_cfg,
               float norm_eps) {
                new (self) vai::TransformerLayerConfig{
                    .embedDim = embed_dim,
                    .headDim = head_dim,
                    .numQHeads = num_q_heads,
                    .numKVHeads = num_kv_heads,
                    .kvCacheDtype = vai::Dtype::Fp16,   // the model sets its own
                    .withQkNorm = with_qk_norm,
                    .ropeCfg = rope_cfg,
                    .ffnHiddenDim = ffn_hidden_dim,
                    .ffnActivation = ffn_activation,
                    .normEps = norm_eps,
                };
            },
            nb::kw_only(),
            nb::arg("embed_dim"),
            nb::arg("head_dim"),
            nb::arg("num_q_heads"),
            nb::arg("num_kv_heads"),
            nb::arg("ffn_hidden_dim"),
            nb::arg("ffn_activation"),
            nb::arg("with_qk_norm") = false,
            nb::arg("rope_cfg") = nb::none(),
            nb::arg("norm_eps") = 1e-5f,
            "Keyword-only, the defaults the C++ struct has.")
        .def_rw("embed_dim", &vai::TransformerLayerConfig::embedDim)
        .def_rw("head_dim", &vai::TransformerLayerConfig::headDim)
        .def_rw("num_q_heads", &vai::TransformerLayerConfig::numQHeads)
        .def_rw("num_kv_heads",
                &vai::TransformerLayerConfig::numKVHeads,
                "Fewer than num_q_heads for GQA; it must divide them.")
        .def_rw("with_qk_norm", &vai::TransformerLayerConfig::withQkNorm, "Per-head RMS norm on Q and K, before RoPE.")
        .def_rw("rope_cfg",
                &vai::TransformerLayerConfig::ropeCfg,
                "RoPE frequencies (RopeConfig), or None for no position encoding. "
                "Edits through the property land in place.")
        .def_rw("ffn_hidden_dim", &vai::TransformerLayerConfig::ffnHiddenDim)
        .def_rw("ffn_activation",
                &vai::TransformerLayerConfig::ffnActivation,
                "The FFN's activation (its GLU variant when the FFN is gated).")
        .def_rw("norm_eps",
                &vai::TransformerLayerConfig::normEps,
                "Epsilon of every norm the layer owns, the QK norm included.");

    nb::class_<vai::TransformerConfig>(
        m, "TransformerConfig", "A stack of num_layers layers, each built from layer_cfg.")
        .def(
            "__init__",
            [](vai::TransformerConfig* self, uint32_t num_layers, const vai::TransformerLayerConfig& layer_cfg) {
                new (self) vai::TransformerConfig{
                    .numLayers = num_layers,
                    .layerCfg = layer_cfg,
                };
            },
            nb::kw_only(),
            nb::arg("num_layers"),
            nb::arg("layer_cfg"))
        .def_rw("num_layers", &vai::TransformerConfig::numLayers)
        .def_rw("layer_cfg", &vai::TransformerConfig::layerCfg, "Every layer's config (TransformerLayerConfig).");

    nb::class_<VisionModelConfig, vai::TransformerConfig>(
        m,
        "VisionModelConfig",
        "A vision transformer: the encoder stack (a TransformerConfig) and the patch "
        "grid an image becomes. The projector to language tokens is not part of it.")
        .def(
            "__init__",
            [](VisionModelConfig* self,
               uint32_t num_layers,
               const vai::TransformerLayerConfig& layer_cfg,
               uint32_t image_size,
               uint32_t image_channels,
               uint32_t patch_size) {
                new (self) VisionModelConfig{
                    vai::TransformerConfig{
                        .numLayers = num_layers,
                        .layerCfg = layer_cfg,
                    },
                    image_size,
                    image_channels,
                    patch_size,
                };
            },
            nb::kw_only(),
            nb::arg("num_layers"),
            nb::arg("layer_cfg"),
            nb::arg("image_size") = 0u,
            nb::arg("image_channels"),
            nb::arg("patch_size"))
        .def_rw("image_size",
                &VisionModelConfig::imageSize,
                "Square input side, a multiple of patch_size; 0 if the weights pin none.")
        .def_rw("image_channels", &VisionModelConfig::imageChannels)
        .def_rw("patch_size", &VisionModelConfig::patchSize)
        .def_prop_ro(
            "patches_per_side",
            [](const VisionModelConfig& self) {
                if (self.patchSize == 0)
                    throw nb::value_error("VisionModelConfig.patches_per_side: patch_size is 0");
                return self.patchesPerSide();
            },
            "Patches along one side of the image: image_size / patch_size.")
        .def_prop_ro(
            "num_patches",
            [](const VisionModelConfig& self) {
                if (self.patchSize == 0)
                    throw nb::value_error("VisionModelConfig.num_patches: patch_size is 0");
                return self.numPatches();
            },
            "Tokens one image becomes: patches_per_side squared.");

    nb::class_<LanguageModelConfig, vai::TransformerConfig>(
        m,
        "LanguageModelConfig",
        "A language model: the decoder stack (a TransformerConfig), the token table "
        "it reads and the context it was trained for. The LM head is not part of it.")
        .def(
            "__init__",
            [](LanguageModelConfig* self,
               uint32_t num_layers,
               const vai::TransformerLayerConfig& layer_cfg,
               uint32_t vocab_size,
               uint32_t max_trained_context) {
                new (self) LanguageModelConfig{
                    vai::TransformerConfig{
                        .numLayers = num_layers,
                        .layerCfg = layer_cfg,
                    },
                    vocab_size,
                    max_trained_context,
                };
            },
            nb::kw_only(),
            nb::arg("num_layers"),
            nb::arg("layer_cfg"),
            nb::arg("vocab_size"),
            nb::arg("max_trained_context") = 0u)
        .def_rw("vocab_size", &LanguageModelConfig::vocabSize)
        .def_rw("max_trained_context",
                &LanguageModelConfig::maxTrainedContext,
                "The trained context, which caps a model's KV cache; 0: no cap (pi0).");
}

}   // namespace pyvai
