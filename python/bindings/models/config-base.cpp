#include "python/bindings/models/models.h"
#include "python/bindings/models/config-base.h"
#include <cstdint>
#include <optional>
#include <string>


namespace pyvai {

// The parts of every family's config: a transformer stack (vai/groups/transformer.h)
// and models/common.h's vision base.
void bindModelConfigs(nb::module_& m)
{
    nb::class_<vai::TransformerConfig>(
        m,
        "TransformerConfig",
        "A transformer stack as a checkpoint states it. Run-time choices (K/V "
        "dtype, cache depth, a batch axis) are the model's, set from its arguments.")
        .def(
            "__init__",
            [](vai::TransformerConfig* self,
               uint32_t n_blocks,
               uint32_t embed_dim,
               uint32_t head_dim,
               uint32_t num_q_heads,
               uint32_t num_kv_heads,
               uint32_t ffn_hidden_dim,
               vai::ActType ffn_act_type,
               float norm_eps,
               bool with_qk_norm,
               std::optional<vai::RopeConfig>
                   rope_cfg) {
                new (self) vai::TransformerConfig{
                    .nBlocks = n_blocks,
                    .embedDim = embed_dim,
                    .headDim = head_dim,
                    .numQHeads = num_q_heads,
                    .numKVHeads = num_kv_heads,
                    .ffnHiddenDim = ffn_hidden_dim,
                    .ffnActType = ffn_act_type,
                    .normEps = norm_eps,
                    .withQkNorm = with_qk_norm,
                    .ropeCfg = rope_cfg,
                };
            },
            nb::kw_only(),
            nb::arg("n_blocks"),
            nb::arg("embed_dim"),
            nb::arg("head_dim"),
            nb::arg("num_q_heads"),
            nb::arg("num_kv_heads"),
            nb::arg("ffn_hidden_dim"),
            nb::arg("ffn_act_type"),
            nb::arg("norm_eps") = 1e-5f,
            nb::arg("with_qk_norm") = false,
            nb::arg("rope_cfg") = nb::none(),
            "Keyword-only, the defaults the C++ struct has.")
        .def_rw("n_blocks", &vai::TransformerConfig::nBlocks)
        .def_rw("embed_dim", &vai::TransformerConfig::embedDim)
        .def_rw("head_dim", &vai::TransformerConfig::headDim)
        .def_rw("num_q_heads", &vai::TransformerConfig::numQHeads)
        .def_rw(
            "num_kv_heads", &vai::TransformerConfig::numKVHeads, "Fewer than num_q_heads for GQA; it must divide them.")
        .def_rw("ffn_hidden_dim", &vai::TransformerConfig::ffnHiddenDim)
        .def_rw("ffn_act_type",
                &vai::TransformerConfig::ffnActType,
                "The FFN's activation (its GLU variant when the FFN is gated).")
        .def_rw(
            "norm_eps", &vai::TransformerConfig::normEps, "Epsilon of every norm in the stack, the QK norm included.")
        .def_rw("with_qk_norm", &vai::TransformerConfig::withQkNorm, "Per-head RMS norm on Q and K, before RoPE.")
        .def_rw(
            "rope_cfg",
            &vai::TransformerConfig::ropeCfg,
            "RoPE frequencies (RopeConfig), or None for no position encoding. "
            "Edits through the property land in place.");

    nb::class_<VisionTransformerConfig, vai::TransformerConfig>(
        m,
        "VisionTransformerConfig",
        "A vision transformer: the encoder stack (a TransformerConfig) and the patch "
        "grid an image becomes. The projector to language tokens is not part of it.")
        .def(
            "__init__",
            [](VisionTransformerConfig* self,
               const vai::TransformerConfig& base,
               uint32_t image_size,
               uint32_t image_channels,
               uint32_t patch_size) {
                new (self) VisionTransformerConfig{
                    base,
                    image_size,
                    image_channels,
                    patch_size,
                };
            },
            nb::kw_only(),
            nb::arg("base"),
            nb::arg("image_size") = 0u,
            nb::arg("image_channels"),
            nb::arg("patch_size"))
        .def_rw("image_size",
                &VisionTransformerConfig::imageSize,
                "Square input side; the pixels past the last whole patch drop. 0 if the weights pin none.")
        .def_rw("image_channels", &VisionTransformerConfig::imageChannels)
        .def_rw("patch_size", &VisionTransformerConfig::patchSize)
        .def_prop_ro(
            "patches_per_side",
            [](const VisionTransformerConfig& self) {
                if (self.patchSize == 0)
                    throw nb::value_error("VisionTransformerConfig.patches_per_side: patch_size is 0");
                return self.patchesPerSide();
            },
            "Patches along one side of the image: image_size / patch_size.")
        .def_prop_ro(
            "num_patches",
            [](const VisionTransformerConfig& self) {
                if (self.patchSize == 0)
                    throw nb::value_error("VisionTransformerConfig.num_patches: patch_size is 0");
                return self.numPatches();
            },
            "Tokens one image becomes: patches_per_side squared.");
}

}   // namespace pyvai
