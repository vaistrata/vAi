#include "models.h"
#include "config-base.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>


namespace pyvai {

namespace {

// A Python per_layer as the C++ TransformerPerLayer. nanobind's std::function
// caster would hand the callee a copy of the layer config and lose its edits,
// so this passes it by reference. A std::function is copied and destroyed
// anywhere, so the reference count moves under the GIL.
struct PyPerLayer
{
    PyObject* fn;

    explicit PyPerLayer(nb::handle f)
    : fn(f.ptr())
    {
        Py_INCREF(fn);
    }

    PyPerLayer(const PyPerLayer& o)
    : fn(o.fn)
    {
        nb::gil_scoped_acquire acq;
        Py_INCREF(fn);
    }

    PyPerLayer(PyPerLayer&& o) noexcept : fn(std::exchange(o.fn, nullptr)) {}
    PyPerLayer& operator=(const PyPerLayer&) = delete;
    PyPerLayer& operator=(PyPerLayer&&) = delete;
    ~PyPerLayer()
    {
        if (!fn) return;
        nb::gil_scoped_acquire acq;
        Py_DECREF(fn);
    }

    void operator()(uint32_t layer, vai::TransformerLayerConfig& cfg) const
    {
        nb::gil_scoped_acquire acq;
        const nb::handle callee(fn);
        callee(layer, nb::cast(&cfg, nb::rv_policy::reference));
    }
};


vai::TransformerPerLayer toPerLayer(nb::handle fn)
{
    if (fn.is_none())
        return nullptr;
    if (!PyCallable_Check(fn.ptr()))
        throw nb::type_error(
            "TransformerConfig: per_layer must be a callable (layer, cfg) -> None, or None");
    return PyPerLayer(fn);
}


// The Python callable a per_layer came from, or a wrapper around a C++ one.
nb::object fromPerLayer(const vai::TransformerPerLayer& f)
{
    if (!f)
        return nb::none();
    if (const auto* py = f.target<PyPerLayer>())
        return nb::borrow(py->fn);
    return nb::cpp_function(
        [f](uint32_t layer, vai::TransformerLayerConfig& cfg) { f(layer, cfg); },
        nb::arg("layer"), nb::arg("cfg"));
}

}   // namespace


// The parts of every family's config: a transformer layer and its stack
// (vai/groups/transformer.h), and models/config-base.h's vision and language bases.
void bindModelConfigs(nb::module_& m)
{
    nb::class_<vai::TransformerLayerConfig>(
        m, "TransformerLayerConfig",
        "One transformer layer as a checkpoint states it. Run-time choices (K/V "
        "dtype, cache depth, a batch axis) are the model's, set from its arguments.")
        .def(
            "__init__",
            [](vai::TransformerLayerConfig* self, uint32_t embed_dim, uint32_t head_dim, uint32_t num_q_heads,
               uint32_t num_kv_heads, uint32_t ffn_hidden_dim, vai::Epilogue ffn_activation, bool kv_cache_ring,
               bool with_qk_norm, std::optional<vai::RopeConfig> rope_cfg, float norm_eps) {
                new (self) vai::TransformerLayerConfig{
                    .embedDim = embed_dim,
                    .headDim = head_dim,
                    .numQHeads = num_q_heads,
                    .numKVHeads = num_kv_heads,
                    .kvCacheDtype = vai::Dtype::Fp16,   // the model sets its own
                    .kvCacheRing = kv_cache_ring,
                    .withQkNorm = with_qk_norm,
                    .ropeCfg = rope_cfg,
                    .ffnHiddenDim = ffn_hidden_dim,
                    .ffnActivation = ffn_activation,
                    .normEps = norm_eps,
                };
            },
            nb::kw_only(), nb::arg("embed_dim"), nb::arg("head_dim"), nb::arg("num_q_heads"), nb::arg("num_kv_heads"),
            nb::arg("ffn_hidden_dim"), nb::arg("ffn_activation"), nb::arg("kv_cache_ring") = false,
            nb::arg("with_qk_norm") = false, nb::arg("rope_cfg") = nb::none(), nb::arg("norm_eps") = 1e-5f,
            "Keyword-only, the defaults the C++ struct has.")
        .def_rw("embed_dim", &vai::TransformerLayerConfig::embedDim)
        .def_rw("head_dim", &vai::TransformerLayerConfig::headDim)
        .def_rw("num_q_heads", &vai::TransformerLayerConfig::numQHeads)
        .def_rw("num_kv_heads", &vai::TransformerLayerConfig::numKVHeads,
                "Fewer than num_q_heads for GQA; it must divide them.")
        .def_rw("kv_cache_ring", &vai::TransformerLayerConfig::kvCacheRing,
                "The layer's cache keeps only the last positions its sliding window reads; "
                "the model sets the window and the ring's size.")
        .def_rw("with_qk_norm", &vai::TransformerLayerConfig::withQkNorm, "Per-head RMS norm on Q and K, before RoPE.")
        .def_rw("rope_cfg", &vai::TransformerLayerConfig::ropeCfg,
                "RoPE frequencies (RopeConfig), or None for no position encoding. "
                "Edits through the property land in place.")
        .def_rw("ffn_hidden_dim", &vai::TransformerLayerConfig::ffnHiddenDim)
        .def_rw("ffn_activation", &vai::TransformerLayerConfig::ffnActivation,
                "The FFN's activation (its GLU variant when the FFN is gated).")
        .def_rw("norm_eps", &vai::TransformerLayerConfig::normEps,
                "Epsilon of every norm the layer owns, the QK norm included.");

    nb::class_<vai::TransformerConfig>(m, "TransformerConfig",
        "A stack of num_layers layers built from layer_cfg. per_layer(layer, cfg), "
        "when set, edits cfg in place before each layer is built, and an edit "
        "stays for the layers after it; layer_cfg_at(i) shows the result.")
        .def(
            "__init__",
            [](vai::TransformerConfig* self, uint32_t num_layers,
               const vai::TransformerLayerConfig& layer_cfg, nb::handle per_layer)
            {
                new (self) vai::TransformerConfig{
                    .numLayers = num_layers,
                    .layerCfg  = layer_cfg,
                    .perLayer  = toPerLayer(per_layer),
                };
            },
            nb::kw_only(),
            nb::arg("num_layers"),
            nb::arg("layer_cfg"),
            nb::arg("per_layer").none() = nb::none())
        .def_rw("num_layers", &vai::TransformerConfig::numLayers)
        .def_rw("layer_cfg",  &vai::TransformerConfig::layerCfg,
                "Every layer's config before per_layer edits it (TransformerLayerConfig).")
        .def_prop_rw(
            "per_layer",
            [](const vai::TransformerConfig& self) { return fromPerLayer(self.perLayer); },
            [](vai::TransformerConfig& self, nb::handle fn) { self.perLayer = toPerLayer(fn); },
            "Callable(layer, cfg) -> None, or None. cfg is only valid during the call.")
        .def(
            "layer_cfg_at",
            [](const vai::TransformerConfig& self, uint32_t layer) {
                if (layer >= self.numLayers)
                    throw nb::index_error(("TransformerConfig.layer_cfg_at: layer "
                        + std::to_string(layer) + " of a " + std::to_string(self.numLayers)
                        + "-layer stack").c_str());
                vai::TransformerLayerConfig cfg = self.layerCfg;
                if (self.perLayer)
                    for (uint32_t i = 0; i <= layer; ++i)
                        self.perLayer(i, cfg);
                return cfg;
            },
            nb::arg("layer"),
            "A copy of the config that layer `layer` is built with.");

    nb::class_<VisionModelConfig, vai::TransformerConfig>(m, "VisionModelConfig",
        "A vision transformer: the encoder stack (a TransformerConfig) and the patch "
        "grid an image becomes. The projector to language tokens is not part of it.")
        .def(
            "__init__",
            [](VisionModelConfig* self, uint32_t num_layers,
               const vai::TransformerLayerConfig& layer_cfg, nb::handle per_layer,
               uint32_t image_size, uint32_t image_channels, uint32_t patch_size)
            {
                new (self) VisionModelConfig{
                    vai::TransformerConfig{
                        .numLayers = num_layers,
                        .layerCfg  = layer_cfg,
                        .perLayer  = toPerLayer(per_layer),
                    },
                    image_size,
                    image_channels,
                    patch_size,
                };
            },
            nb::kw_only(),
            nb::arg("num_layers"),
            nb::arg("layer_cfg"),
            nb::arg("per_layer").none() = nb::none(),
            nb::arg("image_size") = 0u,
            nb::arg("image_channels"),
            nb::arg("patch_size"))
        .def_rw("image_size",     &VisionModelConfig::imageSize,
                "Square input side, a multiple of patch_size; 0 if the weights pin none.")
        .def_rw("image_channels", &VisionModelConfig::imageChannels)
        .def_rw("patch_size",     &VisionModelConfig::patchSize)
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

    nb::class_<LanguageModelConfig, vai::TransformerConfig>(m, "LanguageModelConfig",
        "A language model: the decoder stack (a TransformerConfig), the token table "
        "it reads and the context it was trained for. The LM head is not part of it.")
        .def(
            "__init__",
            [](LanguageModelConfig* self, uint32_t num_layers,
               const vai::TransformerLayerConfig& layer_cfg, nb::handle per_layer,
               uint32_t vocab_size, uint32_t max_trained_context)
            {
                new (self) LanguageModelConfig{
                    vai::TransformerConfig{
                        .numLayers = num_layers,
                        .layerCfg  = layer_cfg,
                        .perLayer  = toPerLayer(per_layer),
                    },
                    vocab_size,
                    max_trained_context,
                };
            },
            nb::kw_only(),
            nb::arg("num_layers"),
            nb::arg("layer_cfg"),
            nb::arg("per_layer").none() = nb::none(),
            nb::arg("vocab_size"),
            nb::arg("max_trained_context") = 0u)
        .def_rw("vocab_size",          &LanguageModelConfig::vocabSize)
        .def_rw("max_trained_context", &LanguageModelConfig::maxTrainedContext,
                "The trained context, which caps a model's KV cache; 0: no cap (pi0).");
}

}   // namespace pyvai
