#ifndef PYVAI_MODELS_CONFIG_BASE_H
#define PYVAI_MODELS_CONFIG_BASE_H

// Shared by the model bindings: preset lookup by checkpoint variant, and config
// checks that raise ValueError where the core aborts or builds a wrong shape.

#include "python/bindings/common.h"
#include "python/bindings/nodes/nodes.h"   // checkRopeConfig
#include "models/config-base.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pyvai {

// A family's C++ ModelConfig factories keyed by variant, in the order `presets` lists.
template <typename Config>
using Presets = std::vector<std::pair<std::string, Config (*)()>>;

// preset(variant) and presets on a family's config class. `presets` must
// outlive the module (a function-local static).
template <typename Config, typename Cls>
void bindPresetLookup(Cls& cls, const char* pyName, const Presets<Config>& presets)
{
    cls.def_static(
            "preset",
            [pyName, &presets](const std::string& variant) -> Config {
                std::string known;
                for (const auto& [name, make] : presets)
                {
                    if (name == variant)
                        return make();
                    known += (known.empty() ? "'" : ", '") + name + "'";
                }
                throw nb::value_error((std::string(pyName) + ".preset: unknown variant '"
                                       + variant + "' (one of " + known + ")").c_str());
            },
            nb::arg("variant"),
            "A fresh config for one of `presets`; ValueError for any other name.")
        .def_prop_ro_static(
            "presets",
            [&presets](nb::handle /*cls*/) {
                std::vector<std::string> names;
                for (const auto& [name, make] : presets)
                    names.push_back(name);
                return names;
            },
            "The variants preset() accepts, smallest first.");
}

// ... plus the read-only `variant` the language model families carry.
template <typename Config, typename Cls>
void bindPresets(Cls& cls, const char* pyName, const Presets<Config>& presets)
{
    bindPresetLookup(cls, pyName, presets);
    cls.def_prop_ro(
        "variant",
        [](const Config& self) -> std::string { return self.variant; },
        "The variant from preset(), also its folder under data/<family>/ckpts/.");
}

[[noreturn]] inline void configError(const std::string& who, const std::string& msg)
{
    throw nb::value_error((who + ": " + msg).c_str());
}

// `path` names the config in the message, e.g. "config.language_cfg".
inline void checkLayerConfig(const vai::TransformerLayerConfig& c, const std::string& who,
                             const std::string& path)
{
    if (c.embedDim == 0 || c.headDim == 0 || c.ffnHiddenDim == 0)
        configError(who, path + ": embed_dim, head_dim and ffn_hidden_dim must be >= 1");
    if (c.numQHeads == 0 || c.numKVHeads == 0)
        configError(who, path + ": num_q_heads and num_kv_heads must be >= 1");
    if (c.numQHeads % c.numKVHeads != 0)
        configError(who, path + ".num_kv_heads must divide num_q_heads");
    if (c.ropeCfg)
    {
        if (c.headDim % 2 != 0)
            configError(who, path + ".head_dim must be even: RoPE rotates pairs");
        checkRopeConfig(*c.ropeCfg, who.c_str(), (path + ".rope_cfg").c_str());
    }
}

inline void checkTransformerConfig(const vai::TransformerConfig& t, const std::string& who,
                                   const std::string& path)
{
    if (t.numLayers == 0)
        configError(who, path + ".num_layers must be >= 1");
    checkLayerConfig(t.layerCfg, who, path + ".layer_cfg");
}

// `clampsContext`: the model sizes its KV cache by max_trained_context, so 0
// would leave it no cache at all.
inline void checkLanguageConfig(const LanguageModelConfig& l, const std::string& who,
                                const std::string& path, bool clampsContext)
{
    if (l.vocabSize == 0)
        configError(who, path + ".vocab_size must be >= 1");
    if (clampsContext && l.maxTrainedContext == 0)
        configError(who, path + ".max_trained_context must be >= 1: the KV cache is clamped to it");
    checkTransformerConfig(l, who, path);
}

// The towers here carry a position table of num_patches rows, so the image
// size is fixed.
inline void checkVisionConfig(const VisionModelConfig& v, const std::string& who,
                              const std::string& path)
{
    if (v.imageChannels == 0)
        configError(who, path + ".image_channels must be >= 1");
    if (v.patchSize == 0 || v.imageSize == 0 || v.imageSize % v.patchSize != 0)
        configError(who, path + ".image_size must be a positive multiple of patch_size");
    checkTransformerConfig(v, who, path);
}

}   // namespace pyvai

#endif // PYVAI_MODELS_CONFIG_BASE_H
