#ifndef PYVAI_MODELS_CONFIG_BASE_H
#define PYVAI_MODELS_CONFIG_BASE_H

// Shared by the model bindings: preset lookup by checkpoint variant, and config
// checks that raise ValueError where the core aborts or builds a wrong shape.

#include "python/bindings/common.h"
#include "python/bindings/nodes/nodes.h"   // checkRopeConfig
#include "models/common.h"
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

// `path` names the config in the message, e.g. "config.trf".
inline void checkTransformerConfig(const vai::TransformerConfig& c, const std::string& who, const std::string& path)
{
    if (c.nBlocks == 0)
        configError(who, path + ".n_blocks must be >= 1");
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

// A language model's decoder, token table and trained context, which the model
// clamps its KV cache to, so 0 would leave it no cache at all.
template <typename Config>
void checkLanguageConfig(const Config& c, const std::string& who)
{
    if constexpr (requires { c.trf; })
        checkTransformerConfig(c.trf, who, "config.trf");
    else
        checkTransformerConfig(c.languageTrf, who, "config.language_trf");
    if (c.vocabSize == 0)
        configError(who, "config.vocab_size must be >= 1");
    if constexpr (requires { c.maxTrainedContext; })
        if (c.maxTrainedContext == 0)
            configError(who, "config.max_trained_context must be >= 1: the KV cache is clamped to it");
}

// The towers here carry a position table of num_patches rows, so the image
// size is fixed.
inline void checkVisionConfig(const VisionTransformerConfig& v, const std::string& who, const std::string& path)
{
    if (v.imageChannels == 0)
        configError(who, path + ".image_channels must be >= 1");
    if (v.patchSize == 0 || v.imageSize < v.patchSize)
        configError(who, path + ".image_size must hold at least one patch_size patch");
    checkTransformerConfig(v, who, path);
}

}   // namespace pyvai

#endif // PYVAI_MODELS_CONFIG_BASE_H
