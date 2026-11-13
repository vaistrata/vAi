#include "python/bindings/models/models.h"
#include "python/bindings/models/lm.h"
#include "python/bindings/models/config-base.h"
#include "python/bindings/models/gpt2/profile-tags.h"
#include "models/production/gpt2/net.h"
#include "models/production/gpt2/openai-weights.h"
#include <string>
#include <utility>


namespace pyvai {

namespace {

const Presets<gpt2::ModelConfig>& gpt2Presets()
{
    static const Presets<gpt2::ModelConfig> presets = {
        { "124M", &gpt2::ModelConfig::_124M },
        { "355M", &gpt2::ModelConfig::_355M },
        { "774M", &gpt2::ModelConfig::_774M },
        { "1.5B", &gpt2::ModelConfig::_1_5B },
    };
    return presets;
}

}   // namespace


void bindGPT2(nb::module_& m)
{
    auto cfgCls = nb::class_<gpt2::ModelConfig>(
        m, "GPT2Config", "A GPT-2 checkpoint (models/production/gpt2/net.h), built by preset(variant).");
    bindPresets(cfgCls, "GPT2Config", gpt2Presets());
    cfgCls.def_rw("language_cfg", &gpt2::ModelConfig::languageCfg,
                  "The decoder stack and its token table (LanguageModelConfig); the "
                  "trained context is the positional table's rows, a hard limit.");

    auto cls = nb::class_<gpt2::Net, vai::NeuralNet>(m, "GPT2Model")
                   .def(
                       "__init__",
                       [](gpt2::Net* self,
                          const gpt2::ModelConfig& config,
                          vai::Dtype w_default_dtype,
                          vai::Dtype o_default_dtype,
                          std::optional<eva::Device>
                              device,
                          std::optional<uint32_t>
                              max_runtime_context,
                          vai::Dtype kv_cache_dtype) {
                           checkLanguageConfig(config.languageCfg,
                                               "GPT2Model",
                                               "config.language_cfg",
                                               /*clampsContext*/ true);
                           const uint32_t context =
                               runtimeContext<gpt2::RuntimeConfig>(max_runtime_context, "GPT2Model");
                           const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                           new (self) gpt2::Net(cfg,
                                                config,
                                                {
                                                    .kvCacheDtype = kv_cache_dtype,
                                                    .maxRuntimeContext = context,
                                                });
                       },
                       nb::arg("config") = gpt2::ModelConfig::_124M(),
                       nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                       nb::arg("o_default_dtype") = vai::Dtype::Fp32,
                       nb::arg("device") = nb::none(),
                       nb::arg("max_runtime_context") = nb::none(),
                       nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                       "Build a GPT-2 model (default config: GPT2Config.preset('124M')). "
                       "`max_runtime_context` bounds the KV cache, clamped to the checkpoint's "
                       "max_trained_context (None: all of it).");
    bindWeightsAndTags<gpt2::Net>(cls,
                                  "GPT2Model",
                                  gpt2::openai::loadWeights,
                                  "Load the upstream HuggingFace GPT-2 checkpoint from its folder "
                                  "(data/gpt2/ckpts/<variant>): every *.safetensors in it is read.",
                                  examples::setProfileTags);

    cls.def(
        "prefill",
        [](nb::handle_t<gpt2::Net> self, nb::handle_t<vai::Tensor> prompt_token_ids) -> nb::object {
            auto& model = nb::cast<gpt2::Net&>(self);
            PrefillArgs a = prefillArgs(self, prompt_token_ids, "GPT2Model.prefill");
            {
                nb::gil_scoped_release rel;   // waits the previous forward
                model.prefill(a.tokens, a.token);
            }
            return a.out;
        },
        nb::arg("prompt_token_ids"),
        "Run the prompt ([T] uint32 ids, Tensor.from_u32) on top of the KV cache and "
        "return the next token: a [1] Tensor to read with to_u32() and hand to decode(), "
        "the same one every time. Appends T tokens; nothing checks the budget.");
    bindDecodeLoop<gpt2::Net>(cls, "GPT2Model");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inTokenIds"}, {"nextTokenId"});
}

}   // namespace pyvai
