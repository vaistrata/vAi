#include "python/bindings/models/models.h"
#include "python/bindings/models/lm.h"
#include "python/bindings/models/config-base.h"
#include "python/bindings/models/llama3/profile-tags.h"
#include "models/production/llama3/net.h"
#include "models/production/llama3/meta-weights.h"
#include <string>
#include <utility>


namespace pyvai {

namespace {

const Presets<llama3::ModelConfig>& llama3Presets()
{
    static const Presets<llama3::ModelConfig> presets = {
        { "3.2-1B", &llama3::ModelConfig::_3_2_1B },
        { "3.2-3B", &llama3::ModelConfig::_3_2_3B },
    };
    return presets;
}

}   // namespace


void bindLlama3(nb::module_& m)
{
    auto cfgCls = nb::class_<llama3::ModelConfig>(
        m, "Llama3Config", "A Llama 3 checkpoint (models/production/llama3/net.h), built by preset(variant).");
    bindPresets(cfgCls, "Llama3Config", llama3Presets());
    cfgCls.def_rw("language_cfg", &llama3::ModelConfig::languageCfg,
                  "The decoder stack and its token table (LanguageModelConfig); every "
                  "layer carries the Llama 3 RoPE scaling.");

    auto cls = nb::class_<llama3::Net, vai::NeuralNet>(m, "Llama3Model")
                   .def(
                       "__init__",
                       [](llama3::Net* self,
                          const llama3::ModelConfig& config,
                          vai::Dtype w_default_dtype,
                          vai::Dtype o_default_dtype,
                          std::optional<eva::Device>
                              device,
                          std::optional<uint32_t>
                              max_runtime_context,
                          uint32_t prefill_chunk,
                          vai::Dtype kv_cache_dtype) {
                           checkLanguageConfig(config.languageCfg,
                                               "Llama3Model",
                                               "config.language_cfg",
                                               /*clampsContext*/ true);
                           const uint32_t context =
                               runtimeContext<llama3::RuntimeConfig>(max_runtime_context, "Llama3Model");
                           const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                           new (self) llama3::Net(cfg,
                                                  config,
                                                  {
                                                      .kvCacheDtype = kv_cache_dtype,
                                                      .maxRuntimeContext = context,
                                                      .prefillChunk = prefill_chunk,
                                                  });
                       },
                       nb::arg("config") = llama3::ModelConfig::_3_2_1B(),
                       nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                       nb::arg("o_default_dtype") = vai::Dtype::Fp32,
                       nb::arg("device") = nb::none(),
                       nb::arg("max_runtime_context") = nb::none(),
                       nb::arg("prefill_chunk") = 0u,
                       nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                       "Build a Llama 3 model (default config: Llama3Config.preset('3.2-1B')). "
                       "`max_runtime_context` bounds the KV cache, clamped to the checkpoint's "
                       "max_trained_context (None: all of it, 128K, so set a real budget). A "
                       "prompt longer than `prefill_chunk` runs as several forwards (0: as one).");
    bindWeightsAndTags<llama3::Net>(cls,
                                    "Llama3Model",
                                    llama3::meta::loadWeights,
                                    "Load the upstream HuggingFace Llama 3 checkpoint from its folder "
                                    "(data/llama3/ckpts/<variant>): every *.safetensors in it is read.",
                                    examples::setProfileTags);

    cls.def(
        "prefill",
        [](nb::handle_t<llama3::Net> self, nb::handle_t<vai::Tensor> prompt_token_ids) -> nb::object {
            auto& model = nb::cast<llama3::Net&>(self);
            PrefillArgs a = prefillArgs(self, prompt_token_ids, "Llama3Model.prefill");
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
    bindDecodeLoop<llama3::Net>(cls, "Llama3Model");
    cls.def_prop_ro(
        "prefill_chunk",
        [](llama3::Net& self) { return self.runtimeCfg.prefillChunk; },
        "Prefill chunk length; 0 runs the whole prompt in one forward.");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inTokenIds"}, {"nextTokenId"});
}

}   // namespace pyvai
