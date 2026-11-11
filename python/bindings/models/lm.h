#ifndef PYVAI_MODELS_LM_H
#define PYVAI_MODELS_LM_H

// What the token-by-token models (GPT-2, Llama 3, Gemma 3, PaliGemma) bind
// alike. The forward they inherit from NeuralNet would skip the KV-cache
// counter, so __call__ is refused.

#include "models.h"
#include <cstdint>
#include <optional>
#include <string>

namespace pyvai {

// max_runtime_context as RuntimeConfig::maxRuntimeContext. None keeps the
// config's default, which the model clamps to the trained context.
template <typename RuntimeConfig>
uint32_t runtimeContext(std::optional<uint32_t> maxRuntimeContext, const char* pyName)
{
    if (!maxRuntimeContext)
        return RuntimeConfig{}.maxRuntimeContext;
    if (*maxRuntimeContext == 0)
        throw nb::value_error((std::string(pyName)
            + ": max_runtime_context must be >= 1 "
              "(None: the checkpoint's max_trained_context)").c_str());
    return *maxRuntimeContext;
}

// Token ids as the models index them: [T], T >= 1.
inline void requireTokenIds(const vai::Tensor& t, const char* who)
{
    if (t.shape().size() != 1 || t.shape()[0] == 0)
        throw nb::value_error((std::string(who)
            + ": prompt_token_ids must be a [T] tensor of token ids (Tensor.from_u32), "
              "T >= 1").c_str());
}

[[noreturn]] inline void refuseCall(const char* pyName)
{
    throw nb::type_error((std::string(pyName)
        + " is driven by prefill() and decode(); it has no forward of its own to call").c_str());
}

// The staged prompt and the net's own nextTokenId, for a model's prefill binding.
struct PrefillArgs
{
    vai::Tensor tokens;
    nb::object out;         // the Python Tensor prefill returns
    vai::Tensor* token;     // ... and the C++ object inside it
};

inline PrefillArgs prefillArgs(nb::handle self, nb::handle promptTokens, const char* who)
{
    PrefillArgs a;
    a.tokens = stageInput(self, "inTokenIds", promptTokens, who);
    requireTokenIds(a.tokens, who);
    a.out = ownOutput(self, "nextTokenId");
    a.token = &nb::cast<vai::Tensor&>(a.out);
    return a;
}

// decode(), the refused __call__, reset_cache() and the properties every one
// of these models has. prefill() differs per model and stays with it.
template <typename Model, typename Cls>
void bindDecodeLoop(Cls& cls, const char* pyName)
{
    cls.def(
            "decode",
            [pyName](nb::handle_t<Model> self, nb::handle_t<vai::Tensor> token_id) -> nb::object
            {
                auto& model = nb::cast<Model&>(self);
                auto& t = nb::cast<vai::Tensor&>(token_id);
                if (t.shape().size() != 1 || t.shape()[0] != 1)
                    throw nb::value_error((std::string(pyName)
                        + ".decode: token_id must be the [1] Tensor prefill() returned").c_str());
                {
                    nb::gil_scoped_release rel;   // waits the previous forward
                    model.decode(&t);
                }
                token_id.attr("_pyvai_net") = self;   // it views the net's buffer now
                return nb::borrow(token_id);
            },
            nb::arg("token_id"),
            "One decode step: `token_id` comes in as the last token and goes out "
            "as the next, in the same Tensor, which is returned. Appends one token "
            "to the KV cache; nothing checks the budget.")
        .def(
            "__call__",
            [pyName](nb::handle, nb::args, nb::kwargs) -> nb::object { refuseCall(pyName); },
            "Not available: the model is driven by prefill() and decode().")
        .def("reset_cache",
             [](Model& self) {
                 nb::gil_scoped_release rel;   // waits the in-flight forward
                 self.resetCache();
             },
             "Rewind the KV cache to an empty context.")
        .def_prop_ro("tokens_so_far",
            [](Model& self) { return self.tokensSoFar_; },
            "Tokens the KV cache currently holds; reset_cache() zeroes it.")
        .def_prop_ro("kv_cache_dtype",
            [](Model& self) { return self.runtimeCfg.kvCacheDtype; },
            "The K/V width: the cache and what feeds it.")
        .def_prop_ro("max_runtime_context",
            [](Model& self) { return self.runtimeCfg.maxRuntimeContext; },
            "The KV-cache budget, clamped to the checkpoint's max_trained_context.");
}

}   // namespace pyvai

#endif // PYVAI_MODELS_LM_H
