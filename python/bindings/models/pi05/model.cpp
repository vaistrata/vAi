#include "python/bindings/models/lm.h"
#include "python/bindings/models/models.h"
#include "python/bindings/convert.h"
#include "python/bindings/models/config-base.h"
#include "python/bindings/models/pi05/processing.h"
#include "python/bindings/models/pi05/profile-tags.h"
#include "models/production/pi05/net.h"
#include "models/production/pi05/lerobot-weights.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>


namespace pyvai {

namespace {

const Presets<pi05::ModelConfig>& pi05Presets()
{
    static const Presets<pi05::ModelConfig> presets = {
        { "3B", &pi05::ModelConfig::_3B },
    };
    return presets;
}


// What prefill() takes on trust: the two shapes and the camera count, and the
// prompt length it only asserts on. Within both limits the cache has room for
// the chunk.
void checkPrefill(const pi05::Net& model, const vai::Tensor& images, const vai::Tensor& promptTokens)
{
    const auto& cfg = model.modelCfg;
    const auto& shape = images.shape();
    const uint32_t S = cfg.visionCfg.imageSize;
    if (shape.size() != 4 || shape[1] != cfg.visionCfg.imageChannels
        || shape[2] != S || shape[3] != S)
        throw nb::value_error(("Pi05Model.prefill: images must be [cameras, "
            + std::to_string(cfg.visionCfg.imageChannels) + ", " + std::to_string(S) + ", "
            + std::to_string(S) + "]").c_str());
    const uint32_t cameras = shape[0];
    if (cameras < 1 || cameras > cfg.maxCameras)
        throw nb::value_error(("Pi05Model.prefill: images carries " + std::to_string(cameras)
            + " cameras; the policy takes 1 to " + std::to_string(cfg.maxCameras)).c_str());
    requireTokenIds(promptTokens, "Pi05Model.prefill");

    const uint32_t tokens = promptTokens.shape()[0];
    if (tokens > cfg.maxPromptTokens)
        throw nb::value_error(("Pi05Model.prefill: the prompt is " + std::to_string(tokens)
            + " tokens, task and state together; the KV cache holds "
            + std::to_string(cfg.maxPromptTokens) + " (max_prompt_tokens)").c_str());
}

}   // namespace


void bindPi05(nb::module_& m)
{
    auto cfgCls = nb::class_<pi05::ModelConfig>(
        m,
        "Pi05Config",
        "A pi0.5 checkpoint (models/production/pi05/net.h): the SigLIP tower, the Gemma 2B "
        "language stack and the Gemma 300M action stack on adaRMS, and the "
        "flow-matching head's shape. Built by preset(variant).");
    bindPresets(cfgCls, "Pi05Config", pi05Presets());
    cfgCls
        .def_rw("max_cameras",
                &pi05::ModelConfig::maxCameras,
                "Image slots the policy was trained with; a prefill may bring fewer.")
        .def_rw("max_prompt_tokens",
                &pi05::ModelConfig::maxPromptTokens,
                "Prompt tokens the KV cache is sized for: the task and the "
                "discretized state, as text.")
        .def_rw("action_dim", &pi05::ModelConfig::actionDim, "Width of one action; the state is zero-padded to it too.")
        .def_rw("action_chunk", &pi05::ModelConfig::actionChunk, "Actions one chunk carries.")
        .def_rw("time_min_period", &pi05::ModelConfig::timeMinPeriod, "Fastest sinusoid of the timestep embedding.")
        .def_rw("time_max_period", &pi05::ModelConfig::timeMaxPeriod, "... and its slowest.")
        .def_rw("vision_cfg", &pi05::ModelConfig::visionCfg, "The SigLIP tower (VisionModelConfig).")
        .def_rw("language_cfg",
                &pi05::ModelConfig::languageCfg,
                "The Gemma 2B stack that reads the cameras and the prompt "
                "(LanguageModelConfig; max_trained_context is 0, nothing clamps it).")
        .def_rw("action_cfg",
                &pi05::ModelConfig::actionCfg,
                "The Gemma 300M action stack (TransformerConfig), its norms set by "
                "the timestep; num_layers, head_dim and num_kv_heads must match "
                "language_cfg's, whose KV cache it shares.")
        .def_prop_ro("max_context",
                     &pi05::ModelConfig::maxContext,
                     "The KV cache's depth: every camera the policy allows, the prompt "
                     "and the chunk.")
        .def_prop_ro("cond_rows",
                     &pi05::ModelConfig::condRows,
                     "Rows of the action stack's per-step condition: scale, shift and "
                     "gate for both norms of every layer, then the final norm's scale "
                     "and shift.");

    auto cls =
        nb::class_<pi05::Net, vai::NeuralNet>(m, "Pi05Model")
            .def(
                "__init__",
                [](pi05::Net* self,
                   const pi05::ModelConfig& config,
                   vai::Dtype w_default_dtype,
                   vai::Dtype o_default_dtype,
                   std::optional<eva::Device>
                       device,
                   uint32_t denoise_steps,
                   vai::Dtype kv_cache_dtype) {
                    checkVisionConfig(config.visionCfg, "Pi05Model", "config.vision_cfg");
                    checkLanguageConfig(config.languageCfg,
                                        "Pi05Model",
                                        "config.language_cfg",
                                        /*clampsContext*/ false);
                    checkTransformerConfig(config.actionCfg, "Pi05Model", "config.action_cfg");
                    // Paired layers share one K/V cache, layer for layer.
                    const vai::TransformerConfig& lang = config.languageCfg;
                    const vai::TransformerConfig& act = config.actionCfg;
                    if (act.numLayers != lang.numLayers || act.layerCfg.headDim != lang.layerCfg.headDim ||
                        act.layerCfg.numKVHeads != lang.layerCfg.numKVHeads)
                        throw nb::value_error(
                            "Pi05Model: config.action_cfg must pair with config.language_cfg "
                            "layer for layer (num_layers, layer_cfg.head_dim, layer_cfg.num_kv_heads): "
                            "the two stacks write one KV cache");
                    if (config.maxCameras == 0 || config.maxPromptTokens == 0 || config.actionDim == 0 ||
                        config.actionChunk == 0)
                        throw nb::value_error("Pi05Model: max_cameras, max_prompt_tokens, action_dim and action_chunk "
                                              "must be >= 1");
                    // The weight loader tables the timestep embedding with
                    // vai::node::timeEmbedding(), which checks neither.
                    if (act.layerCfg.embedDim < 4 || act.layerCfg.embedDim % 2 != 0)
                        throw nb::value_error("Pi05Model: config.action_cfg.layer_cfg.embed_dim must be even and >= 4 "
                                              "(the timestep embedding is sin and cos halves)");
                    if (!(config.timeMinPeriod > 0.0f) || !(config.timeMaxPeriod > config.timeMinPeriod))
                        throw nb::value_error("Pi05Model: need 0 < time_min_period < time_max_period");
                    if (denoise_steps == 0)
                        throw nb::value_error("Pi05Model: denoise_steps must be >= 1");
                    const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                    new (self) pi05::Net(cfg,
                                         config,
                                         {
                                             .denoiseSteps = denoise_steps,
                                             .kvCacheDtype = kv_cache_dtype,
                                         });
                },
                nb::arg("config") = pi05::ModelConfig::_3B(),
                nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                nb::arg("o_default_dtype") = vai::Dtype::Fp32,
                nb::arg("device") = nb::none(),
                nb::arg("denoise_steps") = 10u,
                nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                "Build a pi0.5 model (default config: Pi05Config.preset('3B')). The KV "
                "cache is max_context deep, fixed by the config.");
    bindWeightsAndTags<pi05::Net>(cls,
                                  "Pi05Model",
                                  pi05::lerobot::loadWeights,
                                  "Load the LeRobot pi0.5 checkpoint (lerobot/pi05_base) from its folder "
                                  "(data/pi05/ckpts/<variant>): every *.safetensors shard in it is read. "
                                  "The action stack's per-step condition is computed here, for denoise_steps.",
                                  examples::setProfileTags);

    cls.def(
           "prefill",
           [](nb::handle_t<pi05::Net> self,
              nb::handle_t<vai::Tensor>
                  images,
              nb::handle_t<vai::Tensor>
                  prompt_token_ids) {
               auto& model = nb::cast<pi05::Net&>(self);
               const char* who = "Pi05Model.prefill";
               vai::Tensor img = stageInput(self, "inImages", images, who);
               vai::Tensor ids = stageInput(self, "inPromptTokens", prompt_token_ids, who);
               checkPrefill(model, img, ids);
               nb::gil_scoped_release rel;   // waits the previous forward
               model.prefill(img, ids);
           },
           nb::arg("images"),
           nb::arg("prompt_token_ids"),
           "Leave one observation in the KV cache: `images` [cameras, 3, S, S] floats "
           "(1 to max_cameras, letterboxed, in [-1, 1]) and the [T] uint32 ids "
           "(Tensor.from_u32, T <= max_prompt_tokens) of pi05_make_prompt's text, "
           "which carries the state. Restarts the timestep.")
        .def(
            "decode",
            [](nb::handle_t<pi05::Net> self, nb::handle_t<vai::Tensor> action) -> nb::object {
                auto& model = nb::cast<pi05::Net&>(self);
                auto& x = nb::cast<vai::Tensor&>(action);
                const auto& cfg = model.modelCfg;
                if (model.prefixLength_ == 0)
                    throw std::runtime_error("Pi05Model.decode: call prefill() first");
                if (!x.isShapeOf(cfg.actionChunk, cfg.actionDim))
                    throw nb::value_error(("Pi05Model.decode: action must be [" + std::to_string(cfg.actionChunk) +
                                           ", " + std::to_string(cfg.actionDim) + "], the noise the chunk starts from")
                                              .c_str());
                {
                    nb::gil_scoped_release rel;   // waits the previous forward
                    model.decode(&x);
                }
                action.attr("_pyvai_net") = self;   // it views the net's buffer now
                return nb::borrow(action);
            },
            nb::arg("action"),
            "Denoise one action chunk over the last prefill: `action` [action_chunk, "
            "action_dim] comes in as noise (pi0_sample_noise) and goes out as the chunk, "
            "in the same Tensor, which is returned. Runs all denoise_steps Euler steps.")
        .def(
            "__call__",
            [](nb::handle, nb::args, nb::kwargs) -> nb::object { refuseCall("Pi05Model"); },
            "Not available: the model is driven by prefill() and decode().")

        .def_prop_ro(
            "max_cameras",
            [](pi05::Net& self) { return self.modelCfg.maxCameras; },
            "Most cameras one prefill takes (config.max_cameras).")
        .def_prop_ro(
            "num_patches",
            [](pi05::Net& self) { return self.modelCfg.visionCfg.numPatches(); },
            "Tokens one camera adds to the prefix (vision_cfg.num_patches).")
        .def_prop_ro(
            "image_size",
            [](pi05::Net& self) { return self.modelCfg.visionCfg.imageSize; },
            "Side of the square each camera is letterboxed into.")
        .def_prop_ro(
            "max_prompt_tokens",
            [](pi05::Net& self) { return self.modelCfg.maxPromptTokens; },
            "Longest prompt, task and state together, one prefill takes.")
        .def_prop_ro("action_dim", [](pi05::Net& self) { return self.modelCfg.actionDim; })
        .def_prop_ro("action_chunk", [](pi05::Net& self) { return self.modelCfg.actionChunk; })
        .def_prop_ro(
            "max_context",
            [](pi05::Net& self) { return self.modelCfg.maxContext(); },
            "The KV cache's depth (config.max_context).")
        .def_prop_ro(
            "prefix_length",
            [](pi05::Net& self) { return self.prefixLength_; },
            "Tokens the last prefill left in the cache: the cameras and the prompt.")
        .def_prop_ro(
            "denoise_steps",
            [](pi05::Net& self) { return self.runtimeCfg.denoiseSteps; },
            "Euler steps decode() takes from noise to the action chunk.")
        .def_prop_ro(
            "kv_cache_dtype",
            [](pi05::Net& self) { return self.runtimeCfg.kvCacheDtype; },
            "The K/V width: the cache and what feeds it.");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inImages", "inPromptTokens", "inActions"}, {"outActions"});

    m.def("pi05_make_prompt",
          [](const std::string& task, std::vector<float> state)
          {
              // The bin is an int cast of floor(), undefined for a NaN or past
              // int's range; [-2, 2] already lands every finite value in its bin.
              for (float& x : state)
              {
                  if (!std::isfinite(x))
                      throw nb::value_error("pi05_make_prompt: state values must be finite");
                  x = std::clamp(x, -2.0f, 2.0f);
              }
              return examples::makePrompt(task, state);
          },
          nb::arg("task"), nb::arg("state"),
          "pi0.5's prompt, \"Task: {task}, State: {bins};\\nAction: \": the task "
          "trimmed, its underscores and newlines turned to spaces, and each `state` "
          "value (normalized to [-1, 1], zero-padded to action_dim) as one of 256 "
          "bins, floor((x + 1) * 128) held to [-1, 255]. Encode it with BOS for prefill().");
}

}   // namespace pyvai
