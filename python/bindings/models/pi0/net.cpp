#include "python/bindings/models/lm.h"
#include "python/bindings/models/models.h"
#include "python/bindings/convert.h"
#include "python/bindings/models/config-base.h"
#include "models/production/pi0/processing.h"
#include "models/production/pi0/net.h"
#include "models/production/pi0/lerobot-weights.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pi0 = model::production::pi0;


namespace pyvai {

namespace {

const Presets<pi0::ModelConfig>& pi0Presets()
{
    static const Presets<pi0::ModelConfig> presets = {
        {"3B", &pi0::ModelConfig::_3B},
    };
    return presets;
}


// What prefill() and the cache take on trust: the three shapes, the camera
// count, and room left in the cache for the state token and the chunk.
void checkPrefill(const pi0::Net& model, const vai::Tensor& images, const vai::Tensor& promptTokens,
                  const vai::Tensor& state)
{
    const auto& cfg = model.modelCfg;
    const auto& shape = images.shape();
    const uint32_t S = cfg.visionCfg.imageSize;
    if (shape.size() != 4 || shape[1] != cfg.visionCfg.imageChannels || shape[2] != S || shape[3] != S)
        throw nb::value_error(("Pi0Net.prefill: images must be [cameras, " +
                               std::to_string(cfg.visionCfg.imageChannels) + ", " + std::to_string(S) + ", " +
                               std::to_string(S) + "]")
                                  .c_str());
    const uint32_t cameras = shape[0];
    if (cameras < 1 || cameras > cfg.maxCameras)
        throw nb::value_error(("Pi0Net.prefill: images carries " + std::to_string(cameras) +
                               " cameras; the policy takes 1 to " + std::to_string(cfg.maxCameras))
                                  .c_str());
    requireTokenIds(promptTokens, "Pi0Net.prefill");
    if (!state.isShapeOf(1u, cfg.actionDim))
        throw nb::value_error(
            ("Pi0Net.prefill: state must be [1, " + std::to_string(cfg.actionDim) + "], zero-padded to action_dim")
                .c_str());

    const uint32_t tokens = promptTokens.shape()[0];
    const uint64_t context = uint64_t(cameras) * cfg.visionCfg.numPatches() + tokens + 1 + cfg.actionChunk;
    if (context > cfg.maxContext())
        throw nb::value_error(("Pi0Net.prefill: " + std::to_string(cameras) + " cameras and " + std::to_string(tokens) +
                               " prompt tokens, with the state token and the chunk, are " + std::to_string(context) +
                               " tokens; the KV cache holds " + std::to_string(cfg.maxContext()))
                                  .c_str());
}

}   // namespace


void bindPi0(nb::module_& m)
{
    auto cfgCls =
        nb::class_<pi0::ModelConfig>(m, "Pi0Config",
                                     "A pi0 checkpoint (models/production/pi0/net.h): the SigLIP tower, the Gemma 2B "
                                     "language stack and the Gemma 300M action stack, and the flow-matching "
                                     "head's shape. Built by preset(variant).");
    bindPresets(cfgCls, "Pi0Config", pi0Presets());
    cfgCls
        .def_rw("max_cameras", &pi0::ModelConfig::maxCameras,
                "Image slots the policy was trained with; a prefill may bring fewer.")
        .def_rw("max_prompt_tokens", &pi0::ModelConfig::maxPromptTokens,
                "Task-string tokens the KV cache is sized for (the trained length).")
        .def_rw("action_dim", &pi0::ModelConfig::actionDim, "Width of one action; the state is zero-padded to it too.")
        .def_rw("action_chunk", &pi0::ModelConfig::actionChunk, "Actions one chunk carries.")
        .def_rw("time_min_period", &pi0::ModelConfig::timeMinPeriod, "Fastest sinusoid of the timestep embedding.")
        .def_rw("time_max_period", &pi0::ModelConfig::timeMaxPeriod, "... and its slowest.")
        .def_rw("vision_cfg", &pi0::ModelConfig::visionCfg, "The SigLIP tower (VisionModelConfig).")
        .def_rw("language_cfg", &pi0::ModelConfig::languageCfg,
                "The Gemma 2B stack that reads the cameras and the prompt "
                "(LanguageModelConfig; max_trained_context is 0, nothing clamps it).")
        .def_rw("action_cfg", &pi0::ModelConfig::actionCfg,
                "The Gemma 300M action stack (TransformerConfig); num_layers, head_dim "
                "and num_kv_heads must match language_cfg's, whose KV cache it shares.")
        .def_prop_ro("max_context", &pi0::ModelConfig::maxContext,
                     "The KV cache's depth: every camera the policy allows, the prompt, "
                     "the state token and the chunk.");

    auto cls =
        nb::class_<pi0::Net, vai::NeuralNet>(m, "Pi0Net")
            .def(
                "__init__",
                [](pi0::Net* self, const pi0::ModelConfig& config, vai::Dtype w_default_dtype,
                   vai::Dtype o_default_dtype, std::optional<eva::Device> device, uint32_t denoise_steps,
                   vai::Dtype kv_cache_dtype) {
                    checkVisionConfig(config.visionCfg, "Pi0Net", "config.vision_cfg");
                    checkLanguageConfig(config.languageCfg, "Pi0Net", "config.language_cfg",
                                        /*clampsContext*/ false);
                    checkTransformerConfig(config.actionCfg, "Pi0Net", "config.action_cfg");
                    // Paired layers share one K/V cache, layer for layer.
                    const vai::TransformerConfig& lang = config.languageCfg;
                    const vai::TransformerConfig& act = config.actionCfg;
                    if (act.numLayers != lang.numLayers || act.layerCfg.headDim != lang.layerCfg.headDim ||
                        act.layerCfg.numKVHeads != lang.layerCfg.numKVHeads)
                        throw nb::value_error(
                            "Pi0Net: config.action_cfg must pair with config.language_cfg "
                            "layer for layer (num_layers, layer_cfg.head_dim, layer_cfg.num_kv_heads): "
                            "the two stacks write one KV cache");
                    if (config.maxCameras == 0 || config.maxPromptTokens == 0 || config.actionDim == 0 ||
                        config.actionChunk == 0)
                        throw nb::value_error("Pi0Net: max_cameras, max_prompt_tokens, action_dim and action_chunk "
                                              "must be >= 1");
                    // The weight loader tables the timestep embedding with
                    // vai::node::timeEmbedding(), which checks neither.
                    if (act.layerCfg.embedDim < 4 || act.layerCfg.embedDim % 2 != 0)
                        throw nb::value_error("Pi0Net: config.action_cfg.layer_cfg.embed_dim must be even and >= 4 "
                                              "(the timestep embedding is sin and cos halves)");
                    if (!(config.timeMinPeriod > 0.0f) || !(config.timeMaxPeriod > config.timeMinPeriod))
                        throw nb::value_error("Pi0Net: need 0 < time_min_period < time_max_period");
                    if (denoise_steps == 0)
                        throw nb::value_error("Pi0Net: denoise_steps must be >= 1");
                    const vai::NeuralNetConfig cfg = netConfig(w_default_dtype, o_default_dtype, device);
                    new (self) pi0::Net(cfg, config,
                                        {
                                            .denoiseSteps = denoise_steps,
                                            .kvCacheDtype = kv_cache_dtype,
                                        });
                },
                nb::arg("config") = pi0::ModelConfig::_3B(), nb::arg("w_default_dtype") = vai::Dtype::Fp16,
                nb::arg("o_default_dtype") = vai::Dtype::Fp32, nb::arg("device") = nb::none(),
                nb::arg("denoise_steps") = 10u, nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
                "Build a pi0 model (default config: Pi0Config.preset('3B')). The KV cache "
                "is max_context deep, fixed by the config.");
    bindLoadWeights<pi0::Net>(cls, "Pi0Net", pi0::lerobot::loadWeights,
                              "Load the LeRobot pi0 checkpoint (lerobot/pi0_base) from its folder "
                              "(data/ckpts/pi0/lerobot/<variant>): every *.safetensors shard in it is read.");

    cls.def(
           "prefill",
           [](nb::handle_t<pi0::Net> self, nb::handle_t<vai::Tensor> images, nb::handle_t<vai::Tensor> prompt_token_ids,
              nb::handle_t<vai::Tensor> state) {
               auto& model = nb::cast<pi0::Net&>(self);
               const char* who = "Pi0Net.prefill";
               vai::Tensor img = stageInput(self, "inImages", images, who);
               vai::Tensor ids = stageInput(self, "inPromptTokens", prompt_token_ids, who);
               vai::Tensor st = stageInput(self, "inState", state, who);
               checkPrefill(model, img, ids, st);
               nb::gil_scoped_release rel;   // waits the previous forward
               model.prefill(img, ids, st);
           },
           nb::arg("images"), nb::arg("prompt_token_ids"), nb::arg("state"),
           "Leave one observation in the KV cache: `images` [cameras, 3, S, S] floats "
           "(1 to max_cameras, letterboxed, in [-1, 1]), the task string's [T] uint32 "
           "ids with the trailing newline (Tensor.from_u32) and `state` "
           "[1, action_dim], the zero-padded proprioception. Restarts the timestep.")
        .def(
            "decode",
            [](nb::handle_t<pi0::Net> self, nb::handle_t<vai::Tensor> action) -> nb::object {
                auto& model = nb::cast<pi0::Net&>(self);
                auto& x = nb::cast<vai::Tensor&>(action);
                const auto& cfg = model.modelCfg;
                if (!model.stateToken_.validShape())
                    throw std::runtime_error("Pi0Net.decode: call prefill() first");
                if (!x.isShapeOf(cfg.actionChunk, cfg.actionDim))
                    throw nb::value_error(("Pi0Net.decode: action must be [" + std::to_string(cfg.actionChunk) + ", " +
                                           std::to_string(cfg.actionDim) + "], the noise the chunk starts from")
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
            "__call__", [](nb::handle, nb::args, nb::kwargs) -> nb::object { refuseCall("Pi0Net"); },
            "Not available: the model is driven by prefill() and decode().")

        .def_prop_ro(
            "max_cameras", [](pi0::Net& self) { return self.modelCfg.maxCameras; },
            "Most cameras one prefill takes (config.max_cameras).")
        .def_prop_ro(
            "num_patches", [](pi0::Net& self) { return self.modelCfg.visionCfg.numPatches(); },
            "Tokens one camera adds to the prefix (vision_cfg.num_patches).")
        .def_prop_ro(
            "image_size", [](pi0::Net& self) { return self.modelCfg.visionCfg.imageSize; },
            "Side of the square each camera is letterboxed into.")
        .def_prop_ro("max_prompt_tokens", [](pi0::Net& self) { return self.modelCfg.maxPromptTokens; })
        .def_prop_ro("action_dim", [](pi0::Net& self) { return self.modelCfg.actionDim; })
        .def_prop_ro("action_chunk", [](pi0::Net& self) { return self.modelCfg.actionChunk; })
        .def_prop_ro(
            "max_context", [](pi0::Net& self) { return self.modelCfg.maxContext(); },
            "The KV cache's depth (config.max_context).")
        .def_prop_ro(
            "prefix_length", [](pi0::Net& self) { return self.prefixLength_; },
            "Tokens the last prefill left in the cache: the cameras and the prompt.")
        .def_prop_ro(
            "denoise_steps", [](pi0::Net& self) { return self.runtimeCfg.denoiseSteps; },
            "Euler steps decode() takes from noise to the action chunk.")
        .def_prop_ro(
            "kv_cache_dtype", [](pi0::Net& self) { return self.runtimeCfg.kvCacheDtype; },
            "The K/V width: the cache and what feeds it.");

    // The graph is built in C++, out of sight of the Python input()/output().
    declareBoundaries(cls, {"inImages", "inPromptTokens", "inState", "inStateToken", "inActions"},
                      {"outStateToken", "outActions"});

    m.def(
        "pi0_sample_noise",
        [](size_t count, uint64_t seed) {
            std::vector<float> noise;
            {
                nb::gil_scoped_release rel;
                noise = pi0::sampleNoise(count, seed);
            }
            return ownedNdarray(std::move(noise), {count});
        },
        nb::arg("count"), nb::arg("seed") = 0,
        "`count` standard-normal floats for decode() to start from, seeded "
        "(splitmix64 through Box-Muller): the same seed gives the same chunk.");
}

}   // namespace pyvai
