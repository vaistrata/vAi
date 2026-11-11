#ifndef VAI_ENGINE_PIPELINE_POOL_H
#define VAI_ENGINE_PIPELINE_POOL_H

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <utility>

namespace vai {

/////////////////////////////////////////////////////////////////////////////////////////
// PipelinePool - compute-pipeline cache shared by all nodes, replacing the
// per-node static caches.
//
// The key is everything pipeline creation reads: the shader directory and
// name, the specialization constants, and the pipeline config. That is the
// whole of a pipeline's identity (the descriptor layout is reflected out of
// the SPV), so a node asks with the values it already has and never names a
// variant of its own.
/////////////////////////////////////////////////////////////////////////////////////////
class PipelinePool
{
public:
    // Pipeline-creation extras forwarded to ComputePipelineCreateInfo.
    struct PipelineConfig {
        uint32_t requiredSubgroupSize = 0;   // 0: implementation default
        bool robustBufferAccess = false;
        bool captureStatistics = false;      // diagnostics only; see ComputePipeline::statistics()

        bool operator==(const PipelineConfig&) const = default;
    };

private:
    struct Key {
        std::string shaderDir;
        std::string shaderName;
        eva::SpecializationConstant spec;
        PipelineConfig cfg;

        bool operator==(const Key&) const = default;
    };

    // The hash only picks the bucket; Key::operator== settles equality, so a
    // collision costs a comparison rather than the wrong pipeline.
    struct KeyHash {
        static uint64_t combine(uint64_t h1, uint64_t h2) noexcept
        {
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }

        size_t operator()(const Key& key) const noexcept
        {
            uint64_t h = std::hash<std::string>{}(key.shaderDir);
            h = combine(h, std::hash<std::string>{}(key.shaderName));
            h = combine(h, key.spec.hash());
            h = combine(h, key.cfg.requiredSubgroupSize);
            h = combine(h, key.cfg.robustBufferAccess);
            return (size_t) combine(h, key.cfg.captureStatistics);
        }
    };

    eva::Device device_;
    std::unordered_map<Key, eva::ComputePipeline, KeyHash> pool_;

public:
    PipelinePool(eva::Device device) : device_(std::move(device)) {}

    // Cached pipeline for the shader specialized this way; created on first
    // request. The shader is loaded from the directory and name the key
    // carries, so a key can never point at a different shader.
    eva::ComputePipeline request(
        const std::string& shaderDir,
        const std::string& shaderName,
        eva::SpecializationConstant spec,
        PipelineConfig cfg);

    // A shader built into the engine's own output directory. Model-local
    // shaders name their directory with the four-argument form.
    eva::ComputePipeline request(
        const std::string& shaderName,
        eva::SpecializationConstant spec,
        PipelineConfig cfg);
};


inline eva::ComputePipeline PipelinePool::request(
    const std::string& shaderDir,
    const std::string& shaderName,
    eva::SpecializationConstant spec,
    PipelineConfig cfg = {})
{
    Key key{shaderDir, shaderName, std::move(spec), cfg};

    auto it = pool_.find(key);
    if (it == pool_.end())
    {
        eva::SpvBlob spv = eva::SpvBlob::readFrom(
            (shaderDir + "/" + shaderName + ".comp.spv").c_str());
        eva::ComputePipeline pipe = device_.createComputePipeline({
            .csStage = {spv, eva::SpecializationConstant(key.spec)},
            .requiredSubgroupSize = cfg.requiredSubgroupSize,
            .robustBufferAccess = cfg.robustBufferAccess,
            .captureStatistics = cfg.captureStatistics,
        });
        it = pool_.emplace(std::move(key), std::move(pipe)).first;
    }
    return it->second;
}

inline eva::ComputePipeline PipelinePool::request(
    const std::string& shaderName,
    eva::SpecializationConstant spec,
    PipelineConfig cfg = {})
{
    return request(SHADER_OUTPUT_DIR, shaderName, std::move(spec), cfg);
}

} // namespace vai

#endif   // VAI_ENGINE_PIPELINE_POOL_H
