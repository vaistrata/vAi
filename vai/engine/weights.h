/*=============================================================================
  Written by   oneway@vaistrata.com
  Brief        Checkpoint-distributor tag and the loadWeights<"...">() entry point.
=============================================================================*/

#ifndef VAI_ENGINE_WEIGHTS_H
#define VAI_ENGINE_WEIGHTS_H

#include "safetensors-parser.h"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vai {

// String usable as a template argument (C++20 structural NTTP), so a
// checkpoint distributor can be spelled at the call site: loadWeights<"meta">(net).
template<size_t N>
struct Ckpt
{
    char s[N];
    constexpr Ckpt(const char (&str)[N]) { std::copy_n(str, N, s); }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Checkpoint - the shards of one checkpoint folder, read by tensor name
//
// A large checkpoint is published in more than one file, so a key is served by
// whichever shard carries it. Every *.safetensors in the folder is opened, which
// makes a single-file export the one-shard case rather than a separate path.
/////////////////////////////////////////////////////////////////////////////////////////
class Checkpoint
{
    std::vector<std::unique_ptr<vai::SafetensorsParser>> shards_;
    std::unordered_map<std::string, const vai::SafetensorsParser*> owner_;

public:
    explicit Checkpoint(const std::string& dir)
    {
        for (const auto& entry : std::filesystem::directory_iterator(dir))
        {
            if (entry.path().extension() != ".safetensors")
                continue;

            auto& shard = shards_.emplace_back(
                std::make_unique<vai::SafetensorsParser>(entry.path().string().c_str()));
            for (const std::string& key : shard->getTensorNames())
                owner_.emplace(key, shard.get());
        }

        if (owner_.empty())
            throw std::runtime_error("no *.safetensors under " + dir);
    }

    bool contains(std::string_view key) const { return owner_.contains(std::string(key)); }

    vai::SafetensorRef operator[](std::string_view key) const
    {
        auto it = owner_.find(std::string(key));
        if (it == owner_.end())
            throw std::runtime_error("checkpoint has no tensor " + std::string(key));
        return (*it->second)[key];
    }
};

} // namespace vai

// Load `model`'s weights from the checkpoint distributed by `Name`. Declared
// only — each model ships its own specializations (models/<impl>/<model>/), which
// also own the download location and the on-disk path of that checkpoint.
// An unimplemented (model, distributor) pair fails at link time.
template<vai::Ckpt Name, typename Model>
void loadWeights(Model& model);

#endif   // VAI_ENGINE_WEIGHTS_H
