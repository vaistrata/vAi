#ifndef VAI_ENGINE_NODE_PROFILER_H
#define VAI_ENGINE_NODE_PROFILER_H

#include "tensor.h"
#include <map>
#include <string>
#include <vector>
#include <cstdio>

namespace vai {

class Node;

/////////////////////////////////////////////////////////////////////////////////////////
// NodeProfiler - per-node GPU time, measured by bracketing each tagged node with
// timestamps while the command buffer is recorded.
//
// It owns the tags, so a node keeps no profiling state and everything the
// recording needs comes from one map.
//
// The owner drives four points of a recording, in order:
//   beginRecord -> (beginNode / endNode)* -> endRecord
// and calls collect() once the forward it recorded has completed.
//
// NeuralNet holds this behind a pointer, so this header reaches only the one
// translation unit that records commands.
/////////////////////////////////////////////////////////////////////////////////////////
class NodeProfiler
{
    eva::Device device_;
    eva::QueryPool pool_;

    // Tag per query pair as recorded, so a tag changed since then cannot shift the
    // read. Its size is also how many pairs the pool actually holds, and the query
    // layout is derived from it alone:
    //   [0] forward start, [1 + 2i] node i start, [2 + 2i] node i end, [1 + 2N] forward end
    std::vector<std::string> recordedTags_;

    // Only tagged nodes are present, so its size is the tag count.
    std::map<const Node*, std::string> tags_;
    bool tagsDirty_ = false;

    std::map<std::string, double> accumMs_;
    double accumTotalMs_ = 0.0;
    uint32_t iters_ = 0;

    static constexpr auto kStage = eva::PIPELINE_STAGE::ALL_COMMANDS;

public:
    explicit NodeProfiler(eva::Device device) : device_(std::move(device)) {}

    bool enabled() const { return !tags_.empty(); }
    bool isTagged(const Node* node) const { return tags_.count(node) != 0; }

    const std::string& tagOf(const Node* node) const;

    // An empty tag untags the node. Returns whether anything changed; the next
    // recording then resizes the pool and drops the accumulators, since a
    // different tag set makes the running averages incomparable.
    bool setTag(const Node* node, const std::string& tag);
    bool clearTags();

    // Reallocating is safe only because the owner waited the previous forward
    // before re-recording; the old pool is idle and already read.
    void beginRecord(eva::CommandBuffer cb);
    void beginNode(eva::CommandBuffer cb, const Node* node);
    void endNode(eva::CommandBuffer cb);
    void endRecord(eva::CommandBuffer cb);

    // Valid only once the recorded forward has completed. One read covers every
    // pair; asking pair by pair would pay a driver round-trip per tag.
    void collect();

    void reset();
    double ms(const std::string& tag) const;
    void print(bool doReset);
};


inline const std::string& NodeProfiler::tagOf(const Node* node) const
{
    static const std::string kNone;
    auto it = tags_.find(node);
    return (it == tags_.end()) ? kNone : it->second;
}

inline bool NodeProfiler::setTag(const Node* node, const std::string& tag)
{
    auto it = tags_.find(node);

    if (tag.empty())
    {
        if (it == tags_.end()) return false;
        tags_.erase(it);
    }
    else if (it == tags_.end())
    {
        tags_.emplace(node, tag);
    }
    else
    {
        if (it->second == tag) return false;
        it->second = tag;
    }

    tagsDirty_ = true;
    return true;
}

inline bool NodeProfiler::clearTags()
{
    if (tags_.empty()) return false;
    tags_.clear();
    tagsDirty_ = true;
    return true;
}

inline void NodeProfiler::beginRecord(eva::CommandBuffer cb)
{
    if (tagsDirty_)
    {
        tagsDirty_ = false;

        // Spare room lets a new tag reuse the pool. The surplus stays unwritten,
        // and waiting on a query no recording wrote never returns.
        const uint32_t need = 2 * (uint32_t)tags_.size() + 2;
        if (enabled() && (!pool_ || pool_.queryCount() < need))
        {
            if (pool_) pool_.destroy();
            pool_ = device_.createTimestampQueryPool(2 * need);
        }
        reset();
    }

    recordedTags_.clear();
    recordedTags_.reserve(tags_.size());

    if (!enabled())
        return;

    cb.resetQueryPool(pool_, 0, pool_.queryCount());
    cb.writeTimestamp(kStage, pool_, 0);
}

inline void NodeProfiler::beginNode(eva::CommandBuffer cb, const Node* node)
{
    cb.writeTimestamp(kStage, pool_, 1 + 2 * (uint32_t)recordedTags_.size());
    recordedTags_.push_back(tagOf(node));
}

inline void NodeProfiler::endNode(eva::CommandBuffer cb)
{
    cb.writeTimestamp(kStage, pool_, 2 * (uint32_t)recordedTags_.size());
}

inline void NodeProfiler::endRecord(eva::CommandBuffer cb)
{
    if (!enabled())
        return;

    cb.writeTimestamp(kStage, pool_, 1 + 2 * (uint32_t)recordedTags_.size());
}

inline void NodeProfiler::collect()
{
    if (recordedTags_.empty())
        return;

    const uint32_t pairs = (uint32_t)recordedTags_.size();
    const std::vector<uint64_t> ticks = pool_.getResults(0, 2 * pairs + 2);
    const double toMs = pool_.timestampPeriod() / 1e6;

    for (uint32_t i = 0; i < pairs; ++i)
        accumMs_[recordedTags_[i]] += (ticks[2 + 2 * i] - ticks[1 + 2 * i]) * toMs;

    accumTotalMs_ += (ticks[1 + 2 * pairs] - ticks[0]) * toMs;
    ++iters_;
}

inline void NodeProfiler::reset()
{
    accumMs_.clear();
    accumTotalMs_ = 0.0;
    iters_ = 0;
}

inline double NodeProfiler::ms(const std::string& tag) const
{
    if (iters_ == 0) return 0.0;
    auto it = accumMs_.find(tag);
    return (it == accumMs_.end()) ? 0.0 : it->second / iters_;
}

inline void NodeProfiler::print(bool doReset)
{
    if (iters_ == 0) return;

    // Build a tree from dotted tag names. Each leaf carries its measured ms;
    // each interior node's `total` is the sum of itself and all descendants.
    struct TreeNode {
        std::map<std::string, TreeNode> children;
        double ms = 0.0;
        double total = 0.0;
    };

    TreeNode root;
    for (auto& [tag, ms] : accumMs_)
    {
        TreeNode* node = &root;
        size_t start = 0;
        for (size_t i = 0; i <= tag.size(); ++i)
        {
            if (i == tag.size() || tag[i] == '.')
            {
                node = &node->children[tag.substr(start, i - start)];
                start = i + 1;
            }
        }
        node->ms = ms;
    }

    auto computeTotal = [&](auto& self, TreeNode& n) -> double {
        n.total = n.ms;
        for (auto& [_, c] : n.children) n.total += self(self, c);
        return n.total;
    };
    computeTotal(computeTotal, root);

    const double total = accumTotalMs_;
    double namedSum = 0.0;
    for (auto& [_, v] : accumMs_) namedSum += v;
    const double other = total - namedSum;

    // UTF-8-aware display column counter for tree-connector chars (each
    // box-drawing glyph is a 3-byte BMP code point that occupies 1 column).
    auto displayWidth = [](const std::string& s) -> size_t {
        size_t cols = 0;
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = s[i];
            if      (c < 0x80)            { ++cols;   ++i;     }
            else if ((c & 0xE0) == 0xC0)  { ++cols;   i += 2;  }
            else if ((c & 0xF0) == 0xE0)  { ++cols;   i += 3;  }
            else if ((c & 0xF8) == 0xF0)  { cols += 2; i += 4; }
            else                          {           ++i;     }
        }
        return cols;
    };

    printf("=== Per-node GPU profile (avg over %u iters) ===\n", iters_);

    constexpr int kNameCol = 28;
    constexpr int kBarMax  = 40;   // chars at 100%

    auto printNode = [&](auto& self, const std::string& name, const TreeNode& n,
                         std::vector<bool>& trail, bool isRoot) -> void {
        if (!isRoot)
        {
            std::string prefix;
            for (size_t i = 0; i + 1 < trail.size(); ++i)
                prefix += trail[i] ? "    " : "\xE2\x94\x82   ";   // " " or "│   "
            if (!trail.empty())
                prefix += trail.back() ? "\xE2\x94\x94\xE2\x94\x80\xE2\x94\x80 "    // "└── "
                                       : "\xE2\x94\x9C\xE2\x94\x80\xE2\x94\x80 ";   // "├── "

            std::string label = prefix + name;
            size_t w = displayWidth(label);
            std::string pad(w < (size_t)kNameCol ? kNameCol - w : 0, ' ');

            double avgMs = n.total / iters_;
            double pct   = total > 0.0 ? 100.0 * n.total / total : 0.0;

            int barLen = std::min(int(pct * kBarMax / 100.0 + 0.5), kBarMax);
            std::string bar;
            for (int b = 0; b < barLen; ++b) bar += "\xE2\x96\x88";    // "█"

            printf("  %s%s : %8.4f ms  (%5.2f%%)  %s\n",
                   label.c_str(), pad.c_str(), avgMs, pct, bar.c_str());
        }

        size_t cnt = n.children.size();
        size_t idx = 0;
        for (auto& [cn, cnode] : n.children) {
            ++idx;
            trail.push_back(idx == cnt);
            self(self, cn, cnode, trail, false);
            trail.pop_back();
        }
    };

    std::vector<bool> trail;
    printNode(printNode, "", root, trail, true);

    printf("  %-*s : %8.4f ms  (%5.2f%%)\n", kNameCol, "OTHER (untagged)",
           other / iters_, total > 0.0 ? 100.0 * other / total : 0.0);
    printf("  %-*s : %8.4f ms\n", kNameCol, "TOTAL (full forward)", total / iters_);

    if (doReset)
        reset();
}

} // namespace vai

#endif   // VAI_ENGINE_NODE_PROFILER_H
