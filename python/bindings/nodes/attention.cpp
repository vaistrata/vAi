#include "python/bindings/nodes/bind-node.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/attention.h"
#include "vai/groups/transformer.h"   // createCausalMask, writeCausalMask
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace pyvai {

namespace {

// An attention reads its causal mask from a buffer its owner holds: a C++ model
// keeps one per stack, a node built from Python its own, held here to rewrite.
struct PyAttention : vai::node::Attention
{
    eva::Buffer mask;

    explicit PyAttention(Config config, uint32_t bits)
    : vai::node::Attention(config)
    , mask(vai::createCausalMask(device(), bits))
    {
        setCausalMask(mask);
    }

    void writeMask(uint32_t bits, const vai::CausalBlockEnd& blockEnd)
    {
        // The GPU may still be reading the mask. A node only holds its net const.
        const_cast<vai::NeuralNet*>(ownedNet_)->sync();
        vai::writeCausalMask(mask, bits, blockEnd);
    }
};

}   // namespace


static void bindAttention(nb::module_& m)
{
    nb::class_<PyAttention, vai::Node>(m, "AttentionNode")
        .def(
            "__init__",
            [](PyAttention* self,
               uint32_t num_q_heads, uint32_t head_dim,
               uint32_t kv_cache_size, uint32_t num_kv_heads,
               uint32_t window_size, bool causal,
               vai::Dtype kv_cache_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("AttentionNode");
                // The core's own check divides by the head count.
                if (num_q_heads == 0 || head_dim == 0)
                    throw nb::value_error("AttentionNode: num_q_heads and head_dim must be >= 1");
                if (num_kv_heads > 0 &&
                    (num_kv_heads > num_q_heads || num_q_heads % num_kv_heads != 0))
                    throw nb::value_error(
                        "AttentionNode: num_kv_heads must divide num_q_heads");
                if (kv_cache_dtype == vai::Dtype::None)
                    throw nb::value_error("AttentionNode: kv_cache_dtype is required");
                new (self) PyAttention(
                    vai::node::Attention::Config{
                        .headDim      = head_dim,
                        .numQHeads    = num_q_heads,
                        .numKVHeads   = num_kv_heads,
                        .kvCacheSize  = kv_cache_size,
                        .windowSize   = window_size,
                        .kvCacheDtype = kv_cache_dtype,
                        .oDtype       = o_dtype.value_or(vai::Dtype::None),
                    },
                    causal ? 1u : 0u);
            },
            nb::arg("num_q_heads"),
            nb::arg("head_dim"),
            nb::arg("kv_cache_size")  = 0u,   // 0 = no cache
            nb::arg("num_kv_heads")   = 0u,
            nb::arg("window_size")    = 0u,   // 0 = attend the whole prefix
            nb::arg("causal")         = true,
            nb::arg("kv_cache_dtype") = vai::Dtype::Fp16,
            nb::arg("o_dtype")        = nb::none(),
            "Attention over Q [..., Hq, Tq, head_dim] and K / V [..., Hkv, Tq, "
            "head_dim], leading axes a batch (of one when kv_cache_size > 0), each at "
            "whatever dtype arrives; out0 "
            "is token-major [..., Tq, Hq, head_dim]. causal=False lets a query see "
            "the newer tokens of its own forward, and window_size > 0 limits it to "
            "its last window_size positions. kv_cache_size > 0 keeps that many "
            "tokens at kv_cache_dtype and needs set_kv_cache_counter() before the "
            "first forward."
        )
        .def(
            "update_causality",
            [](PyAttention& self, uint32_t bits, const std::vector<uint32_t>& block_end)
            {
                // The core asserts the order; the list is a fixed-size array there.
                constexpr size_t maxEnds = std::tuple_size_v<vai::CausalBlockEnd>;
                if (block_end.size() > maxEnds)
                    throw nb::value_error(("AttentionNode.update_causality: at most "
                        + std::to_string(maxEnds) + " segment ends").c_str());
                vai::CausalBlockEnd ends = {};
                for (size_t i = 0; i < block_end.size(); ++i)
                {
                    if (block_end[i] == 0 || (i > 0 && block_end[i] <= block_end[i - 1]))
                        throw nb::value_error(
                            "AttentionNode.update_causality: block_end must be positive and "
                            "increasing");
                    ends[i] = block_end[i];
                }
                nb::gil_scoped_release rel;   // waits for a forward still reading it
                self.writeMask(bits, ends);
            },
            nb::arg("bits")      = 1u,
            nb::arg("block_end") = std::vector<uint32_t>{},
            "Split each forward's new tokens into segments and mark which are "
            "causal. block_end lists the increasing token index where each segment "
            "ends; the last one runs to the end. A token sees the cache, every "
            "earlier segment, and its own segment up to itself if causal, whole if "
            "not. Segment k of n is bit n-1-k of `bits`: bits=1 with no block_end is "
            "plain causal attention, 0 bidirectional. A change needs no "
            "re-recording; it waits for a forward still in flight."
        )
        .def(
            "set_kv_cache_counter",
            [](PyAttention& self, KVCacheCounter& counter) {
                // Without a cache there is nothing to count from.
                if (self.config_.kvCacheSize == 0)
                    throw nb::value_error(
                        "AttentionNode.set_kv_cache_counter: needs kv_cache_size > 0");
                self.setKvCacheCounter(counter.buf);
            },
            nb::arg("counter"),
            // The counter's dtor destroys its buffer — pin it to the node.
            nb::keep_alive<1, 2>(),
            "Required before the first forward when kv_cache_size > 0: a missing "
            "counter crashes the process. Rejected when kv_cache_size is 0."
        );
}
PYVAI_NODE_BINDING(bindAttention);

}   // namespace pyvai
