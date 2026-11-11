#include "nodes.h"
#include "kvcache-counter.h"
#include "vai/nodes/token-embedding.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindTokenEmbedding(nb::module_& m)
{
    nb::class_<vai::node::TokenEmbedding, vai::Node>(m, "TokenEmbeddingNode")
        .def(
            "__init__",
            [](vai::node::TokenEmbedding* self, uint32_t vocab_size, uint32_t embed_dim, uint32_t pos_table_size,
               float scale, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("TokenEmbeddingNode");
                new (self) vai::node::TokenEmbedding(vai::node::TokenEmbedding::Config{
                    .vocabSize = vocab_size,
                    .embedDim = embed_dim,
                    .posTableSize = pos_table_size,
                    .scale = scale,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("vocab_size"), nb::arg("embed_dim"), nb::arg("pos_table_size") = 0u, nb::arg("scale") = 1.0f,
            nb::arg("o_dtype") = nb::none(),   // lookup only: no input dtype
            "Token embedding: in0 is uint32 token IDs [..., T], out0 [..., T, "
            "embed_dim], and the table is the `weight` slot. pos_table_size > 0 adds "
            "row cached_count + n of a `posWeight` slot [pos_table_size, embed_dim], "
            "n restarting at each sequence, and needs set_kv_cache_counter(). scale "
            "multiplies the looked-up token embedding.")
        .def(
            "set_kv_cache_counter",
            [](vai::node::TokenEmbedding& self, KVCacheCounter& counter) { self.setKvCacheCounter(counter.buf); },
            nb::arg("counter"),
            // The counter's dtor destroys its buffer — pin it to the node.
            nb::keep_alive<1, 2>(),
            "Bind the KV-cache token counter whose cached_count offsets the "
            "`posWeight` rows. Required before the first forward when "
            "pos_table_size > 0.");
}

}   // namespace pyvai
