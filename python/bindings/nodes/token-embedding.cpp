#include "python/bindings/nodes/bind-node.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/token-embedding.h"

namespace pyvai {

static void bindTokenEmbedding(nb::module_& m)
{
    using C = vai::node::TokenEmbedding::Config;
    bindNode<vai::node::TokenEmbedding>(m,
                                        "TokenEmbeddingNode",
                                        "Token embedding: in0 is uint32 token IDs [..., T], out0 [..., T, "
                                        "embed_dim], and the table is the `weight` slot. pos_table_size > 0 adds "
                                        "row cached_count + n of a `posWeight` slot [pos_table_size, embed_dim], "
                                        "n restarting at each sequence, and needs set_kv_cache_counter(). scale "
                                        "multiplies the looked-up token embedding.",
                                        required("vocab_size", &C::vocabSize),
                                        required("embed_dim", &C::embedDim),
                                        field("pos_table_size", &C::posTableSize),
                                        field("scale", &C::scale))
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
PYVAI_NODE_BINDING(bindTokenEmbedding);

}   // namespace pyvai
