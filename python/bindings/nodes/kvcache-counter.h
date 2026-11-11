#ifndef PYVAI_NODES_KVCACHE_COUNTER_H
#define PYVAI_NODES_KVCACHE_COUNTER_H

// Python-only wrapper for the uint32 KV-cache token counter, which the C++ models
// keep as a raw eva::Buffer. It is not a node, but the KV-aware nodes take one.
#include "nodes.h"
#include "vai/nodes/node-common.h"   // eva::Buffer, vai::getVaiDevice()

namespace pyvai {

struct KVCacheCounter
{
    eva::Buffer buf;

    KVCacheCounter() = default;
    explicit KVCacheCounter(eva::Buffer b) : buf(b) {}

    uint32_t& count() { return *reinterpret_cast<uint32_t*>(buf.map()); }

    // Nodes pin the counter with keep_alive, so this never runs under a live node.
    ~KVCacheCounter() { buf.destroy(); }
    KVCacheCounter(const KVCacheCounter&) = delete;
    KVCacheCounter& operator=(const KVCacheCounter&) = delete;
};

void bindKVCacheCounter(nb::module_& m);

}   // namespace pyvai

#endif // PYVAI_NODES_KVCACHE_COUNTER_H
