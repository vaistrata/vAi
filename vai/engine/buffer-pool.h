#ifndef VAI_ENGINE_BUFFER_POOL_H
#define VAI_ENGINE_BUFFER_POOL_H

#include "eva-runtime.h"
#include <map>
#include <unordered_map>
#include <cstdio>
#include <memory>

namespace vai {

struct BufferPoolConfig {
    eva::Device device;
    uint32_t logLevel = 0;  // 0: off, 1: basic, 2: verbose
};


class BufferPool;

struct PooledBuffer
{
    eva::Buffer buffer;
    std::shared_ptr<BufferPool> owner;

    PooledBuffer() = default;
    PooledBuffer(eva::Buffer buf, std::shared_ptr<BufferPool> pool) : buffer(std::move(buf)), owner(std::move(pool)) {}

    // Unique ownership: eva::Buffer is a non-owning handle, so copying would let
    // two PooledBuffers return the same buffer to the pool twice. Move-only + a
    // releasing destructor make ownership unambiguous and reclaim automatic.
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;

    PooledBuffer(PooledBuffer&& other) noexcept
    : buffer(other.buffer), owner(std::move(other.owner))
    {
        other.buffer = eva::Buffer{};
    }

    PooledBuffer& operator=(PooledBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();  // return our current buffer before taking over the other's
            buffer = other.buffer;
            owner = std::move(other.owner);
            other.buffer = eva::Buffer{};
        }
        return *this;
    }

    ~PooledBuffer() { release(); }

    operator bool() const { return (bool)buffer; }
    void release();
};


class BufferPool : public std::enable_shared_from_this<BufferPool>
{
    friend struct PooledBuffer;

    eva::Device device_;

    std::unordered_map<
        eva::BUFFER_USAGE,
        std::multimap<size_t, std::pair<eva::Buffer, eva::MEMORY_PROPERTY>>> bufferPool_;

    size_t totalAllocated_ = 0;

public:
    uint32_t logLevel;

    BufferPool(BufferPoolConfig config)
    : device_(config.device), logLevel(config.logLevel) {}

    PooledBuffer requestBuffer(
        eva::BUFFER_USAGE usageFlags,
        eva::MEMORY_PROPERTY reqMemProps,
        size_t minSize,
        size_t maxSize = size_t(-1)
    )
    {
        minSize = (minSize + 3u) & ~size_t(3);
        if (maxSize < minSize)   // keep maxSize a valid upper bound (and != sentinel)
            maxSize = minSize;

        auto& subPool = bufferPool_[usageFlags];

        if (logLevel >= 1)
            std::printf("[BufferPool] ");
        if (logLevel >= 2)
        {
            if (subPool.empty())
                std::printf("(empty)");
            else {
                std::printf("{ ");
                for (auto& [sz, _] : subPool)
                   std::printf("%zu, ", sz);
                std::printf("}");
            }
            std::printf("\n");

            std::printf("request: min=%zu, max=%zu\n", minSize, maxSize);
        }

        for (auto it = subPool.lower_bound(minSize);
            it != subPool.end() && it->first <= maxSize;
            ++it)
        {
            const auto& [buffer, memProps] = it->second;
            if (eva::hasFlag(memProps, reqMemProps))
            {
                eva::Buffer result = std::move(it->second.first);
                subPool.erase(it);
                if (logLevel >= 1)
                    std::printf("=> reuse %zu bytes\n", result.size());
                return PooledBuffer{std::move(result), shared_from_this()};
            }
        }

        // Create new buffer
        totalAllocated_ += minSize;
        if (logLevel >= 1)
            std::printf("=> alloc %zu bytes (total: %zu)\n", minSize, totalAllocated_);
        return PooledBuffer{
            device_.createBuffer({
                .size = minSize,
                .usage = usageFlags,
                .reqMemProps = reqMemProps,
            }),
            shared_from_this(),
        };
    }
};


inline void PooledBuffer::release()
{
    if (buffer && owner)
    {
        auto memProps = buffer.memoryProperties();
        auto size = buffer.size();
        auto usage = buffer.usage();

        owner->bufferPool_[usage].emplace(
            size,
            std::make_pair(std::move(buffer), memProps)
        );
    }
    buffer = eva::Buffer{};
    owner.reset();
}

} // namespace vai

#endif   // VAI_ENGINE_BUFFER_POOL_H
