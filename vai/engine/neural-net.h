#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "tensor.h"
#include "pipeline-pool.h"
#include <map>
#include <memory>

namespace vai {

namespace detail {

inline constexpr size_t kMiBToBytes = 1024 * 1024;
inline constexpr size_t kDefaultUploadChunkMiB = 256;
inline constexpr size_t kBufferCopyAlignment = 4;

inline constexpr uint64_t kInputUploadRing = 3;

template<typename T>
[[nodiscard]] constexpr T alignUp(T value, T alignment) noexcept {
    static_assert(std::is_unsigned_v<T>, "alignUp requires unsigned type");
    return (value + alignment - 1) & ~(alignment - 1);
}

template<typename T>
[[nodiscard]] constexpr T alignDown(T value, T alignment) noexcept {
    static_assert(std::is_unsigned_v<T>, "alignDown requires unsigned type");
    return value & ~(alignment - 1);
}

}

class TensorUploader
{
public:
    struct Config {
        size_t chunkSizeMiB = detail::kDefaultUploadChunkMiB;
        size_t alignment = detail::kBufferCopyAlignment;
    };

    explicit TensorUploader(eva::Device& device, Config config);
    explicit TensorUploader(eva::Device& device);
    ~TensorUploader() noexcept;

    TensorUploader(const TensorUploader&) = delete;
    TensorUploader& operator=(const TensorUploader&) = delete;
    TensorUploader(TensorUploader&&) = delete;
    TensorUploader& operator=(TensorUploader&&) = delete;

    void upload(const void* hostData, size_t byteSize, vai::PooledBuffer& dstBuffer);

    void flush();

    [[nodiscard]] bool hasPending() const noexcept { return hasPendingCopies_; }

private:
    void flushAndReset();

    eva::Device& device_;
    Config config_;
    size_t chunkSizeBytes_;

    eva::Buffer uploadBuffer_;
    uint8_t* mappedPtr_ = nullptr;
    eva::CommandBuffer cmdBuffer_;

    size_t currentOffset_ = 0;
    bool hasPendingCopies_ = false;
};


class Node;
class NodeSlot;
class NeuralNet;
struct NodeFlow;

using TensorDict = std::map<std::string, Tensor>;

using TensorRefDict = std::map<std::string, Tensor*>;


class NodeSlot
{
    friend class NeuralNet;

public:
    enum Type { input, output, temporary, persistent, none };

private:
    Type type_;
    Node& parent_;
    std::vector<NodeSlot*> connections_;

    Tensor value_;

public:
    NodeSlot(Type type, Node* parent) : type_(type), parent_(*parent) {}

    Type type() const
    {
        return type_;
    }

    const Node& parent() const
    {
        return parent_;
    }

    const std::vector<NodeSlot*>& connections() const
    {
        return connections_;
    }

    Tensor& getValueRef()
    {
        return value_;
    }

    const Tensor& getValueRef() const
    {
        return value_;
    }

    static void connect(NodeSlot& a, NodeSlot& b)
    {
        VAI_ASSERT(a.type_ == output && b.type_ == input);
        for (NodeSlot* s : a.connections_)
            if (s == &b) return;
        a.connections_.push_back(&b);
        b.connections_.push_back(&a);
    }

    static void disconnect(NodeSlot& a, NodeSlot& b)
    {
        std::erase(a.connections_, &b);
        std::erase(b.connections_, &a);
    }
};


class Node
{
    friend class NeuralNet;
    std::map<std::string, NodeSlot> slots_;

public:
    enum Flag : uint32_t {
        gate = 1u << 0,

        partialInputs = 1u << 1,
    };

protected:
    const NeuralNet* ownedNet_ = nullptr;
    uint32_t flags_ = 0;

    void addSlot(const std::string& name, NodeSlot::Type type)
    {
        slots_.try_emplace(name, type, this);
    }

public:
    Node();

    eva::Device device() const;
    vai::Dtype wDefaultDtype() const;
    vai::Dtype oDefaultDtype() const;
    PipelinePool& pipelinePool() const;

    const std::string& getProfileTag() const;

    void setProfileTag(const std::string& tag);

    uint32_t flags() const
    {
        return flags_;
    }

    const std::map<std::string, NodeSlot>& slots() const
    {
        return slots_;
    }

    NodeSlot& slot(const std::string& name)
    {
        return slots_.at(name);
    }

    Tensor& operator[](const std::string& name)
    {
        return slots_.at(name).getValueRef();
    }

    const Tensor& operator[](const std::string& name) const
    {
        return slots_.at(name).getValueRef();
    }

    operator NodeFlow();

    virtual void prepare() = 0;
    virtual void run(eva::CommandBuffer cmdBuff) = 0;
    virtual ~Node() = default;

    const NodeSlot* findSlot(const std::string& slotName) const
    {
        auto it = slots_.find(slotName);
        if (it == slots_.end())
            return nullptr;
        return &it->second;
    }

};


class InputNode : public Node
{
    friend class NeuralNet;
    std::vector<uint32_t> recordedShape_;
    eva::Buffer recordedInputBuffer_;
    uint32_t recordedInputBaseOffset_ = 0;

public:
    InputNode();
    void prepare() override;
    void run(eva::CommandBuffer) override {}
};


class OutputNode : public Node
{
    friend class NeuralNet;
    Tensor cachedOutput_;
    bool reached_ = false;

public:
    OutputNode();
    void prepare() override;
    void run(eva::CommandBuffer) override;
};


void initVAIEngine(eva::Device device=eva::Device{});

eva::Device getVaiDevice();


struct NeuralNetConfig
{
    // Device that owns this net's buffers, pipelines, and command buffer.
    // Uses the process-wide getVaiDevice() when left null.
    eva::Device device = {};

    // Weight-upload chunk size in MiB. Chunking avoids large HOST_VISIBLE allocations
    // that may exceed driver or system limits; full chunks are flushed and reused.
    size_t uploadChunkSizeMiB = detail::kDefaultUploadChunkMiB;

    std::shared_ptr<vai::BufferPool> bufferPool = nullptr;
    uint32_t bufferPoolLogLevel = 0;  // 0: off, 1: basic, 2: verbose.

    // Shared pipeline cache. When null, the net creates its own pool.
    std::shared_ptr<vai::PipelinePool> pipelinePool = nullptr;

    // Fallback dtype for parameters such as weights and KV cache.
    // Layers may override it; Dtype::None is not allowed.
    Dtype wDefaultDtype = Dtype::Fp16;

    // Fallback dtype for layer outputs. A layer's input dtype is whatever its
    // producer wrote; Dtype::None is not allowed.
    Dtype oDefaultDtype = Dtype::Fp32;

    // Nodes per queue submission; 0 records the whole forward into one.
    // Some drivers reset the GPU when one submission runs too long (Adreno
    // after a few seconds), so a long forward there needs this set.
    uint32_t nodesPerSubmit = 0;
};


class NeuralNet
{
    friend class Node;
    friend class InputNode;
    friend class OutputNode;

    thread_local inline static NeuralNet* currentNet_ = nullptr;
    eva::Device device_;

    // One forward's commands, split every NeuralNetConfig::nodesPerSubmit
    // nodes; the first recordedCmdBuffers_ hold the current recording.
    std::vector<eva::CommandBuffer> cmdBuffers_;
    size_t recordedCmdBuffers_ = 0;

    std::array<eva::CommandBuffer, detail::kInputUploadRing> uploadCmds_;
    std::map<uint64_t, std::vector<PooledBuffer>> pendingStaging_;

    NeuralNetConfig config_;
    std::vector<Node*> sortedNodes_;

    std::map<std::string, InputNode> inputs_;
    std::map<std::string, OutputNode> outputs_;

    std::vector<InputNode*> activeInputs_;

    bool needToRebuildGraph_ = true;
    bool needToUploadParams_ = true;
    mutable bool needToRerecordCommands_ = true;

    std::unique_ptr<class GraphController> controller_;

    eva::TimelineSemaphore timeline_;
    uint64_t submitCount_ = 0;
    uint64_t syncedCount_ = 0;

    std::shared_ptr<vai::BufferPool> bufferPool_;
    std::shared_ptr<vai::PipelinePool> pipelinePool_;

    std::unique_ptr<class NodeProfiler> profiler_;

public:
    static NeuralNet* current() { return currentNet_; }

    NeuralNet(const NeuralNetConfig& config = {});

    ~NeuralNet();

    void sync();

    const NeuralNetConfig& config() const { return config_; }

    eva::Device device() const { return device_; }

    PipelinePool& pipelinePool() const { return *pipelinePool_; }

    InputNode& input(const std::string& name)
    {
        return inputs_.try_emplace(name).first->second;
    }

    OutputNode& output(const std::string& name)
    {
        return outputs_.try_emplace(name).first->second;
    }

    void invalidateCommandBuffer() const
    {
        needToRerecordCommands_ = true;
    }

    void setProfileTag(const Node* node, const std::string& tag) const;
    const std::string& profileTagOf(const Node* node) const;

    void sortNodes();

    void prepareInputs();

    void recordCommands(const TensorRefDict& outputs);

    void uploadModelParams();

    void resetProfileRecords();
    void printProfile(bool withResetRecords = true);
    double getGpuTimeMs(const std::string& tag);

    void clearProfileTags();

    void operator()(TensorDict inputs, TensorRefDict outputs, bool syncPrev = true)
    {
        VAI_ASSERT(!inputs.empty());

        std::vector<InputNode*> active;
        active.reserve(inputs.size());
        for (const auto& [name, tensor] : inputs)
        {
            auto found = inputs_.find(name);
            VAI_ASSERT(found != inputs_.end());
            found->second["in0"] = tensor;
            active.push_back(&found->second);
        }

        if (active != activeInputs_)
        {
            activeInputs_ = std::move(active);
            needToRerecordCommands_ = true;
        }

        if (syncPrev)
            sync();

        prepareInputs();

        for (InputNode* node : activeInputs_)
        {
            Tensor& in = (*node)["in0"];
            if (node->recordedShape_ != in.shape() ||
                node->recordedInputBuffer_ != in.buffer().buffer ||
                node->recordedInputBaseOffset_ != in.baseOffset())
                needToRerecordCommands_ = true;
        }

        for (auto& [name, dst] : outputs)
        {
            VAI_ASSERT(dst);
            if (dst->data_ != outputs_.at(name).cachedOutput_.data_)
                needToRerecordCommands_ = true;
        }

        if (needToRebuildGraph_)
        {
            sortNodes();
            needToRebuildGraph_ = false;
            needToUploadParams_ = true;
        }

        if (needToUploadParams_)
        {
            uploadModelParams();
            needToUploadParams_ = false;
            needToRerecordCommands_ = true;
        }

        if (needToRerecordCommands_)
        {
            recordCommands(outputs);
            needToRerecordCommands_ = false;
        }

        // Separate submissions, so a driver times each part on its own. The
        // nodes' own barriers order them across command buffers.
        for (size_t i = 0; i + 1 < recordedCmdBuffers_; ++i)
            device_.queue() << cmdBuffers_[i];
        device_.queue() << cmdBuffers_[recordedCmdBuffers_ - 1] / timeline_(++submitCount_);

        for (auto& [name, node] : outputs_)
            if (node.reached_)
                node.cachedOutput_.data_->lastSubmission_ = {this, submitCount_};

        for (auto& [name, dst] : outputs)
        {
            OutputNode& node = outputs_.at(name);
            *dst = node.reached_ ? node.cachedOutput_ : Tensor{};
        }
    }
};

inline Node::Node() : ownedNet_(NeuralNet::current()) {}

inline eva::Device Node::device() const { return ownedNet_->device_; }
inline vai::Dtype Node::wDefaultDtype() const { return ownedNet_->config().wDefaultDtype; }
inline vai::Dtype Node::oDefaultDtype() const { return ownedNet_->config().oDefaultDtype; }
inline PipelinePool& Node::pipelinePool() const { return ownedNet_->pipelinePool(); }

inline void Node::setProfileTag(const std::string& tag) { ownedNet_->setProfileTag(this, tag); }
inline const std::string& Node::getProfileTag() const { return ownedNet_->profileTagOf(this); }


}

#endif
