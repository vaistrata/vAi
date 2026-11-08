#include "neural-net.h"
#include "graph-controller.h"
#include "node-profiler.h"
#include <stdexcept>
#include <queue>
#include <optional>
#include <unordered_set>
#include <unordered_map>


namespace vai {

// Process-wide device used when a net does not provide its own.
static eva::Device gVaiDevice;

eva::DescriptorPool gVaiDestSetPool;


void initVAIEngine(eva::Device device)
{
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

    gVaiDevice = device ? device : eva::Runtime::get().device({.enableComputeQueues = true});

    gVaiDestSetPool = gVaiDevice.createDescriptorPool({
        .maxTypes = {
            eva::DESCRIPTOR_TYPE::STORAGE_BUFFER <= 10000,
            eva::DESCRIPTOR_TYPE::UNIFORM_BUFFER <= 1000
        },
        .maxSets = 5000
    });
}


eva::Device getVaiDevice()
{
    if (!gVaiDevice) 
        initVAIEngine();
    return gVaiDevice;
}


// TensorUploader
TensorUploader::TensorUploader(eva::Device& device)
: TensorUploader(device, Config{})
{}

TensorUploader::TensorUploader(eva::Device& device, Config config)
: device_(device)
, config_(config)
{
    chunkSizeBytes_ = detail::alignUp(config.chunkSizeMiB * detail::kMiBToBytes, config.alignment);

    uploadBuffer_ = device_.createBuffer({
        .size = chunkSizeBytes_,
        .usage = eva::BUFFER_USAGE::TRANSFER_SRC,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
    });

    cmdBuffer_ = device_.newCommandBuffer(
        eva::queue_compute,
        eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);

    // Keep the staging buffer mapped for the uploader lifetime.
    mappedPtr_ = uploadBuffer_.map();
    if (!mappedPtr_) {
        uploadBuffer_.destroy();
        throw std::runtime_error("TensorUploader: Failed to map upload buffer");
    }

    cmdBuffer_.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
}

TensorUploader::~TensorUploader() noexcept
{
    try {
        flush();
    } catch (...) {
        // Destruction must not propagate upload errors.
    }
    uploadBuffer_.destroy();
}

void TensorUploader::upload(const void* hostData, size_t byteSize, vai::PooledBuffer& dstBuffer)
{
    const uint8_t* srcPtr = static_cast<const uint8_t*>(hostData);
    size_t remaining = byteSize;
    size_t dstOffset = 0;

    while (remaining > 0)
    {
        size_t available = chunkSizeBytes_ - currentOffset_;

        // Flush when the remaining space cannot satisfy the next aligned copy.
        if (available < config_.alignment)
        {
            flushAndReset();
            available = chunkSizeBytes_;
        }

        // chunkSizeBytes_ is aligned, so aligning copySize cannot cross the buffer end.
        size_t copySize = std::min(remaining, available);

        VAI_ASSERT(copySize > 0);
        VAI_ASSERT(currentOffset_ + copySize <= chunkSizeBytes_);

        
        std::memcpy(mappedPtr_ + currentOffset_, srcPtr, copySize);

        
        cmdBuffer_.copyBuffer(
            uploadBuffer_(currentOffset_, copySize),
            dstBuffer.buffer(dstOffset, copySize));

        
        currentOffset_ += detail::alignUp(copySize, config_.alignment);
        srcPtr += copySize;
        dstOffset += copySize;
        remaining -= copySize;
        hasPendingCopies_ = true;
    }
}

void TensorUploader::flush()
{
    if (hasPendingCopies_) 
    {
        cmdBuffer_.end().submit().wait();
        cmdBuffer_.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
        currentOffset_ = 0;
        hasPendingCopies_ = false;
    }
}

void TensorUploader::flushAndReset()
{
    flush();
    currentOffset_ = 0;
}


// NeuralNet
// The GPU may still be reading buffers this net owns, so the last submit has to
// land before any of them is released.
NeuralNet::~NeuralNet()
{
    if (syncedCount_ != submitCount_)
        timeline_.wait(submitCount_);
}

NeuralNet::NeuralNet(const NeuralNetConfig& config)
: device_(config.device ? config.device : getVaiDevice())
, config_(config)
{
    NeuralNet::currentNet_ = this;

    controller_ = std::make_unique<GraphController>();
    profiler_ = std::make_unique<NodeProfiler>(device_);

    if (config.bufferPool)
        bufferPool_ = config.bufferPool;
    else
        bufferPool_ = std::make_shared<vai::BufferPool>(
            vai::BufferPoolConfig{device_, config.bufferPoolLogLevel});

    if (config.pipelinePool)
        pipelinePool_ = config.pipelinePool;
    else
        pipelinePool_ = std::make_shared<vai::PipelinePool>(device_);

    cmdBuffers_.push_back(device_.newCommandBuffer(
        eva::queue_compute,
        eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER));

    for (auto& uploadCmd : uploadCmds_)
        uploadCmd = device_.newCommandBuffer(
            eva::queue_compute,
            eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER);

    timeline_ = device_.createTimelineSemaphore();

    VAI_ASSERT(config_.wDefaultDtype != Dtype::None);
    VAI_ASSERT(config_.oDefaultDtype != Dtype::None);
}

void NeuralNet::setProfileTag(const Node* node, const std::string& tag) const
{
    if (profiler_->setTag(node, tag))
        invalidateCommandBuffer();
}

const std::string& NeuralNet::profileTagOf(const Node* node) const
{
    return profiler_->tagOf(node);
}

void NeuralNet::clearProfileTags()
{
    if (profiler_->clearTags())
        invalidateCommandBuffer();
}

void NeuralNet::sync()
{
    if (syncedCount_ == submitCount_)
        return;

    // NOTE: a device lost during a submit may still signal the timeline
    // successfully. In that case the loss crashes the runtime at the next submit.
    timeline_.wait(submitCount_);
    syncedCount_ = submitCount_;

    profiler_->collect();   // the timestamps are valid only now
}

void NeuralNet::prepareInputs()
{
    const uint64_t gen = submitCount_ + 1;

    const uint64_t completed = timeline_.value();
    std::erase_if(pendingStaging_, [completed](const auto& e) { return e.first <= completed; });

    eva::CommandBuffer uploadCmd = uploadCmds_[gen % detail::kInputUploadRing];
    std::vector<PooledBuffer> staging;

    for (InputNode* node : activeInputs_)
    {
        Tensor& in = (*node)["in0"];

        const bool bound = in.gpuBufferBound();
        const bool data  = in.hasHostData() && !in.hostUploaded() && !in.lastSubmission().second;

        VAI_ASSERT((bound || data) && "graph input has neither host data nor a device buffer");

        if (!data)
            continue;

        const size_t bytes = in.numBytes();

        if (!bound)
            in.bindBuffer(bufferPool_->requestBuffer(
                eva::BUFFER_USAGE::STORAGE_BUFFER
                    | eva::BUFFER_USAGE::TRANSFER_SRC
                    | eva::BUFFER_USAGE::TRANSFER_DST,
                eva::MEMORY_PROPERTY::DEVICE_LOCAL,
                bytes, nextPow2(bytes)));

        if (staging.empty())
        {
            if (gen > detail::kInputUploadRing)
                timeline_.wait(gen - detail::kInputUploadRing);
            uploadCmd.begin(eva::COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT);
        }

        PooledBuffer src = bufferPool_->requestBuffer(
            eva::BUFFER_USAGE::TRANSFER_SRC,
            eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT,
            bytes, nextPow2(bytes));

        std::memcpy(src.buffer.map(), in.hostData(), bytes);
        uploadCmd
            .barrier(eva::PIPELINE_STAGE::COMPUTE_SHADER / eva::PIPELINE_STAGE::TRANSFER)
            // .barrier(eva::SYNC_SCOPE::COMPUTE_READ / in.buffer() / eva::SYNC_SCOPE::TRANSFER_DST)
            .copyBuffer(src.buffer, in.buffer().buffer, 0, in.byteOffset(), bytes)
            .barrier(eva::SYNC_SCOPE::TRANSFER_DST / in.buffer() / eva::SYNC_SCOPE::COMPUTE_READ);
        staging.push_back(std::move(src));

        // The host copy stays readable; only a setter makes it new again.
        in.markHostUploaded();
    }

    if (!staging.empty())
    {
        uploadCmd.end().submit();
        pendingStaging_.emplace(gen, std::move(staging));
    }
}

void NeuralNet::sortNodes()
{
    std::unordered_map<Node*, std::vector<Node*>> outNodes;
    std::unordered_map<Node*, std::vector<Node*>> inNodes;
    std::unordered_map<Node*, uint32_t> inDegree;
    std::unordered_set<Node*> visited;

    auto dfs = [&](auto& self, Node& node) -> void
    {
        if (!visited.insert(&node).second)
            return;

        for (auto& [name, slot] : node.slots_)
        {
            if (slot.type() != NodeSlot::output)
                continue;

            for (NodeSlot* connected : slot.connections_)
            {
                Node& nextNode = connected->parent_;
                outNodes[&node].push_back(&nextNode);
                inNodes[&nextNode].push_back(&node);
                inDegree[&nextNode]++;
                self(self, nextNode);
            }
        }
    };

    for (auto& [name, inputNode] : inputs_)
    {
        inDegree[&inputNode] = 0; // Explicitly seed graph roots.
        inNodes[&inputNode] = {}; 
        dfs(dfs, inputNode);
    }

    sortedNodes_.clear();

    // Topologically order the reachable graph with Kahn's algorithm.
    std::queue<Node*> q;

    for (const auto& [node, deg] : inDegree)
    {
        if (deg == 0)
            q.push(node);
    }

    while (!q.empty())
    {
        Node* curr = q.front(); q.pop();
        sortedNodes_.push_back(curr);

        for (Node* next : outNodes[curr])
        {
            if (--inDegree[next] == 0)
                q.push(next);
        }
    }

    if (sortedNodes_.size() != visited.size())
        throw std::runtime_error("Cycle detected in graph!");

    std::vector<const Node*> totalInputs;
    totalInputs.reserve(inputs_.size());
    for (auto& [name, inputNode] : inputs_)
        totalInputs.push_back(&inputNode);

    controller_->build(sortedNodes_, totalInputs);

    // Wiring errors, like the cycle above: caught once per graph build, so they
    // cost nothing to check in every configuration.
    if (!controller_->fanInIsGateMergeOnly())
        throw std::runtime_error("Input slot with two producers outside a gate merge!");
}

void NeuralNet::uploadModelParams()
{
    // Built on the first tensor that needs it: a graph whose weights are already
    // up must not allocate a chunkSizeMiB staging buffer for nothing.
    std::optional<TensorUploader> uploader;

    for (Node* node : sortedNodes_)
    {
        for (auto& [name, slot] : node->slots_)
        {
            if (!(slot.type() == NodeSlot::persistent))
                continue;

            Tensor& tensor = slot.getValueRef();
            if (tensor.gpuBufferBound())
                continue;

            VAI_ASSERT(tensor.validShape());
            VAI_ASSERT(slot.connections_.size() == 0);

            const size_t byteSize = tensor.numBytes();

            vai::PooledBuffer pooledBuffer = bufferPool_->requestBuffer(
                eva::BUFFER_USAGE::STORAGE_BUFFER
                | eva::BUFFER_USAGE::TRANSFER_DST,
                eva::MEMORY_PROPERTY::DEVICE_LOCAL,
                byteSize,
                byteSize
            );

            if (tensor.hasHostData())
            {
                if (!uploader)
                    uploader.emplace(device_, TensorUploader::Config{
                        .chunkSizeMiB = config_.uploadChunkSizeMiB,
                        .alignment = detail::kBufferCopyAlignment
                    });

                uploader->upload(tensor.hostData(), byteSize, pooledBuffer);

                tensor.clearHostData(true);
            }

            tensor.bindBuffer(std::move(pooledBuffer));
        }
    }
}

Tensor NeuralNet::paramTensor(const std::vector<uint32_t>& shape, Dtype dtype,
                              const std::function<void(void*)>& fill)
{
    Tensor tensor(shape, dtype);
    const size_t byteSize = tensor.numBytes();
    if (device_.deviceType() != eva::DEVICE_TYPE::INTEGRATED_GPU)
    {
        tensor.data_->hostData_.resize(byteSize);
        fill(tensor.data_->hostData_.data());
        return tensor;
    }

    vai::PooledBuffer pooledBuffer = bufferPool_->requestBuffer(
        eva::BUFFER_USAGE::STORAGE_BUFFER | eva::BUFFER_USAGE::TRANSFER_DST,
        eva::MEMORY_PROPERTY::DEVICE_LOCAL | eva::MEMORY_PROPERTY::HOST_VISIBLE
        | eva::MEMORY_PROPERTY::HOST_COHERENT,
        byteSize,
        byteSize);
    fill(pooledBuffer.buffer.map());
    pooledBuffer.buffer.unmap();
    tensor.bindBuffer(std::move(pooledBuffer));
    return tensor;
}

void NeuralNet::recordCommands(const TensorRefDict& outputs)
{
    sync();

    std::map<const NodeSlot*, Tensor*> destinations;
    for (auto& [name, dst] : outputs)
    {
        const NodeSlot& in = outputs_.at(name).slot("in0");
        for (NodeSlot* producer : in.connections())
            destinations[producer] = dst;
    }

    controller_->beginRecord(activeInputs_);

    for (auto& [name, node] : outputs_)
        node.reached_ = false;

    recordedCmdBuffers_ = 1;
    eva::CommandBuffer cmdBuffer = cmdBuffers_[0];
    uint32_t nodesInCmdBuffer = 0;

    cmdBuffer.begin(eva::COMMAND_BUFFER_USAGE::SIMULTANEOUS_USE);
    profiler_->beginRecord(cmdBuffer);

    for (size_t nodeIdx = 0; nodeIdx < sortedNodes_.size(); ++nodeIdx)
    {
        if (!controller_->hasNeedsFed(nodeIdx))
        {
            for (auto& [name, slot] : sortedNodes_[nodeIdx]->slots_)
                if (slot.type() != NodeSlot::persistent)
                    slot.getValueRef() = Tensor();
            continue;
        }

        if (config_.nodesPerSubmit && nodesInCmdBuffer == config_.nodesPerSubmit)
        {
            cmdBuffer.end();
            if (recordedCmdBuffers_ == cmdBuffers_.size())
                cmdBuffers_.push_back(device_.newCommandBuffer(
                    eva::queue_compute,
                    eva::COMMAND_POOL_CREATE::RESET_COMMAND_BUFFER));
            cmdBuffer = cmdBuffers_[recordedCmdBuffers_++];
            cmdBuffer.begin(eva::COMMAND_BUFFER_USAGE::SIMULTANEOUS_USE);
            nodesInCmdBuffer = 0;
        }
        ++nodesInCmdBuffer;

        Node* node = sortedNodes_[nodeIdx];
        const bool profileTarget = profiler_->isTagged(node);

        // prepare() validates inputs and determines tensors needed by this recording.
        node->prepare();

        for (auto& [name, slot] : node->slots_)
        {
            Tensor& tensor = slot.getValueRef();

            if (slot.type() == NodeSlot::persistent)
            {
                // Persistent tensors must already reside on the device.
                VAI_ASSERT(tensor.gpuBufferBound());
            }
            else if (slot.type() == NodeSlot::input)
            {
                // Nothing to require of an input here: a wired slot is not
                // necessarily read. A gate leaves one branch out of the
                // recording, and a partialInputs node takes whichever inputs
                // arrived. Tensor::buffer() carries the requirement instead,
                // where the node reads the slot and its own contract applies.
            }
            else
            {
                // Temporary slots may stay unassigned (e.g. a split-K partial
                // buffer when this shape does not split).
                if (slot.type() == NodeSlot::temporary && !tensor.validShape())
                    continue;

                if (!tensor.gpuBufferBound())
                {
                    size_t byteSize = tensor.numBytes();

                    auto dstIt = destinations.find(&slot);
                    Tensor* dst = (dstIt == destinations.end()) ? nullptr : dstIt->second;

                    if (dst && dst->gpuBufferBound() && dst->baseOffset() == 0 &&
                        dst->dtype() == tensor.dtype() &&
                        dst->buffer().buffer.size() >= byteSize)
                        tensor.data_ = dst->data_;
                    else
                        tensor.bindBuffer(bufferPool_->requestBuffer(
                            eva::BUFFER_USAGE::STORAGE_BUFFER
                            | eva::BUFFER_USAGE::TRANSFER_SRC // TODO: only needed for final output tensors, so we should remove for general case.
                            | eva::BUFFER_USAGE::TRANSFER_DST // Keep this allocation in the same buffer-pool cache group as transfer destinations.
                            ,
                            eva::MEMORY_PROPERTY::DEVICE_LOCAL,
                            byteSize,
                            byteSize * 2.5f
                        ));
                }

                
                if (slot.type() == NodeSlot::output)
                {
                    for (NodeSlot* connected : slot.connections_)
                        connected->getValueRef() = tensor;
                }
            }
        }

        
        if (profileTarget)
            profiler_->beginNode(cmdBuffer, node);

        node->run(cmdBuffer);

        if (profileTarget)
            profiler_->endNode(cmdBuffer);

        // Release transient bindings so their pooled buffers can be reused.
        for (auto& [name, slot] : node->slots_)
        {
            if (slot.type() == NodeSlot::persistent)
                continue;

            slot.getValueRef() = Tensor();
        }
    }

    profiler_->endRecord(cmdBuffer);
    cmdBuffer.end();
}


void NeuralNet::resetProfileRecords()
{
    // Flush any in-flight forward (e.g. the last warm-up) first, so its timestamps
    // are consumed here instead of leaking into the post-reset accumulation.
    sync();
    profiler_->reset();
}

double NeuralNet::getGpuTimeMs(const std::string& tag)
{
    sync();
    return profiler_->ms(tag);
}

void NeuralNet::printProfile(bool withResetRecords)
{
    sync();  // Include timestamps from the latest in-flight forward.
    profiler_->print(withResetRecords);
}


InputNode::InputNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void InputNode::prepare()
{
    Tensor& in = (*this)["in0"];
    (*this)["out0"] = in;   // Consumers read directly from the input buffer.

    // operator() re-records when any of these recorded bindings changes.
    recordedShape_ = in.shape();
    recordedInputBuffer_ = in.buffer().buffer;
    recordedInputBaseOffset_ = in.baseOffset();
}


OutputNode::OutputNode()
{
    addSlot("in0", NodeSlot::input);
    // addSlot("out0", NodeSlot::output);
}

void OutputNode::prepare()
{
    VAI_ASSERT((*this)["in0"].validShape());
}

void OutputNode::run(eva::CommandBuffer)
{
    // Preserve the graph output before slot tensors are cleared after recording.
    cachedOutput_ = std::move((*this)["in0"]);
    reached_ = true;
}


} // namespace vai
