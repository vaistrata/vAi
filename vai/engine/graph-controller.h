#ifndef VAI_ENGINE_GRAPH_CONTROLLER_H
#define VAI_ENGINE_GRAPH_CONTROLLER_H

#include "vai/nodes/gate.h"
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

namespace vai {

/////////////////////////////////////////////////////////////////////////////////////////
// GraphController - what a recording may leave out, precomputed from the wiring
//
// Two conditions retire a node, and both are answered per node by one bitmask
// test because the wiring, not the conditions, decides the table:
//
//   gate branch   which gate branches reach the node, one bit per gate. A node
//                 both branches reach carries the bit on both sides, which is
//                 what keeps a merge point and its downstream alive whichever
//                 way the gate goes.
//   input         which graph inputs the node needs named, one bit per input.
//                 Not "which inputs can reach it": a slot left without a
//                 producer stops the node, so the requirement is the AND over
//                 its wired slots of the OR over each slot's producers. That is
//                 a monotone condition, held here as the minimal input sets that
//                 satisfy it — one set for every node a fan-in does not feed.
//
// build() runs once per graph; beginRecord() freezes the two conditions a
// recording bakes in.
//
// NeuralNet holds this behind a pointer, so this header reaches only the one
// translation unit that records commands.
/////////////////////////////////////////////////////////////////////////////////////////
class GraphController
{
    struct Reach { uint64_t thenBits = 0, elseBits = 0; };

    std::vector<const node::Gate*> gates_;
    std::unordered_map<const Node*, size_t> index_;
    std::vector<Reach> reach_;

    std::unordered_map<const Node*, uint64_t> inputMask_;
    std::vector<std::vector<uint64_t>> listOfNeededInputs_;

    uint64_t thenDead_ = 0, elseDead_ = 0;   // beginRecord() snapshot
    uint64_t currentInputs_ = 0;

    bool isGateMerge(const NodeSlot& in) const;

    // Which branch of each gate this recording leaves out.
    void snapshotGates();

    // Keeps `sets` to the minimal masks: a candidate that already contains one
    // of them asks for more than it must, and one that contains the candidate
    // has been superseded.
    static void addMinimal(std::vector<uint64_t>& sets, uint64_t candidate);

public:
    // `totalInputs` are every graph input node, in the order their bits take.
    // beginRecord() later names the subset a recording actually brings.
    void build(const std::vector<Node*>& sortedNodes, const std::vector<const Node*>& totalInputs);

    // A fan-in slot is legal only where one gate's two branches meet.
    bool fanInIsGateMergeOnly() const;

    // Freeze what this recording bakes in: the gate states, and the inputs
    // this forward brings.
    void beginRecord(const std::vector<InputNode*>& activeInputs);

    // Every input this node needs is produced by a branch beginRecord() kept,
    // from the inputs it named. How many it needs is the node's own business:
    // a partialInputs node is fed by one of them.
    bool hasNeedsFed(size_t nodeIdx) const;
};


/*
One sweep in topological order fills, for every node, which gate branches reach
it. Reading a node means its producers are already done, so a single pass covers
every gate at once — no per-gate traversal, and nothing to redo when a condition
flips.
*/
inline void GraphController::addMinimal(std::vector<uint64_t>& sets, uint64_t candidate)
{
    for (uint64_t m : sets)
        if ((candidate & m) == m)
            return;
    std::erase_if(sets, [candidate](uint64_t m) { return (m & candidate) == candidate; });
    sets.push_back(candidate);
}

inline void GraphController::build(const std::vector<Node*>& sortedNodes, const std::vector<const Node*>& totalInputs)
{
    gates_.clear();
    index_.clear();
    inputMask_.clear();

    // Node::gate certifies the cast: only node::Gate sets it.
    for (size_t i = 0; i < sortedNodes.size(); ++i)
    {
        index_[sortedNodes[i]] = i;
        if (sortedNodes[i]->flags() & Node::gate)
            gates_.push_back(static_cast<const node::Gate*>(sortedNodes[i]));
    }
    // One bit per gate. Past that the shift below is undefined, so this cannot
    // be an assert that a release build drops.
    if (gates_.size() > 64)
        throw std::runtime_error("More than 64 gates in graph!");
    if (totalInputs.size() > 64)
        throw std::runtime_error("More than 64 inputs in graph!");

    for (size_t i = 0; i < totalInputs.size(); ++i)
        inputMask_[totalInputs[i]] = uint64_t(1) << i;

    reach_.assign(sortedNodes.size(), {});
    listOfNeededInputs_.assign(sortedNodes.size(), {});

    size_t gateSeen = 0;
    for (size_t i = 0; i < sortedNodes.size(); ++i)
    {
        Node* node = sortedNodes[i];

        // An input asks for itself; every other node asks for what its slots ask
        // for, and a slot is satisfied by any one of its producers.
        if (auto found = inputMask_.find(node); found != inputMask_.end())
        {
            listOfNeededInputs_[i] = { found->second };
        }
        else if (node->flags() & Node::partialInputs)
        {
            // Any one input carries this node, so its sets are the union of its
            // slots' rather than their product. Not the empty set: a node that
            // gets nothing has nothing to stack, and a recording that reaches
            // none of its producers has to drop it.
            std::vector<uint64_t> anySlot;
            for (auto& [name, slot] : node->slots())
            {
                if (slot.type() != NodeSlot::input || slot.connections().empty())
                    continue;

                for (const NodeSlot* producer : slot.connections())
                    for (uint64_t m : listOfNeededInputs_[index_.at(&producer->parent())])
                        addMinimal(anySlot, m);
            }
            listOfNeededInputs_[i] = std::move(anySlot);
        }
        else
        {
            std::vector<uint64_t> neededInputs = { 0 };
            for (auto& [name, slot] : node->slots())
            {
                if (slot.type() != NodeSlot::input || slot.connections().empty())
                    continue;

                std::vector<uint64_t> anyProducer;
                for (const NodeSlot* producer : slot.connections())
                    for (uint64_t m : listOfNeededInputs_[index_.at(&producer->parent())])
                        addMinimal(anyProducer, m);

                std::vector<uint64_t> both;
                for (uint64_t a : neededInputs)
                    for (uint64_t b : anyProducer)
                        addMinimal(both, a | b);
                neededInputs = std::move(both);
            }
            listOfNeededInputs_[i] = std::move(neededInputs);
        }

        // Copied before the writes below, which may target this node's own
        // consumers but never this node.
        const Reach inherited = reach_[i];

        // A gate's own bit goes to the consumers of that one branch. Gates were
        // numbered in this same order, so counting them again lands on the bit.
        if (node->flags() & Node::gate)
        {
            const uint64_t gateBit = uint64_t(1) << gateSeen++;
            for (const NodeSlot* c : node->slot("then").connections())
                reach_[index_.at(&c->parent())].thenBits |= gateBit;
            for (const NodeSlot* c : node->slot("else").connections())
                reach_[index_.at(&c->parent())].elseBits |= gateBit;
        }

        // What this node inherited flows on through every branch it feeds.
        for (auto& [name, slot] : node->slots())
        {
            if (slot.type() != NodeSlot::output)
                continue;
            for (const NodeSlot* c : slot.connections())
            {
                Reach& downstream = reach_[index_.at(&c->parent())];
                downstream.thenBits |= inherited.thenBits;
                downstream.elseBits |= inherited.elseBits;
            }
        }
    }

    // Until a recording names its inputs, nothing is ruled out.
    snapshotGates();
    currentInputs_ = ~uint64_t(0);
}

inline void GraphController::snapshotGates()
{
    thenDead_ = elseDead_ = 0;
    for (size_t b = 0; b < gates_.size(); ++b)
        (gates_[b]->isOpen() ? elseDead_ : thenDead_) |= uint64_t(1) << b;
}

inline void GraphController::beginRecord(const std::vector<InputNode*>& activeInputs)
{
    snapshotGates();

    currentInputs_ = 0;
    for (const InputNode* node : activeInputs)
        if (auto found = inputMask_.find(node); found != inputMask_.end())
            currentInputs_ |= found->second;
}

inline bool GraphController::hasNeedsFed(size_t nodeIdx) const
{
    const Reach& r = reach_[nodeIdx];
    if (((r.thenBits & ~r.elseBits & thenDead_)
       | (r.elseBits & ~r.thenBits & elseDead_)) != 0)
        return false;

    for (uint64_t neededInputs : listOfNeededInputs_[nodeIdx])
        if ((currentInputs_ & neededInputs) == neededInputs)
            return true;
    return false;
}

/*
Two producers on one slot are only meaningful under mutual exclusion, so every
producer must sit on one of a single gate's two branches, and both branches must
be used. Membership is exclusive on purpose: a node both branches reach lies past
the merge and stays live, so it must not count as either side.
*/
inline bool GraphController::isGateMerge(const NodeSlot& in) const
{
    for (size_t b = 0; b < gates_.size(); ++b)
    {
        const node::Gate* gate = gates_[b];
        const uint64_t gateBit = uint64_t(1) << b;
        const NodeSlot* thenSlot = gate->findSlot("then");
        const NodeSlot* elseSlot = gate->findSlot("else");

        bool usedThen = false, usedElse = false, allOnThisGate = true;

        for (const NodeSlot* producer : in.connections())
        {
            const Reach& r = reach_.at(index_.at(&producer->parent()));
            const bool onThen = producer == thenSlot
                             || ((r.thenBits & gateBit) && !(r.elseBits & gateBit));
            const bool onElse = producer == elseSlot
                             || ((r.elseBits & gateBit) && !(r.thenBits & gateBit));

            if      (onThen) usedThen = true;
            else if (onElse) usedElse = true;
            else { allOnThisGate = false; break; }
        }

        if (allOnThisGate && usedThen && usedElse)
            return true;
    }
    return false;
}

inline bool GraphController::fanInIsGateMergeOnly() const
{
    for (const auto& [node, i] : index_)
        for (const auto& [name, slot] : node->slots())
            if (slot.type() == NodeSlot::input && slot.connections().size() >= 2)
                if (!isGateMerge(slot))
                    return false;
    return true;
}

} // namespace vai

#endif   // VAI_ENGINE_GRAPH_CONTROLLER_H
