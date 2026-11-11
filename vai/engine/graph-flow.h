#ifndef VAI_ENGINE_GRAPH_FLOW_H
#define VAI_ENGINE_GRAPH_FLOW_H

#include "neural-net.h"
#include <map>
#include <optional>
#include <type_traits>
#include <string>
#include <utility>

namespace vai {

/////////////////////////////////////////////////////////////////////////////////////////
// Graph wiring DSL - connects slots with `-` and picks them with `/`
//
//   input - projQ - splitQ - "Q"/attn - projOut;
//
// `/` names the slot on the side it faces, so `"Q"/attn` is attn's input Q and
// `attn/"out1"` is its second output. `-` connects the left flow's output slot
// to the right one's input slot and yields the right flow, which is what lets a
// chain keep going. Unnamed sides default to in0 / out0.
//
// A flow is a node (or group) plus the two slot names the chain is currently
// pointing at. It lives only for the duration of one wiring expression.
/////////////////////////////////////////////////////////////////////////////////////////
struct NodeFlow
{
    std::string inSlotName;
    Node& node;
    std::string outSlotName;
};

inline Node::operator NodeFlow()
{
    return NodeFlow("in0", *this, "out0");
}

inline NodeFlow operator/(std::string name, Node& node)
{
    return { name, node, "out0" };
}

inline NodeFlow operator/(Node& node, std::string name)
{
    return { "in0", node, name };
}

inline NodeFlow&& operator/(NodeFlow&& other, std::string name)
{
    other.outSlotName = name;
    return std::move(other);
}

inline void operator-(NodeSlot& a, NodeSlot& b)
{
    NodeSlot::connect(a, b);
}

inline NodeFlow&& operator-(NodeFlow&& inflow, NodeFlow&& outflow)
{
    inflow.node.slot(inflow.outSlotName) - outflow.node.slot(outflow.inSlotName);
    return std::move(outflow);
}


/////////////////////////////////////////////////////////////////////////////////////////
// NodeGroup - a subgraph that wires like a single node
//
// It owns no slots of its own; defineSlot() republishes a member node's slot
// under a group-level name, so the same `-` and `/` reach inside.
/////////////////////////////////////////////////////////////////////////////////////////
struct GroupFlow;

class NodeGroup
{
    std::map<std::string, NodeSlot*> slots_;

protected:
    const NeuralNet* ownedNet_ = nullptr;

    void defineSlot(const std::string& name, NodeSlot& slot)
    {
        slots_.try_emplace(name, &slot);
    }

public:
    NodeGroup() : ownedNet_(NeuralNet::current()) {}

    NodeSlot& slot(const std::string& name)
    {
        return *slots_.at(name);
    }

    operator GroupFlow();
};

struct GroupFlow
{
    std::string inSlotName;
    NodeGroup& group;
    std::string outSlotName;
};

inline NodeGroup::operator GroupFlow()
{
    return { "in0", *this, "out0" };
}

inline GroupFlow operator/(std::string name, NodeGroup& gp)
{
    return { name, gp, "out0" };
}

inline GroupFlow operator/(NodeGroup& gp, std::string name)
{
    return { "in0", gp, name };
}

inline GroupFlow&& operator/(GroupFlow&& other, std::string name)
{
    other.outSlotName = name;
    return std::move(other);
}

inline GroupFlow&& operator-(GroupFlow&& inflow, GroupFlow&& outflow)
{
    inflow.group.slot(inflow.outSlotName) - outflow.group.slot(outflow.inSlotName);
    return std::move(outflow);
}

inline GroupFlow&& operator-(NodeFlow&& inflow, GroupFlow&& outflow)
{
    inflow.node.slot(inflow.outSlotName) - outflow.group.slot(outflow.inSlotName);
    return std::move(outflow);
}

inline NodeFlow&& operator-(GroupFlow&& inflow, NodeFlow&& outflow)
{
    inflow.group.slot(inflow.outSlotName) - outflow.node.slot(outflow.inSlotName);
    return std::move(outflow);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Optional links - a stop that configuration may have left out
//
// A std::optional<Node> or std::optional<NodeGroup> wires like the thing it
// holds. An empty one is not an error: the flow passes through it, so one chain
// spells a wiring whose stops are configuration, instead of one chain per
// combination of them.
//
//     input - projQ - splitQ - normQ - perQ - ropeQ - "Q"/attn;
//
// Slot naming works the same ("in1"/stop, stop/"then"), and an optional may sit
// anywhere: first, last, or beside another optional.
//
// Nodes and groups reach the same representation here, a pair of slots, so the
// two kinds compose without one operator per pairing. These return by value: an
// empty stop hands back the link that came in, so there is nothing to return a
// reference to.
/////////////////////////////////////////////////////////////////////////////////////////
struct FlowLink
{
    NodeSlot* inSlot;    // what the previous link connects to
    NodeSlot* outSlot;   // what the next one connects from
};

namespace detail {

inline std::optional<FlowLink> linkOf(NodeFlow&& f)
{
    return FlowLink{&f.node.slot(f.inSlotName), &f.node.slot(f.outSlotName)};
}

inline std::optional<FlowLink> linkOf(GroupFlow&& f)
{
    return FlowLink{&f.group.slot(f.inSlotName), &f.group.slot(f.outSlotName)};
}

inline std::optional<FlowLink> linkOf(std::optional<FlowLink>&& link)
{
    return std::move(link);
}

template<typename T> requires std::is_base_of_v<Node, T>
inline std::optional<FlowLink> linkOf(std::optional<T>& stop)
{
    return stop ? linkOf(static_cast<NodeFlow>(*stop)) : std::nullopt;
}

template<typename T> requires std::is_base_of_v<NodeGroup, T>
inline std::optional<FlowLink> linkOf(std::optional<T>& stop)
{
    return stop ? linkOf(static_cast<GroupFlow>(*stop)) : std::nullopt;
}

template<typename T> inline constexpr bool isOptional = false;
template<typename T> inline constexpr bool isOptional<std::optional<T>> = true;

template<typename T>
concept Linkable = requires(T&& t) { linkOf(std::forward<T>(t)); };

template<typename T>
concept MaybeEmpty = isOptional<std::remove_cvref_t<T>>;

}   // namespace detail


// Wires the two links when both are there; otherwise passes on the one that is.
template<typename L, typename R>
    requires detail::Linkable<L> && detail::Linkable<R>
          && (detail::MaybeEmpty<L> || detail::MaybeEmpty<R>)
inline std::optional<FlowLink> operator-(L&& inflow, R&& outflow)
{
    std::optional<FlowLink> in  = detail::linkOf(std::forward<L>(inflow));
    std::optional<FlowLink> out = detail::linkOf(std::forward<R>(outflow));

    if (in && out)
        *in->outSlot - *out->inSlot;

    return out ? out : in;
}

template<typename T>
inline std::optional<FlowLink> operator/(std::string name, std::optional<T>& stop)
{
    if (!stop)
        return std::nullopt;
    if constexpr (std::is_base_of_v<NodeGroup, T>)
        return detail::linkOf(GroupFlow{name, *stop, "out0"});
    else
        return detail::linkOf(NodeFlow{name, *stop, "out0"});
}

template<typename T>
inline std::optional<FlowLink> operator/(std::optional<T>& stop, std::string name)
{
    if (!stop)
        return std::nullopt;
    if constexpr (std::is_base_of_v<NodeGroup, T>)
        return detail::linkOf(GroupFlow{"in0", *stop, name});
    else
        return detail::linkOf(NodeFlow{"in0", *stop, name});
}

} // namespace vai

#endif   // VAI_ENGINE_GRAPH_FLOW_H
