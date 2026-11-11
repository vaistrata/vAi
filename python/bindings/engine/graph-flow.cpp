#include "python/bindings/module.h"
#include "vai/engine/graph-flow.h"   // NodeFlow / NodeGroup
#include "vai/engine/neural-net.h"
#include <stdexcept>
#include <utility>
#include <vector>

namespace pyvai {

namespace {

struct NodeGroupAccess : vai::NodeGroup
{
    using vai::NodeGroup::defineSlot;
};


// Slot lookups funnel here so a wrong name lists the real slots.
[[noreturn]] void throwNoSuchSlot(const vai::Node& n, const std::string& name)
{
    std::string msg = "Node: no such slot '" + name + "' (slots:";
    const char* sep = " ";
    for (const auto& [slotName, slot] : n.slots())
    {
        msg += sep;
        msg += slotName;
        sep = ", ";
    }
    msg += ")";
    throw nb::value_error(msg.c_str());
}


// The core links raw NodeSlot*, so a wired node must outlive its net. A flow
// holds every wrapper wired through it, and a connection hands them to
// `net._pyvai_dsl_refs` once either side is anchored, so temporaries survive.
// Wrappers wired before either side is anchored hold each other in
// `_pyvai_peers`, and the net that reaches one of them adopts them all.
struct PyNodeFlow
{
    std::string     inSlotName  = "in0";
    vai::Node*      node        = nullptr;
    vai::NodeGroup* group       = nullptr;
    std::string     outSlotName = "out0";

    std::vector<nb::object> refs;   // the chain's wrappers, the endpoint first

    vai::NodeSlot& slotByName(const std::string& name) const
    {
        if (node)
        {
            if (!node->findSlot(name))
                throwNoSuchSlot(*node, name);
            return node->slot(name);
        }
        try
        {
            return group->slot(name);
        }
        catch (const std::out_of_range&)
        {
            throw nb::value_error(
                ("NodeGroup: no such slot '" + name + "'").c_str());
        }
    }
};


PyNodeFlow flowOf(nb::handle_t<vai::Node> h)
{
    PyNodeFlow f;
    f.node = &nb::cast<vai::Node&>(h);
    f.refs = { nb::borrow<nb::object>(h) };
    return f;
}


PyNodeFlow flowOf(nb::handle_t<vai::NodeGroup> h)
{
    PyNodeFlow f;
    f.group = &nb::cast<vai::NodeGroup&>(h);
    f.refs  = { nb::borrow<nb::object>(h) };
    return f;
}


// The net anchoring this chain, or a null object when it has none yet.
nb::object anchorNetOf(const PyNodeFlow& f)
{
    for (const nb::object& o : f.refs)
        if (nb::hasattr(o, "_pyvai_net"))
            return o.attr("_pyvai_net");
    return nb::object();
}


// Anchor and stash every not-yet-anchored wrapper of `f`, and the peers those
// were wired to before: the net owns them now.
void adoptInto(nb::object net, const PyNodeFlow& f)
{
    nb::list bag;
    if (nb::hasattr(net, "_pyvai_dsl_refs"))
        bag = nb::cast<nb::list>(net.attr("_pyvai_dsl_refs"));
    else
        net.attr("_pyvai_dsl_refs") = bag;

    std::vector<nb::object> pending(f.refs);
    while (!pending.empty())
    {
        nb::object o = std::move(pending.back());
        pending.pop_back();
        if (nb::hasattr(o, "_pyvai_net"))
            continue;
        o.attr("_pyvai_net") = net;
        bag.append(o);
        if (nb::hasattr(o, "_pyvai_peers"))
            for (nb::handle peer : o.attr("_pyvai_peers"))
                pending.push_back(nb::borrow(peer));
    }
}


// The two wrappers whose slots were just linked hold each other. The lists
// live in the instances' __dict__, so the GC sees the cycle of a dropped chain.
void linkPeers(const nb::object& a, const nb::object& b)
{
    auto hold = [](const nb::object& o, const nb::object& peer) {
        if (!nb::hasattr(o, "_pyvai_peers"))
            o.attr("_pyvai_peers") = nb::list();
        nb::cast<nb::list>(o.attr("_pyvai_peers")).append(peer);
    };
    hold(a, b);
    hold(b, a);
}


PyNodeFlow connectFlowToFlow(const PyNodeFlow& src, const PyNodeFlow& dst)
{
    vai::NodeSlot& srcSlot = src.slotByName(src.outSlotName);
    vai::NodeSlot& dstSlot = dst.slotByName(dst.inSlotName);
    // The core connect() asserts (aborts) on slot-type mismatch.
    if (srcSlot.type() != vai::NodeSlot::output || dstSlot.type() != vai::NodeSlot::input)
        throw nb::value_error(
            ("connect: '" + src.outSlotName + "' must be an output slot and '" +
             dst.inSlotName + "' an input slot").c_str());
    vai::NodeSlot::connect(srcSlot, dstSlot);

    nb::object net = anchorNetOf(src);
    if (!net.is_valid())
        net = anchorNetOf(dst);
    if (net.is_valid())
    {
        adoptInto(net, src);
        adoptInto(net, dst);
    }
    else
        linkPeers(src.refs.front(), dst.refs.front());   // the endpoints come first

    // The chain goes on from dst and carries every ref for a later adoption.
    PyNodeFlow out = dst;
    out.refs.insert(out.refs.end(), src.refs.begin(), src.refs.end());
    return out;
}


// Any wirable object (Node / NodeGroup / NodeFlow) as a flow.
PyNodeFlow toFlow(nb::handle h)
{
    if (nb::isinstance<PyNodeFlow>(h))
        return nb::cast<const PyNodeFlow&>(h);
    PyNodeFlow f;
    if (nb::isinstance<vai::Node>(h))
        f.node = &nb::cast<vai::Node&>(h);
    else if (nb::isinstance<vai::NodeGroup>(h))
        f.group = &nb::cast<vai::NodeGroup&>(h);
    else
        throw nb::type_error("connect: expected a Node, NodeGroup or NodeFlow");
    f.refs = { nb::borrow<nb::object>(h) };
    return f;
}


// Call wiring: positional args go to in0, in1, ...; keywords to the named slots.
PyNodeFlow connectCall(PyNodeFlow out, nb::args args, nb::kwargs kwargs)
{
    if (nb::len(args) == 0 && nb::len(kwargs) == 0)
        throw nb::type_error("connect: pass at least one upstream node or flow");
    size_t i = 0;
    for (nb::handle a : args)
    {
        PyNodeFlow dst = out;
        dst.inSlotName = "in" + std::to_string(i++);
        out = connectFlowToFlow(toFlow(a), dst);
    }
    for (auto [k, v] : kwargs)
    {
        PyNodeFlow dst = out;
        dst.inSlotName = nb::cast<std::string>(k);
        out = connectFlowToFlow(toFlow(v), dst);
    }
    return out;
}


// PyNodeFlow's refs live in a C++ vector the GC cannot see; tp_traverse and
// tp_clear make a cycle through them collectable.
int nodeFlowTraverse(PyObject* self, visitproc visit, void* arg)
{
    if (!nb::inst_ready(self))
        return 0;
    PyNodeFlow* f = nb::inst_ptr<PyNodeFlow>(self);
    for (const nb::object& o : f->refs)
        Py_VISIT(o.ptr());
    Py_VISIT(Py_TYPE(self));
    return 0;
}


int nodeFlowClear(PyObject* self)
{
    if (nb::inst_ready(self))
        nb::inst_ptr<PyNodeFlow>(self)->refs.clear();
    return 0;
}


PyType_Slot nodeFlowSlots[] = {
    { Py_tp_traverse, reinterpret_cast<void*>(nodeFlowTraverse) },
    { Py_tp_clear,    reinterpret_cast<void*>(nodeFlowClear) },
    { 0, nullptr }
};


void checkSlotExists(vai::Node& n, const std::string& name)
{
    if (!n.findSlot(name))
        throwNoSuchSlot(n, name);
}

}   // namespace


void bindGraphFlow(nb::module_& m)
{
    // NodeSlot: opaque handle for NodeGroup.define_slot and the operators.
    nb::class_<vai::NodeSlot>(m, "NodeSlot");

    // dynamic_attr: carries the `_pyvai_net` anchor.
    nb::class_<vai::Node>(m, "Node", nb::dynamic_attr())
        .def(
            "slot",
            [](vai::Node& self, const std::string& name) -> vai::NodeSlot& {
                checkSlotExists(self, name);
                return self.slot(name);
            },
            nb::arg("name"), nb::rv_policy::reference_internal)

        .def("__getitem__",
             [](vai::Node& self, const std::string& name) -> vai::Tensor {
                 checkSlotExists(self, name);
                 return self[name];
             })

        .def("__setitem__",
             [](vai::Node& self, const std::string& name, const vai::Tensor& t) {
                 checkSlotExists(self, name);
                 self[name] = t;
             })

        // a - b: connect a.out0 to b.in0, return b's flow for chaining.
        .def("__sub__",
             [](nb::handle_t<vai::Node> a, nb::handle_t<vai::Node> b) -> PyNodeFlow {
                 return connectFlowToFlow(flowOf(a), flowOf(b));
             })
        .def("__sub__",
             [](nb::handle_t<vai::Node> a, nb::handle_t<vai::NodeGroup> b) -> PyNodeFlow {
                 return connectFlowToFlow(flowOf(a), flowOf(b));
             })
        .def("__sub__",
             [](nb::handle_t<vai::Node> a, const PyNodeFlow& b) -> PyNodeFlow {
                 return connectFlowToFlow(flowOf(a), b);
             })

        // Call-style wiring: y = linear(relu(x))
        .def(
            "__call__",
            [](nb::handle_t<vai::Node> self, nb::args args, nb::kwargs kwargs) -> PyNodeFlow {
                return connectCall(flowOf(self), args, kwargs);
            },
            "Wire nodes or flows into this node and return its flow: positional "
            "args go to in0, in1, ...; keywords to the named slots.")

        // node / "name": out-slot spec
        .def("__truediv__",
             [](nb::handle_t<vai::Node> self, const std::string& name) -> PyNodeFlow {
                 PyNodeFlow f = flowOf(self);
                 f.outSlotName = name;
                 return f;
             })
        // "name" / node: in-slot spec
        .def("__rtruediv__",
             [](nb::handle_t<vai::Node> self, const std::string& name) -> PyNodeFlow {
                 PyNodeFlow f = flowOf(self);
                 f.inSlotName = name;
                 return f;
             })

        .def("__repr__",
             [](nb::handle self) -> std::string { return nb::cast<std::string>(self.type().attr("__name__")) + "()"; });

    nb::class_<vai::InputNode,  vai::Node>(m, "InputNode",
        "A net's input boundary, from NeuralNet.input(name). The tensor passed "
        "to __call__ under `name` leaves through out0.");
    nb::class_<vai::OutputNode, vai::Node>(m, "OutputNode",
        "A net's output boundary, from NeuralNet.output(name). What is wired "
        "into in0 comes back from __call__ under `name`, or as an empty Tensor "
        "when only a closed GateNode branch reaches it.");

    nb::class_<vai::NodeGroup>(m, "NodeGroup", nb::dynamic_attr())
        .def(nb::init<>())

        .def(
            "slot",
            [](vai::NodeGroup& self, const std::string& name) -> vai::NodeSlot& {
                return self.slot(name);
            },
            nb::arg("name"),
            nb::rv_policy::reference_internal
        )

        // NodeGroupAccess exposes the protected defineSlot. keep_alive<1, 3>: the
        // group holds the slot wrapper, which holds its node (reference_internal).
        .def("define_slot", &NodeGroupAccess::defineSlot,
             nb::arg("name"), nb::arg("slot"), nb::keep_alive<1, 3>())

        .def(
            "__sub__",
            [](nb::handle_t<vai::NodeGroup> a, nb::handle_t<vai::Node> b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](nb::handle_t<vai::NodeGroup> a, nb::handle_t<vai::NodeGroup> b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](nb::handle_t<vai::NodeGroup> a, const PyNodeFlow& b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), b);
            }
        )

        .def(
            "__call__",
            [](nb::handle_t<vai::NodeGroup> self, nb::args args, nb::kwargs kwargs) -> PyNodeFlow {
                return connectCall(flowOf(self), args, kwargs);
            },
            "Wire nodes or flows into this group and return its flow: positional "
            "args go to in0, in1, ...; keywords to the named slots."
        )

        .def(
            "__truediv__",
            [](nb::handle_t<vai::NodeGroup> self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow f = flowOf(self);
                f.outSlotName = name;
                return f;
            }
        )
        .def(
            "__rtruediv__",
            [](nb::handle_t<vai::NodeGroup> self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow f = flowOf(self);
                f.inSlotName = name;
                return f;
            }
        )

        .def(
            "__repr__",
            [](nb::handle self) -> std::string {
                std::string typeName = nb::cast<std::string>(
                    self.type().attr("__name__"));
                return typeName + "()";
            }
        );

    nb::class_<PyNodeFlow>(m, "NodeFlow", nb::type_slots(nodeFlowSlots))
        .def(
            "__sub__",
            [](const PyNodeFlow& a, nb::handle_t<vai::Node> b) -> PyNodeFlow {
                return connectFlowToFlow(a, flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](const PyNodeFlow& a, nb::handle_t<vai::NodeGroup> b) -> PyNodeFlow {
                return connectFlowToFlow(a, flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](const PyNodeFlow& a, const PyNodeFlow& b) -> PyNodeFlow {
                return connectFlowToFlow(a, b);
            }
        )
        .def(
            "__truediv__",
            [](const PyNodeFlow& self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow out = self;
                out.outSlotName = name;
                return out;
            }
        )
        // "name" / flow: in-slot spec
        .def(
            "__rtruediv__",
            [](const PyNodeFlow& self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow out = self;
                out.inSlotName = name;
                return out;
            }
        )

        .def(
            "__repr__",
            [](const PyNodeFlow& f) -> std::string {
                std::string target;
                if (f.node)
                    target = "node=<Node>";
                else if (f.group)
                    target = "group=<NodeGroup>";
                return "NodeFlow(" + target +
                       ", in='" + f.inSlotName +
                       "', out='" + f.outSlotName + "')";
            }
        );
}

}   // namespace pyvai
