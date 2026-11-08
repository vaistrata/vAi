#ifndef PYVAI_NODES_BIND_NODE_H
#define PYVAI_NODES_BIND_NODE_H

// bindNode(): a node's Python class from the list of its Config fields.
//
//     using C = vai::node::Slice::Config;
//     bindNode<vai::node::Slice>(m, "SliceNode", "Take `count` ...",
//         required("start", &C::start),
//         field("count", &C::count).check(atLeast(1u), "must be >= 1"),
//         field("axis", &C::axis),
//         field("o_dtype", &C::oDtype));
//     PYVAI_NODE_BINDING(bindSlice);
//
// The fields become the constructor's keyword arguments, in the order listed:
// - field(name, &C::member): the default is Config{}'s. A vai::Dtype field takes
//   None for that default, so `o_dtype=None` keeps the core's own (the net's
//   default for Dtype::None, F32 where the core says F32).
// - field(name, &C::member, value): Python's default is `value` instead.
// - required(name, &C::member): no default.
// .check(pred, message) runs once every field is set, in field order; pred takes
// the value, or the value and the whole Config to compare fields. A failure is
// "<Node>: <name> <message>" as a ValueError: the core's VAI_ASSERT would abort
// the process (or, with NDEBUG, not run at all).
//
// The node is built as Bound(config), or Bound() when it has no Config. Bound may
// be a binding-local subclass that holds what a Python-built node needs.
// bindNode returns the nb::class_ for methods to chain onto.
//
// PYVAI_NODE_BINDING(fn) / PYVAI_MODEL_NODE_BINDING(model, fn) at the end of a
// binding file registers fn: bindNodes() calls the core ones, bindModels() the
// model ones in pyvai._pyvai.model_nodes.<model>. No list to edit elsewhere.

#include "python/bindings/nodes/nodes.h"
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyvai {

namespace detail {

struct ConfigDefault
{
};
struct PythonDefault
{
};
struct Required
{
};

template <class Node>
concept HasConfig = requires { typename Node::Config; };

}   // namespace detail


template <class C, class T, class Kind>
class Field
{
public:
    // None stands for Config{}'s dtype, so a dtype keeps the core's default.
    static constexpr bool noneMeansDefault =
        std::is_same_v<Kind, detail::ConfigDefault> && std::is_same_v<T, vai::Dtype>;
    using PyType = std::conditional_t<noneMeansDefault, std::optional<T>, T>;

    Field(const char* name, T C::* member, T pyDefault = T{})
        : name_(name), member_(member), pyDefault_(std::move(pyDefault))
    {
    }

    // pred(value) or pred(value, config) -> bool; false raises "<Node>: <name> <message>".
    template <class Pred>
    Field check(Pred pred, std::string message) const
    {
        Field f = *this;
        if constexpr (std::is_invocable_r_v<bool, Pred, const T&, const C&>)
            f.checks_.push_back({std::move(pred), std::move(message)});
        else
            f.checks_.push_back({[pred](const T& v, const C&) { return pred(v); }, std::move(message)});
        return f;
    }

    auto arg() const
    {
        if constexpr (std::is_same_v<Kind, detail::Required>)
            return nb::arg(name_);
        else if constexpr (noneMeansDefault)
            return nb::arg(name_) = nb::none();
        else if constexpr (std::is_same_v<Kind, detail::ConfigDefault>)
            return nb::arg(name_) = C{}.*member_;
        else
            return nb::arg(name_) = pyDefault_;
    }

    void assign(C& config, const PyType& value) const
    {
        if constexpr (noneMeansDefault)
        {
            if (value)
                config.*member_ = *value;
        }
        else
            config.*member_ = value;
    }

    void validate(const char* node, const C& config) const
    {
        for (const auto& [pred, message] : checks_)
            if (!pred(config.*member_, config))
                throw nb::value_error((std::string(node) + ": " + name_ + " " + message).c_str());
    }

private:
    struct Check
    {
        std::function<bool(const T&, const C&)> pred;
        std::string message;
    };

    const char* name_;
    T C::* member_;
    T pyDefault_;
    std::vector<Check> checks_;
};


template <class C, class T>
Field<C, T, detail::ConfigDefault> field(const char* name, T C::* member)
{
    return {name, member};
}

template <class C, class T, class D>
Field<C, T, detail::PythonDefault> field(const char* name, T C::* member, D pyDefault)
{
    return {name, member, T(std::move(pyDefault))};
}

template <class C, class T>
Field<C, T, detail::Required> required(const char* name, T C::* member)
{
    return {name, member};
}


template <class T>
auto atLeast(T low)
{
    return [low](const T& v) { return v >= low; };
}


template <class Bound, class... Fields>
nb::class_<Bound, vai::Node> bindNode(nb::module_& m, const char* name, const char* doc, Fields... fields)
{
    nb::class_<Bound, vai::Node> cls(m, name);
    cls.def(
        "__init__",
        [name, fields...](Bound* self, typename Fields::PyType... values) {
            requireCurrentNet(name);
            if constexpr (detail::HasConfig<Bound>)
            {
                typename Bound::Config config{};
                (fields.assign(config, values), ...);
                (fields.validate(name, config), ...);
                new (self) Bound(config);
            }
            else
                new (self) Bound();
        },
        fields.arg()..., doc);
    return cls;
}


// The registry PYVAI_NODE_BINDING / PYVAI_MODEL_NODE_BINDING fill as the module
// loads. `model` is empty for a core node.
struct NodeBinding
{
    std::string model;
    std::string id;
    void (*bind)(nb::module_&);
};

std::vector<NodeBinding>& nodeBindings();

struct RegisterNodeBinding
{
    RegisterNodeBinding(const char* model, const char* id, void (*bind)(nb::module_&))
    {
        nodeBindings().push_back({model, id, bind});
    }
};

// The model's bindings in a stable order (by id); model "" is the core's.
void bindRegisteredNodes(nb::module_& m, const std::string& model);

}   // namespace pyvai

#define PYVAI_NODE_BINDING(fn)                                                                                         \
    static const ::pyvai::RegisterNodeBinding pyvaiNodeBinding_##fn                                                    \
    {                                                                                                                  \
        "", #fn, &fn                                                                                                   \
    }

#define PYVAI_MODEL_NODE_BINDING(model, fn)                                                                            \
    static const ::pyvai::RegisterNodeBinding pyvaiNodeBinding_##model##_##fn                                          \
    {                                                                                                                  \
        #model, #fn, &fn                                                                                               \
    }

#endif   // PYVAI_NODES_BIND_NODE_H
