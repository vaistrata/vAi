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
// - field(name, &C::member): the default is Config{}'s. A vai::Dtype field also
//   accepts None, its Python default, which leaves Config{}'s value in place:
//   `o_dtype=None` keeps the core's own (the net's default for Dtype::None, F32
//   where the core says F32).
// - field(name, &C::member, value): Python's default is `value` instead.
// - required(name, &C::member): no default.
// .check(pred, message) runs once every field is set, in field order; pred takes
// the value, or the value and the whole Config to compare fields. A failure is
// "<Node>: <name> <message>" as a ValueError: the core's VAI_ASSERT would abort
// the process (or, with NDEBUG, not run at all).
//
// The node is built as NodeClass(config), or NodeClass() when it has no Config.
// NodeClass may be a binding-local subclass that holds what a Python-built node
// needs. bindNode returns the nb::class_ for methods to chain onto.
//
// When each part runs:
// - build: the field types decide what each field does (the if constexpr below);
// - `import pyvai`, once: the binding function runs, field() / required() /
//   .check() make the fields, and bindNode registers the class and its arguments;
// - every NodeClass(...) call from Python: each field's assign(), then each
//   field's validate(), then the node's constructor. A forward runs none of this.
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

template <typename NodeClass>
concept HasConfig = requires { typename NodeClass::Config; };

}   // namespace detail


// Where a field's Python default comes from.
enum class FieldDefault {
    Required,     // none: required()
    FromConfig,   // Config{}'s value: field(name, member)
    FromPython,   // a value the binding gives: field(name, member, value)
};


// One Config field as one keyword argument of the Python constructor.
template <typename Config, typename Value, FieldDefault from>
class Field
{
public:
    // Whether the argument also accepts None, which keeps Config{}'s value. Set
    // for a dtype field with Config{}'s default; whether a call passed None is
    // assign()'s question.
    static constexpr bool acceptsNone = from == FieldDefault::FromConfig && std::is_same_v<Value, vai::Dtype>;
    // The C++ type nanobind converts the Python argument to.
    using ArgType = std::conditional_t<acceptsNone, std::optional<Value>, Value>;

    Field(const char* name, Value Config::* member, Value pyDefault = Value{})
        : name_(name), member_(member), pyDefault_(std::move(pyDefault))
    {
    }

    // A copy of this field with one more check. pred(value) or pred(value,
    // config) -> bool; false raises "<Node>: <name> <message>".
    template <typename Pred>
    Field check(Pred pred, std::string message) const
    {
        Field f = *this;
        if constexpr (std::is_invocable_r_v<bool, Pred, const Value&, const Config&>)
            f.checks_.push_back({std::move(pred), std::move(message)});
        else   // pred takes the value alone: give it the (value, config) shape checks_ holds
            f.checks_.push_back(
                {[pred](const Value& value, const Config&) { return pred(value); }, std::move(message)});
        return f;
    }

    // At import: the keyword argument and its default.
    auto arg() const
    {
        if constexpr (from == FieldDefault::Required)
            return nb::arg(name_);
        else if constexpr (acceptsNone)
            return nb::arg(name_) = nb::none();
        else if constexpr (from == FieldDefault::FromConfig)
            return nb::arg(name_) = Config{}.*member_;
        else
            return nb::arg(name_) = pyDefault_;
    }

    // On every construction: the argument into its Config field. None leaves
    // Config{}'s value.
    void assign(Config& config, const ArgType& value) const
    {
        if constexpr (acceptsNone)
        {
            if (value)
                config.*member_ = *value;
        }
        else
            config.*member_ = value;
    }

    // On every construction, after every field is assigned, so a check may read
    // the other fields.
    void validate(const char* node, const Config& config) const
    {
        for (const auto& [pred, message] : checks_)
            if (!pred(config.*member_, config))
                throw nb::value_error((std::string(node) + ": " + name_ + " " + message).c_str());
    }

private:
    struct Check
    {
        std::function<bool(const Value&, const Config&)> pred;
        std::string message;
    };

    const char* name_;         // the Python keyword
    Value Config::* member_;   // which Config field it sets
    Value pyDefault_;          // FieldDefault::FromPython only
    std::vector<Check> checks_;
};


// The template arguments come from the member: &C::count gives Config = C and
// Value = count's type. FieldDefault comes from which function is called.
template <typename Config, typename Value>
Field<Config, Value, FieldDefault::FromConfig> field(const char* name, Value Config::* member)
{
    return {name, member};
}

template <typename Config, typename Value, typename PyDefault>
Field<Config, Value, FieldDefault::FromPython> field(const char* name, Value Config::* member, PyDefault pyDefault)
{
    return {name, member, Value(std::move(pyDefault))};
}

template <typename Config, typename Value>
Field<Config, Value, FieldDefault::Required> required(const char* name, Value Config::* member)
{
    return {name, member};
}


template <typename Value>
auto atLeast(Value low)
{
    return [low](const Value& value) { return value >= low; };
}


template <typename NodeClass, typename... Fields>
nb::class_<NodeClass, vai::Node> bindNode(nb::module_& m, const char* name, const char* doc, Fields... fields)
{
    nb::class_<NodeClass, vai::Node> cls(m, name);
    // The lambda keeps copies of the fields: it is the Python constructor, run on
    // every call. fields.arg()... runs here, once.
    cls.def(
        "__init__",
        [name, fields...](NodeClass* self, typename Fields::ArgType... args) {
            requireCurrentNet(name);
            if constexpr (detail::HasConfig<NodeClass>)
            {
                typename NodeClass::Config config{};
                (fields.assign(config, args), ...);     // each field, in order
                (fields.validate(name, config), ...);   // then each field's checks
                new (self) NodeClass(config);           // into the storage nanobind gives
            }
            else
                new (self) NodeClass();
        },
        fields.arg()..., doc);
    return cls;
}


// The registry PYVAI_NODE_BINDING / PYVAI_MODEL_NODE_BINDING fill as the module
// loads.
struct NodeBinding
{
    std::string model;            // "" for a core node, else its models/<model>/
    std::string id;               // the bind function's name; bindings run in id order
    void (*bind)(nb::module_&);   // the bind function
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

// A static object whose constructor adds fn to nodeBindings(). It runs as the
// extension loads, before the module init that calls bindNodes() / bindModels().
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
