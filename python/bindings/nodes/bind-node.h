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
// - field(name, &C::member): the default is Config{}'s. A vai::Dtype field has
//   None as its Python default instead, and None leaves Config{}'s value in
//   place: `o_dtype=None` keeps the core's own (the net's default for
//   Dtype::None, F32 where the core says F32).
// - field(name, &C::member, value): the binding's `value` is the default.
// - required(name, &C::member): no default.
// .check(predicate, errorMessage) runs once every field is set, in field order;
// predicate takes the value, or the value and the whole Config to compare
// fields. A failure is "<Node>: <name> <errorMessage>" as a ValueError: the
// core's VAI_ASSERT would abort the process (or, with NDEBUG, not run at all).
//
// The node is built as NodeClass(config), or NodeClass() when it has no Config.
// NodeClass may be a binding-local subclass that holds what a Python-built node
// needs. bindNode returns the nb::class_ for methods to chain onto.
//
// When each part runs:
// - build: the field types decide what each field does (the if constexpr below);
// - `import pyvai`, once: the binding function runs, field() / required() /
//   .check() make the fields, and bindNode registers the class and its arguments;
// - every NodeClass(...) call from Python: each field's assignToConfig(), then
//   each field's validate(), then the node's constructor. A forward runs none of
//   this.
//
// PYVAI_NODE_BINDING(bindFunction) / PYVAI_MODEL_NODE_BINDING(modelName,
// bindFunction) at the end of a binding file registers the function: bindNodes()
// calls the core ones, bindModels() the model ones in
// pyvai._pyvai.model_nodes.<modelName>. No list to edit elsewhere.

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


// How a field's keyword argument gets its default.
enum class FieldDefaultPolicy {
    Required,          // no default: required()
    ConfigDefault,     // Config{}'s value: field(name, member)
    ExplicitDefault,   // a value the binding gives: field(name, member, value)
};


// One Config member as one keyword argument of the Python constructor.
template <typename Config, typename MemberType, FieldDefaultPolicy defaultPolicy>
class Field
{
public:
    // A dtype field with Config{}'s default has None as its Python default; None
    // then leaves Config{}'s value. Whether a call passed None is
    // assignToConfig()'s question.
    static constexpr bool hasNoneDefault =
        defaultPolicy == FieldDefaultPolicy::ConfigDefault && std::is_same_v<MemberType, vai::Dtype>;
    // The C++ type nanobind converts the Python argument to.
    using ConvertedArgumentType = std::conditional_t<hasNoneDefault, std::optional<MemberType>, MemberType>;

    Field(const char* argumentName, MemberType Config::* configMember, MemberType explicitDefault = MemberType{})
        : argumentName_(argumentName), configMember_(configMember), explicitDefault_(std::move(explicitDefault))
    {
    }

    // A copy of this field with one more check. predicate(value) or
    // predicate(value, config) -> bool; false raises "<Node>: <name> <errorMessage>".
    template <typename PredicateType>
    Field check(PredicateType predicate, std::string errorMessage) const
    {
        Field checkedField = *this;
        if constexpr (std::is_invocable_r_v<bool, PredicateType, const MemberType&, const Config&>)
            checkedField.checks_.push_back({std::move(predicate), std::move(errorMessage)});
        else   // predicate takes the value alone: give it the (value, config) shape checks_ holds
            checkedField.checks_.push_back(
                {[predicate](const MemberType& value, const Config&) { return predicate(value); },
                 std::move(errorMessage)});
        return checkedField;
    }

    // At import: the keyword argument and its default.
    auto makeArgument() const
    {
        if constexpr (defaultPolicy == FieldDefaultPolicy::Required)
            return nb::arg(argumentName_);
        else if constexpr (hasNoneDefault)
            return nb::arg(argumentName_) = nb::none();
        else if constexpr (defaultPolicy == FieldDefaultPolicy::ConfigDefault)
            return nb::arg(argumentName_) = Config{}.*configMember_;
        else
            return nb::arg(argumentName_) = explicitDefault_;
    }

    // On every construction: the argument into its Config member. None leaves
    // Config{}'s value.
    void assignToConfig(Config& config, const ConvertedArgumentType& argumentValue) const
    {
        if constexpr (hasNoneDefault)
        {
            if (argumentValue)
                config.*configMember_ = *argumentValue;
        }
        else
            config.*configMember_ = argumentValue;
    }

    // On every construction, after every field is assigned, so a check may read
    // the other fields.
    void validate(const char* pythonClassName, const Config& config) const
    {
        for (const auto& [predicate, errorMessage] : checks_)
            if (!predicate(config.*configMember_, config))
                throw nb::value_error(
                    (std::string(pythonClassName) + ": " + argumentName_ + " " + errorMessage).c_str());
    }

private:
    struct Check
    {
        std::function<bool(const MemberType&, const Config&)> predicate;
        std::string errorMessage;
    };

    const char* argumentName_;            // the Python keyword
    MemberType Config::* configMember_;   // the Config member it sets
    MemberType explicitDefault_;          // FieldDefaultPolicy::ExplicitDefault only
    std::vector<Check> checks_;
};


// The template arguments come from the member: &C::count gives Config = C and
// MemberType = count's type. The FieldDefaultPolicy comes from which function is
// called.
template <typename Config, typename MemberType>
Field<Config, MemberType, FieldDefaultPolicy::ConfigDefault> field(const char* argumentName,
                                                                   MemberType Config::* configMember)
{
    return {argumentName, configMember};
}

template <typename Config, typename MemberType, typename ExplicitDefaultType>
Field<Config, MemberType, FieldDefaultPolicy::ExplicitDefault>
field(const char* argumentName, MemberType Config::* configMember, ExplicitDefaultType explicitDefault)
{
    return {argumentName, configMember, MemberType(std::move(explicitDefault))};
}

template <typename Config, typename MemberType>
Field<Config, MemberType, FieldDefaultPolicy::Required> required(const char* argumentName,
                                                                 MemberType Config::* configMember)
{
    return {argumentName, configMember};
}


template <typename NumberType>
auto atLeast(NumberType minimum)
{
    return [minimum](const NumberType& value) { return value >= minimum; };
}


template <typename NodeClass, typename... FieldTypes>
nb::class_<NodeClass, vai::Node> bindNode(nb::module_& m, const char* pythonClassName, const char* docstring,
                                          FieldTypes... fields)
{
    nb::class_<NodeClass, vai::Node> cls(m, pythonClassName);
    // The lambda keeps copies of the fields: it is the Python constructor, run on
    // every call. fields.makeArgument()... runs here, once.
    cls.def(
        "__init__",
        [pythonClassName, fields...](NodeClass* self, typename FieldTypes::ConvertedArgumentType... argumentValues) {
            requireCurrentNet(pythonClassName);
            if constexpr (detail::HasConfig<NodeClass>)
            {
                typename NodeClass::Config config{};
                (fields.assignToConfig(config, argumentValues), ...);   // each field, in order
                (fields.validate(pythonClassName, config), ...);        // then each field's checks
                new (self) NodeClass(config);                           // into the storage nanobind gives
            }
            else
                new (self) NodeClass();
        },
        fields.makeArgument()..., docstring);
    return cls;
}


// The registry PYVAI_NODE_BINDING / PYVAI_MODEL_NODE_BINDING fill as the module
// loads.
struct NodeBinding
{
    std::string modelName;                // "" for a core node, else its models/<modelName>/
    std::string bindFunctionName;         // bindings run in the order of this name
    void (*bindFunction)(nb::module_&);   // the binding file's bind function
};

std::vector<NodeBinding>& nodeBindings();

struct NodeBindingRegistration
{
    NodeBindingRegistration(const char* modelName, const char* bindFunctionName, void (*bindFunction)(nb::module_&))
    {
        nodeBindings().push_back({modelName, bindFunctionName, bindFunction});
    }
};

// The model's bindings in a stable order (by bindFunctionName); modelName "" is
// the core's.
void bindRegisteredNodes(nb::module_& m, const std::string& modelName);

}   // namespace pyvai

// A static object whose constructor adds bindFunction to nodeBindings(). It runs
// as the extension loads, before the module init that calls bindNodes() /
// bindModels().
#define PYVAI_NODE_BINDING(bindFunction)                                                                               \
    static const ::pyvai::NodeBindingRegistration pyvaiNodeBinding_##bindFunction                                      \
    {                                                                                                                  \
        "", #bindFunction, &bindFunction                                                                               \
    }

#define PYVAI_MODEL_NODE_BINDING(modelName, bindFunction)                                                              \
    static const ::pyvai::NodeBindingRegistration pyvaiNodeBinding_##modelName##_##bindFunction                        \
    {                                                                                                                  \
        #modelName, #bindFunction, &bindFunction                                                                       \
    }

#endif   // PYVAI_NODES_BIND_NODE_H
