#ifndef PYVAI_ENGINE_FORWARD_H
#define PYVAI_ENGINE_FORWARD_H

// The body of NeuralNet.__call__, the input staging the prefill() / decode()
// models share with it, and the NeuralNetConfig both are built with. State
// lives on the net object:
//   _pyvai_io                    (inputs, outputs) a C++-built model declares
//   _pyvai_input/output_names    names passed to net.input() / net.output()
//   _pyvai_inputs / _pyvai_outputs   staging / result Tensors, kept across
//                                calls since a new buffer means re-recording

#include "python/bindings/common.h"
#include "vai/engine/neural-net.h"
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pyvai {

// An absent device leaves the null handle, and the net then lives on the
// process-wide device.
inline vai::NeuralNetConfig netConfig(vai::Dtype wDefaultDtype, vai::Dtype oDefaultDtype,
                                      const std::optional<eva::Device>& device)
{
    vai::NeuralNetConfig cfg;
    cfg.device = device.value_or(eva::Device{});
    cfg.wDefaultDtype = wDefaultDtype;
    cfg.oDefaultDtype = oDefaultDtype;
    return cfg;
}

// A C++-built model declares its boundary names on the class.
template <typename Cls>
void declareBoundaries(Cls& cls,
                       std::initializer_list<const char*> inputs,
                       std::initializer_list<const char*> outputs)
{
    auto asTuple = [](std::initializer_list<const char*> names) {
        nb::list l;
        for (const char* n : names)
            l.append(nb::str(n));
        return nb::tuple(l);
    };
    cls.attr("_pyvai_io") = nb::make_tuple(asTuple(inputs), asTuple(outputs));
}

// Records a net.input() / net.output() name on the instance.
inline void recordBoundary(nb::handle net, const char* instanceAttr, const std::string& name)
{
    if (!nb::hasattr(net, instanceAttr))
        net.attr(instanceAttr) = nb::set();
    nb::object s = net.attr(instanceAttr);
    nb::borrow<nb::set>(s).add(nb::str(name.c_str()));
}

// Class declaration plus instance records; ioIndex 0 = inputs, 1 = outputs.
inline nb::set boundaryNames(nb::handle net, const char* instanceAttr, size_t ioIndex)
{
    nb::set names;

    nb::object io = nb::getattr(net, "_pyvai_io", nb::none());
    if (!io.is_none())
    {
        nb::object part = nb::borrow<nb::tuple>(io)[ioIndex];
        for (nb::handle n : part)
            names.add(n);
    }

    nb::object inst = nb::getattr(net, instanceAttr, nb::none());
    if (!inst.is_none())
        for (nb::handle n : inst)
            names.add(n);

    return names;
}

inline nb::set inputNamesOf(nb::handle net)
{
    return boundaryNames(net, "_pyvai_input_names", 0);
}

inline nb::set outputNamesOf(nb::handle net)
{
    return boundaryNames(net, "_pyvai_output_names", 1);
}

// For the input_names / output_names properties.
inline std::vector<std::string> sortedNames(nb::set names)
{
    std::vector<std::string> out;
    for (nb::handle n : names)
        out.push_back(nb::cast<std::string>(n));
    std::sort(out.begin(), out.end());
    return out;
}

// A subset of the inputs is fine (an omitted input prunes its branch); an
// empty dict, an unknown name or a net without an input boundary is what the
// core asserts on.
inline void checkInputKeys(nb::handle net, nb::dict inputs)
{
    if (inputs.size() == 0)
        throw nb::key_error("NeuralNet.__call__: at least one input is required");

    nb::set expected = inputNamesOf(net);
    if (expected.size() == 0)
        throw std::runtime_error(
            "NeuralNet.__call__: the net has no input boundary — call net.input(name) "
            "while building it");

    // clang-format off
    for (auto [k, v] : inputs)
        if (!expected.contains(k))
            throw nb::key_error(
                ("NeuralNet.__call__: unknown input " + nb::cast<std::string>(nb::repr(k))
                 + "; the inputs are " + nb::cast<std::string>(nb::repr(expected))).c_str());
    // clang-format on
}

// Host-byte copy into a same-shape, same-dtype tensor; setData() also marks
// it for re-upload.
inline void copyHostBytes(vai::Tensor& dst, const vai::Tensor& src)
{
    const void* p = src.hostData();
    switch (vai::bytesPerElement(dst.dtype()))
    {
        case 2: dst.setData(static_cast<const uint16_t*>(p)); break;
        case 4: dst.setData(static_cast<const uint32_t*>(p)); break;
        default:
            throw std::runtime_error("NeuralNet.__call__: unsupported input element size");
    }
}

// Host tensors are copied into the net's staging tensor for that name;
// device-resident ones (forward outputs) pass through. `who` prefixes the errors.
inline vai::TensorDict stageInputs(nb::handle net, nb::dict inputs,
                                   const char* who = "NeuralNet.__call__")
{
    const std::string at = std::string(who) + ": ";
    if (!nb::hasattr(net, "_pyvai_inputs"))
        net.attr("_pyvai_inputs") = nb::dict();
    nb::object cacheObj = net.attr("_pyvai_inputs");
    nb::dict cache = nb::borrow<nb::dict>(cacheObj);

    vai::TensorDict in;
    for (auto [k, v] : inputs)
    {
        if (!nb::isinstance<nb::str>(k) || !nb::isinstance<vai::Tensor>(v))
            throw nb::type_error((at + "inputs must map str -> Tensor").c_str());

        const std::string name = nb::cast<std::string>(k);
        const vai::Tensor& user = nb::cast<const vai::Tensor&>(v);

        if (user.gpuBufferBound() && user.lastSubmission().second != 0)
        {
            in.emplace(name, user);   // already on the device
            continue;
        }
        if (!user.validShape())
            throw nb::value_error(
                (at + "input '" + name + "' has no shape").c_str());
        // clang-format off
        if (!user.hasHostData())
            throw std::runtime_error(
                (at + "input '" + name + "' has no host data — its host "
                 "copy was released by an upload; set() it again").c_str());
        // clang-format on

        vai::Tensor* staged = nullptr;
        nb::object stagedObj;
        if (cache.contains(k))
        {
            stagedObj = cache[k];
            vai::Tensor& t = nb::cast<vai::Tensor&>(stagedObj);
            if (t.shape() == user.shape() && t.dtype() == user.dtype())
                staged = &t;
        }
        if (!staged)
        {
            stagedObj = nb::cast(vai::Tensor(user.shape(), user.dtype()));
            cache[k] = stagedObj;
            staged = &nb::cast<vai::Tensor&>(stagedObj);
        }

        copyHostBytes(*staged, user);
        in.emplace(name, *staged);
    }
    return in;
}

// The Tensors a forward writes into: the caller's, or the net's own.
inline nb::dict resolveOutputs(nb::handle net, std::optional<nb::dict> outputs)
{
    nb::dict res;

    if (outputs)
    {
        for (auto [k, v] : *outputs)
        {
            if (!nb::isinstance<nb::str>(k) || !nb::isinstance<vai::Tensor>(v))
                throw nb::type_error("NeuralNet.__call__: outputs must map str -> Tensor");
            res[k] = v;
        }
        return res;
    }

    nb::set names = outputNamesOf(net);
    if (names.size() == 0)
        throw std::runtime_error(
            "NeuralNet.__call__: the net has no output boundary — call net.output(name) "
            "while building it, or pass outputs=");

    if (!nb::hasattr(net, "_pyvai_outputs"))
        net.attr("_pyvai_outputs") = nb::dict();
    nb::object cacheObj = net.attr("_pyvai_outputs");
    nb::dict cache = nb::borrow<nb::dict>(cacheObj);

    for (nb::handle name : names)
    {
        if (!cache.contains(name))
            cache[name] = nb::cast(vai::Tensor());
        res[name] = cache[name];
    }
    return res;
}

// For the models driven by prefill() / decode() instead of a forward: one
// input staged as __call__ stages it, and the net's own Tensor for an output
// (kept across calls, pinning the net as every forward output does).
inline vai::Tensor stageInput(nb::handle net, const char* name, nb::handle tensor, const char* who)
{
    nb::dict one;
    one[name] = tensor;
    return stageInputs(net, one, who).at(name);
}

inline nb::object ownOutput(nb::handle net, const char* name)
{
    if (!nb::hasattr(net, "_pyvai_outputs"))
        net.attr("_pyvai_outputs") = nb::dict();
    nb::object cacheObj = net.attr("_pyvai_outputs");
    nb::dict cache = nb::borrow<nb::dict>(cacheObj);
    if (!cache.contains(name))
        cache[name] = nb::cast(vai::Tensor());
    nb::object out = cache[name];
    out.attr("_pyvai_net") = net;
    return out;
}

// `run(TensorDict, TensorRefDict, bool syncPrev)` is the C++ call. Outputs
// view the net's buffers, so each holds the net in `_pyvai_net` (a __dict__
// attribute, not keep_alive: the net caches the outputs too, and only a
// GC-visible edge lets that cycle be collected).
template <typename Run>
nb::dict forward(nb::handle self, nb::dict inputs,
                 std::optional<nb::dict> outputs, bool syncPrev, Run&& run)
{
    checkInputKeys(self, inputs);
    vai::TensorDict in = stageInputs(self, inputs);
    nb::dict outDict = resolveOutputs(self, std::move(outputs));

    vai::TensorRefDict refs;
    for (auto [k, v] : outDict)
        refs.emplace(nb::cast<std::string>(k), &nb::cast<vai::Tensor&>(v));

    try
    {
        nb::gil_scoped_release rel;
        run(std::move(in), std::move(refs), syncPrev);
    }
    catch (const std::out_of_range&)   // outputs_.at(name) on an unknown output name
    {
        throw nb::key_error(
            "NeuralNet.__call__: an output name does not match the net's output boundaries");
    }

    for (auto [k, v] : outDict)
        v.attr("_pyvai_net") = self;
    return outDict;
}


// __call__ forms: net(x) for a single-input net, net({name: x}), or
// net(name=x); plus outputs= and sync_prev=.

inline constexpr const char* callSig =
    "def __call__(self, *args, outputs: dict[str, Tensor] | None = None, "
    "sync_prev: bool = True, **inputs) -> Tensor | dict[str, Tensor]";

struct CallArgs
{
    nb::dict inputs;
    std::optional<nb::dict> outputs;
    bool syncPrev = true;
};

inline CallArgs parseCallArgs(nb::handle net, nb::args args, nb::kwargs kwargs)
{
    CallArgs c;
    nb::dict kw = nb::borrow<nb::dict>(kwargs);

    if (kw.contains("outputs"))
    {
        nb::object o = kw["outputs"];
        if (!o.is_none())
        {
            if (!nb::isinstance<nb::dict>(o))
                throw nb::type_error(
                    "NeuralNet.__call__: outputs must be a dict[str, Tensor] or None");
            c.outputs = nb::borrow<nb::dict>(o);
        }
        PyDict_DelItemString(kw.ptr(), "outputs");
    }
    if (kw.contains("sync_prev"))
    {
        c.syncPrev = nb::cast<bool>(kw["sync_prev"]);
        PyDict_DelItemString(kw.ptr(), "sync_prev");
    }

    if (args.size() == 1 && kw.size() == 0)
    {
        nb::object a = args[0];
        if (nb::isinstance<nb::dict>(a))
        {
            c.inputs = nb::borrow<nb::dict>(a);
            return c;
        }
        if (nb::isinstance<vai::Tensor>(a))
        {
            nb::set names = inputNamesOf(net);
            // clang-format off
            if (names.size() != 1)
                throw nb::type_error(
                    ("NeuralNet.__call__: net(x) needs a net with exactly one input; "
                     "this one has "
                     + std::to_string(names.size()) + " ("
                     + nb::cast<std::string>(nb::repr(names)) + ") — name them, "
                     "e.g. net(in0=x, in1=y)").c_str());
            // clang-format on
            for (nb::handle n : names)
                c.inputs[n] = a;
            return c;
        }
        throw nb::type_error(
            "NeuralNet.__call__: the positional argument must be a Tensor or a dict[str, Tensor]");
    }
    if (args.size() == 0 && kw.size() > 0)
    {
        c.inputs = kw;   // the remaining keywords are the inputs
        return c;
    }
    throw nb::type_error(
        "NeuralNet.__call__: pass the inputs as one Tensor, one dict, or keywords "
        "(net(x) / net({'in0': x}) / net(in0=x)), plus outputs=");
}

inline nb::object shapeResult(nb::dict res)
{
    if (res.size() == 1)
        for (auto [k, v] : res)
            return nb::borrow(v);
    return res;
}

template <typename Run>
nb::object callForward(nb::handle self, nb::args args, nb::kwargs kwargs, Run&& run)
{
    CallArgs c = parseCallArgs(self, args, kwargs);
    return shapeResult(forward(self, c.inputs, std::move(c.outputs), c.syncPrev,
                               std::forward<Run>(run)));
}

}   // namespace pyvai

#endif // PYVAI_ENGINE_FORWARD_H
