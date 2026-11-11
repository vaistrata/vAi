#include "python/bindings/module.h"
#include "python/bindings/convert.h"
#include "vai/engine/tensor.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>


namespace pyvai {

namespace {

// A Tensor holds F32 or F16 values and numpy floats convert to its dtype both
// ways. The core has no integer dtype: its uint32 indices (token ids, argmax /
// top-k output) ride as raw bits in F32 slots, which from_u32 / to_u32 expose.

// Any C-contiguous CPU array, dtype untouched: the callers branch on the dtype,
// so nanobind converts nothing behind their back.
using NpAny = nb::ndarray<nb::c_contig, nb::device::cpu>;

std::vector<uint32_t> shapeFromNdarray(const NpAny& array)
{
    std::vector<uint32_t> shape(array.ndim());
    for (size_t i = 0; i < array.ndim(); ++i)
        shape[i] = static_cast<uint32_t>(array.shape(i));
    return shape;
}


// dtype descriptors and names

using Code = nb::dlpack::dtype_code;

bool is(const nb::dlpack::dtype& dt, Code code, uint8_t bits)
{
    return dt.code == static_cast<uint8_t>(code) && dt.bits == bits && dt.lanes == 1;
}


// numpy's spelling of a DLPack dtype, for error messages ("float64", "int16").
std::string npName(const nb::dlpack::dtype& dt)
{
    const char* base = "?";
    switch (static_cast<Code>(dt.code))
    {
        case Code::Float:   base = "float";   break;
        case Code::Int:     base = "int";     break;
        case Code::UInt:    base = "uint";    break;
        case Code::Bool:    return "bool";
        case Code::Complex: base = "complex"; break;
        case Code::Bfloat:  base = "bfloat";  break;
        default: break;
    }
    return base + std::to_string(dt.bits);
}


const char* vaiName(vai::Dtype dt)
{
    switch (dt)
    {
        case vai::Dtype::Fp32: return "F32";
        case vai::Dtype::Fp16: return "F16";
        case vai::Dtype::None: return "None";
        case vai::Dtype::Q4:   return "Q4";
    }
    return "?";
}


// The numpy dtype a tensor's values read back as. C++ has no portable fp16
// scalar, so that descriptor is spelled out.
nb::dlpack::dtype npDtypeOf(vai::Dtype dt)
{
    if (dt == vai::Dtype::Fp16)
        return nb::dlpack::dtype{ static_cast<uint8_t>(Code::Float), 16, 1 };
    return nb::dtype<float>();
}


// numpy -> host bytes

// The core's setFloats / setData assert (abort) on each of these.
void checkWritable(const vai::Tensor& dst, size_t count, const char* where)
{
    if (!dst.validShape())
        throw nb::value_error(
            (std::string(where)
             + ": tensor has no shape (use Tensor(shape) or from_numpy)").c_str());
    if (dst.baseOffset() != 0)
        throw nb::value_error(
            (std::string(where) + ": not supported on an offset view").c_str());
    if (dst.numElements() != count)
        throw nb::value_error(
            (std::string(where) + ": array has " + std::to_string(count) + " elements, the tensor "
             + std::to_string(dst.numElements())).c_str());
}


// Float values in, converted to the tensor's dtype. Same-width sources are a
// straight copy (float32 -> F32, float16 -> F16); the rest go through fp32.
void ingestFloats(vai::Tensor& dst, const NpAny& array, const char* where)
{
    const nb::dlpack::dtype dt = array.dtype();
    const size_t n = array.size();
    const void* p = array.data();

    if (!is(dt, Code::Float, 32) && !is(dt, Code::Float, 64) && !is(dt, Code::Float, 16))
        throw nb::type_error(
            (std::string(where) + ": got " + npName(dt) + "; only float32 / float16 / float64 "
             "arrays are accepted (token ids go through Tensor.from_u32)").c_str());

    checkWritable(dst, n, where);

    if (is(dt, Code::Float, 32))
    {
        dst.setFloats(static_cast<const float*>(p));
    }
    else if (is(dt, Code::Float, 64))
    {
        std::vector<float> tmp(n);
        const double* src = static_cast<const double*>(p);
        for (size_t i = 0; i < n; ++i)
            tmp[i] = static_cast<float>(src[i]);
        dst.setFloats(tmp.data());
    }
    else if (dst.dtype() == vai::Dtype::Fp16)
    {
        dst.setData(static_cast<const uint16_t*>(p));   // fp16 -> F16: bit-exact
    }
    else
    {
        std::vector<float> tmp(n);
        vai::convertDtype(p, vai::Dtype::Fp16, tmp.data(), vai::Dtype::Fp32, n);
        dst.setFloats(tmp.data());
    }
}


// uint32 index bits into a 4-byte (F32) slot: core setData<uint32_t>.
void ingestU32(vai::Tensor& dst, const NpAny& array, const char* where)
{
    const nb::dlpack::dtype dt = array.dtype();
    if (!is(dt, Code::UInt, 32) && !is(dt, Code::Int, 32))
        throw nb::type_error(
            (std::string(where) + ": got " + npName(dt) + "; pass an int32 / uint32 array "
             "(np.asarray(ids, dtype=np.uint32)) or a plain sequence of ints").c_str());
    if (dst.dtype() != vai::Dtype::Fp32)
        throw nb::value_error(
            (std::string(where) + ": uint32 payloads live in F32 (4-byte) tensors; this one is "
             + vaiName(dst.dtype())).c_str());

    checkWritable(dst, array.size(), where);
    dst.setData(static_cast<const uint32_t*>(array.data()));
}


// `ids` as a contiguous CPU ndarray: an ndarray passes through (ingestU32
// checks its dtype), anything else is packed by numpy as uint32.
NpAny asU32Array(nb::handle ids)
{
    NpAny array;
    if (nb::try_cast<NpAny>(ids, array, /*convert*/ false))
        return array;
    nb::module_ np = nb::module_::import_("numpy");
    nb::object packed = np.attr("asarray")(ids, nb::arg("dtype") = np.attr("uint32"));
    return nb::cast<NpAny>(packed);
}


// host bytes <- Tensor (device readback when needed)

// A forward output is downloaded (waiting for its forward, GIL released) and a
// host copy is read directly. A bound buffer with neither (an uploaded input or
// weight) is what the core cannot serve, so it is refused.
const void* hostBytes(vai::Tensor& t, const char* who)
{
    if (!t.validShape())
        throw nb::value_error((std::string(who) + ": tensor has no shape").c_str());

    if (t.gpuBufferBound() && t.lastSubmission().second != 0)
    {
        nb::gil_scoped_release rel;
        return t.downloadToHost().hostData();
    }
    if (t.hasHostData())
        return t.hostData();
    if (t.gpuBufferBound())
        throw std::runtime_error(
            std::string(who) + ": the tensor's host copy was released by an upload and no "
            "forward has written it since");
    throw nb::value_error((std::string(who) + ": tensor has no data").c_str());
}


// A fresh numpy array of the tensor's bytes under `npDtype` (same element
// width). Always a copy: the host buffer may be released by the next upload.
nb::ndarray<nb::numpy> toNdarray(vai::Tensor& t, nb::dlpack::dtype npDtype, const char* who)
{
    const char* src = static_cast<const char*>(hostBytes(t, who));
    std::vector<char> bytes(src, src + t.numBytes());

    const auto& shape = t.shape();
    const std::vector<size_t> npShape(shape.begin(), shape.end());
    return ownedNdarray(std::move(bytes), npShape, npDtype);
}


nb::ndarray<nb::numpy> toNumpy(vai::Tensor& t)
{
    if (t.dtype() == vai::Dtype::None)
        throw nb::value_error("Tensor.to_numpy: empty tensor");
    return toNdarray(t, npDtypeOf(t.dtype()), "Tensor.to_numpy");
}


nb::ndarray<nb::numpy> toU32(vai::Tensor& t)
{
    if (t.dtype() != vai::Dtype::Fp32)
        throw nb::value_error(
            (std::string("Tensor.to_u32: uint32 payloads live in F32 (4-byte) tensors; "
                         "this one is ")
             + vaiName(t.dtype())).c_str());
    return toNdarray(t, nb::dtype<uint32_t>(), "Tensor.to_u32");
}

}   // namespace


void bindTensor(nb::module_& m)
{
    // BF16 never surfaces: the parser widens it to fp32. Dtype::None is absent
    // because "unspecified" is `None` on every dtype parameter.
    nb::enum_<vai::Dtype>(m, "Dtype")
        .value("F32", vai::Dtype::Fp32)
        .value("F16", vai::Dtype::Fp16);

    // dynamic_attr: a forward output holds its net in `_pyvai_net` (forward.h).
    nb::class_<vai::Tensor>(m, "Tensor", nb::dynamic_attr(),
        "An F32 or F16 array of values, on the host and/or the device.\n\n"
        "from_numpy() / set() take float arrays and convert them to the tensor's "
        "dtype; to_numpy() gives them back in that dtype. uint32 indices (token "
        "ids, argmax and sampler output) ride as raw bits in an F32 tensor through "
        "from_u32() / to_u32(). Every transfer is a copy.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](vai::Tensor* self, const std::vector<uint32_t>& shape, vai::Dtype dtype) {
                new (self) vai::Tensor(shape, dtype);
            },
            nb::arg("shape"),
            nb::arg("dtype") = vai::Dtype::Fp32,
            "An uninitialised tensor of `shape`; fill it with set() or hand it to "
            "a forward as an output."
        )

        .def_static(
            "from_numpy",
            [](NpAny array, std::optional<vai::Dtype> dtype) -> vai::Tensor {
                vai::Dtype dt = dtype ? *dtype
                              : is(array.dtype(), Code::Float, 16) ? vai::Dtype::Fp16
                                                                   : vai::Dtype::Fp32;
                vai::Tensor t(shapeFromNdarray(array), dt);
                ingestFloats(t, array, "Tensor.from_numpy");
                return t;
            },
            nb::arg("array"),
            nb::arg("dtype") = nb::none(),
            "A tensor holding `array`'s values: a float32 / float16 / float64 array, "
            "converted to `dtype` (default: F16 for float16, else F32). TypeError "
            "for any other dtype; indices go through from_u32()."
        )

        .def_static(
            "from_u32",
            [](nb::handle ids) -> vai::Tensor {
                NpAny array = asU32Array(ids);
                vai::Tensor t(shapeFromNdarray(array), vai::Dtype::Fp32);
                ingestU32(t, array, "Tensor.from_u32");
                return t;
            },
            nb::arg("ids"),
            "A tensor carrying uint32 indices (token ids) as raw bits in F32 slots. "
            "`ids` is an int32 / uint32 array or a sequence of ints. Read it back "
            "with to_u32(), not to_numpy()."
        )

        .def_prop_ro("shape",
            [](const vai::Tensor& t) { return shapeTuple(t.shape()); },
            "Dimensions as a tuple of ints; () for an empty tensor.")
        .def_prop_ro("ndim",
            [](const vai::Tensor& t) -> size_t { return t.shape().size(); },
            "Number of dimensions; 0 for an empty tensor.")
        .def("__len__",
            [](const vai::Tensor& t) -> size_t {
                if (t.shape().empty())
                    throw nb::type_error("Tensor.__len__: tensor has no shape");
                return t.shape()[0];
            },
            "The leading dimension. TypeError for an empty tensor.")
        // Dtype::None is not a Python enum member: an empty tensor reports None.
        .def_prop_ro("dtype",
            [](const vai::Tensor& t) -> std::optional<vai::Dtype> {
                if (t.dtype() == vai::Dtype::None)
                    return std::nullopt;
                return t.dtype();
            },
            "The element dtype, or None for an empty tensor (Tensor()).")
        .def_prop_ro("num_elements", &vai::Tensor::numElements)
        .def_prop_ro("num_bytes",    &vai::Tensor::numBytes)

        // Where the data currently lives — what the readback path looks at.
        .def_prop_ro("has_host_data", &vai::Tensor::hasHostData,
            "True while a host copy exists: a forward's upload releases it and a "
            "device readback refills it.")
        .def_prop_ro("gpu_buffer_bound", &vai::Tensor::gpuBufferBound,
            "True once a forward has uploaded this tensor or written it as an output.")
        .def_prop_ro("base_offset", &vai::Tensor::baseOffset,
            "Element offset into the bound buffer; non-zero only for a slice.")

        .def(
            "set",
            [](vai::Tensor& self, NpAny array) -> vai::Tensor& {
                ingestFloats(self, array, "Tensor.set");
                return self;
            },
            nb::arg("array"),
            nb::rv_policy::reference,
            "Copy `array` into this tensor, converted to its dtype: the in-place "
            "from_numpy(). The shape must match. The next forward that takes the "
            "tensor uploads it again. Returns self."
        )

        .def(
            "reshape",
            [](nb::handle_t<vai::Tensor> self, std::vector<uint32_t> shape) -> nb::object {
                const auto& t = nb::cast<const vai::Tensor&>(self);
                // The core asserts on an empty tensor: it has no storage to share.
                if (t.dtype() == vai::Dtype::None)
                    throw nb::value_error("Tensor.reshape: empty tensor (no data to view)");
                size_t total = 1;
                for (uint32_t d : shape)
                    total *= d;
                if (t.numElements() != total)
                    throw nb::value_error(
                        "Tensor.reshape: total element count must remain constant");
                nb::object view = nb::cast(t.reshape(shape));
                // A view of a forward output pins its net too.
                if (nb::hasattr(self, "_pyvai_net"))
                    view.attr("_pyvai_net") = self.attr("_pyvai_net");
                return view;
            },
            nb::arg("shape"),
            "The same data seen as `shape`: a view that shares storage and leaves "
            "this tensor as it is. A forward output can be viewed too, and reading "
            "the view waits for its forward."
        )

        .def(
            "permute",
            [](const vai::Tensor& self, nb::args perms) -> vai::Tensor {
                if (!self.hasHostData())
                    throw nb::value_error(
                        "Tensor.permute: requires host data (call set() or use from_numpy)");

                std::vector<uint32_t> p;
                p.reserve(perms.size());
                for (auto h : perms)
                    p.push_back(nb::cast<uint32_t>(h));

                const uint32_t rank = self.shape().size();
                if (p.size() != rank)
                    throw nb::value_error("Tensor.permute: arg count must match tensor rank");
                for (uint32_t v : p)
                    if (v >= rank)
                        throw nb::value_error("Tensor.permute: index out of range");
                for (size_t i = 0; i < p.size(); ++i)
                    for (size_t j = i + 1; j < p.size(); ++j)
                        if (p[i] == p[j])
                            throw nb::value_error("Tensor.permute: duplicate index");

                // vai::Tensor::permute is a variadic template; dispatch by rank.
                switch (rank)
                {
                    case 1: return self.permute(p[0]);
                    case 2: return self.permute(p[0], p[1]);
                    case 3: return self.permute(p[0], p[1], p[2]);
                    case 4: return self.permute(p[0], p[1], p[2], p[3]);
                    default:
                        throw nb::value_error("Tensor.permute: rank > 4 is not supported");
                }
            },
            "A new tensor with the axes reordered; this one is left as it is."
        )

        .def("to_numpy",
            [](vai::Tensor& t) { return toNumpy(t); },
            "A new numpy array of the values, in the tensor's dtype. A forward "
            "output is read back from the device, waiting for its forward, and a "
            "handle read after a later forward sees that forward's result. "
            "RuntimeError when an upload released the host copy and no forward has "
            "written the tensor since (a consumed input, a weight). Indices go "
            "through to_u32().")

        .def("to_u32",
            [](vai::Tensor& t) { return toU32(t); },
            "The uint32 indices an F32 tensor carries as raw bits (ArgmaxNode / "
            "TopKSamplingNode output, a model's `nextTokenId`), as a new uint32 "
            "array. Readback as in to_numpy(); ValueError on an F16 tensor.")

        // Always a fresh copy, so copy=False (numpy >= 2) is refused, as the
        // protocol requires.
        .def(
            "__array__",
            [](vai::Tensor& t, nb::object dtype, nb::object copy) -> nb::object {
                if (!copy.is_none() && !nb::cast<bool>(copy))
                    throw nb::value_error(
                        "Tensor.__array__: a Tensor cannot be viewed without a copy "
                        "(copy=False); use copy=None or copy=True");
                nb::object arr = nb::cast(toNumpy(t));
                if (!dtype.is_none())
                    arr = arr.attr("astype")(dtype, nb::arg("copy") = false);
                return arr;
            },
            nb::arg("dtype") = nb::none(),
            nb::arg("copy")  = nb::none(),
            "numpy array protocol: the copy to_numpy() makes, cast to `dtype` when "
            "given, so np.asarray(t) and ufuncs take a Tensor. copy=False raises "
            "ValueError."
        )

        .def("__repr__", [](const vai::Tensor& t) -> std::string {
            std::string s = "Tensor(shape=[";
            if (t.validShape())
            {
                const auto& sh = t.shape();
                for (size_t i = 0; i < sh.size(); ++i)
                {
                    if (i)
                        s += ", ";
                    s += std::to_string(sh[i]);
                }
            }
            s += "], dtype=";
            s += vaiName(t.dtype());
            s += ")";
            return s;
        });
}

}   // namespace pyvai
