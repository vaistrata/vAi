#ifndef PYVAI_CONVERT_H
#define PYVAI_CONVERT_H

// Conversions shared across areas: C++ values on their way to Python.
#include "common.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace pyvai {

// A shape as Python hands it out: a tuple of ints (torch.Size, np.shape).
using PyShape = nb::typed<nb::tuple, int, nb::ellipsis>;

inline PyShape shapeTuple(const std::vector<uint32_t>& shape)
{
    nb::list l;
    for (uint32_t d : shape)
        l.append(nb::int_(d));
    return nb::tuple(l);
}

// A numpy array that owns `values`: the vector moves into a capsule the array
// keeps alive, so the elements are not copied. Needs the GIL.
template <typename T>
nb::ndarray<nb::numpy> ownedNdarray(std::vector<T>&& values, const std::vector<size_t>& shape,
                                    nb::dlpack::dtype dtype = nb::dtype<T>())
{
    auto owned = std::make_unique<std::vector<T>>(std::move(values));
    nb::capsule owner(owned.get(), [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    std::vector<T>* raw = owned.release();   // the capsule owns it now
    return nb::ndarray<nb::numpy>(
        raw->data(), shape.size(), shape.data(), owner, /*strides*/ nullptr, dtype);
}

}   // namespace pyvai

#endif // PYVAI_CONVERT_H
