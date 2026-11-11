#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/permute.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pyvai {

static void bindPermute(nb::module_& m)
{
    nb::class_<vai::node::Permute, vai::Node>(m, "PermuteNode")
        .def(
            "__init__",
            [](vai::node::Permute* self,
               std::vector<uint32_t> perms,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("PermuteNode");
                // The core stores perms unvalidated and indexes with them —
                // out-of-range or duplicate entries are an OOB read at prepare.
                const size_t rank = perms.size();
                if (rank == 0 || rank > PERMUTE_MAX_RANK)
                    throw nb::value_error(
                        ("PermuteNode: perms must hold between 1 and "
                         + std::to_string(PERMUTE_MAX_RANK) + " axes").c_str());
                std::vector<bool> seen(rank, false);
                for (uint32_t axis : perms)
                {
                    if (axis >= rank || seen[axis])
                        throw nb::value_error(
                            "PermuteNode: perms must be a permutation of (0, .., len(perms) - 1)");
                    seen[axis] = true;
                }
                new (self) vai::node::Permute(
                    std::move(perms),
                    vai::node::Permute::Config{
                        .oDtype = o_dtype.value_or(vai::Dtype::None),
                    });
            },
            nb::arg("perms"),
            nb::arg("o_dtype") = nb::none(),
            "out0 dimension i is in0 dimension perms[i], so perms lists each axis of "
            "in0 once. out0 aliases in0's buffer when the permutation keeps memory "
            "order and the dtype is unchanged."
        );
}
PYVAI_NODE_BINDING(bindPermute);

}   // namespace pyvai
