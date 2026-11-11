#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/type-cast.h"

namespace pyvai {

static void bindTypeCast(nb::module_& m)
{
    using C = vai::node::TypeCast::Config;
    bindNode<vai::node::TypeCast>(m, "TypeCastNode",
                                  "Element-wise cast of in0 to o_dtype, between fp32 and fp16. When in0 "
                                  "already has o_dtype, out0 aliases in0's buffer.",
                                  required("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindTypeCast);

}   // namespace pyvai
