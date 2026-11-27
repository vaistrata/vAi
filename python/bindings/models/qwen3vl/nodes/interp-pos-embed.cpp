#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/qwen3vl/nodes/interp-pos-embed.h"
#include <cstdint>

namespace pyvai {

namespace {

using model::experimental::qwen3vl::node::InterpPosEmbed;

// InterpPosEmbed reads the grid width from a uniform its owner holds: a C++ model
// writes one per image, a node built from Python its own, set at construction.
struct PyInterpPosEmbed : InterpPosEmbed
{
    eva::Buffer gridWidth;

    PyInterpPosEmbed(Config config, uint32_t width)
    : InterpPosEmbed(config)
    {
        gridWidth = vai::createKvCacheCounter(device());   // a host-visible uniform uint
        *(uint32_t*)gridWidth.map() = width;
        setGridWidth(gridWidth);
    }
    ~PyInterpPosEmbed()
    {
        if (gridWidth)
            gridWidth.destroy();
    }
};

}   // namespace


static void bindInterpPosEmbed(nb::module_& m)
{
    nb::class_<PyInterpPosEmbed, vai::Node>(m, "InterpPosEmbedNode")
        .def(
            "__init__",
            [](PyInterpPosEmbed* self, uint32_t grid_width, uint32_t tile_size) {
                requireCurrentNet("InterpPosEmbedNode");
                if (grid_width == 0 || tile_size == 0 || grid_width % tile_size != 0)
                    throw nb::value_error(
                        "InterpPosEmbedNode: grid_width and tile_size must be positive, "
                        "grid_width a multiple of tile_size");
                new (self) PyInterpPosEmbed({ .tileSize = tile_size }, grid_width);
            },
            nb::arg("grid_width"),
            nb::arg("tile_size") = 1u,
            "Qwen3-VL's position table added in place. in0 [..., N, D] holds patches of a "
            "grid grid_width wide, N / grid_width tall, in tile_size x tile_size tiles; the "
            "parameter table [S * S, D] is an S x S grid resampled bilinearly (corners "
            "aligned) to the patch grid. out0 aliases in0.");
}
PYVAI_MODEL_NODE_BINDING(qwen3vl, bindInterpPosEmbed);

}   // namespace pyvai
