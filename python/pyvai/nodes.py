"""The built-in graph nodes, re-exported from the extension module.

Base classes (`Node`, `NodeGroup`, ...), `KVCacheCounter` and the config types
(`Epilogue`, `ResidualOrder`, `RopeType`, `RopeConfig`) stay at the top level of
`pyvai`.

A new node is bound in bindings/nodes/<node>.cpp, through bindNode() and one
PYVAI_NODE_BINDING line (bindings/nodes/bind-node.h), and re-exported here
(the import list and `__all__`; tests/test_nodes.py fails if one is missing).
"""
from ._pyvai import (
    # shape / plumbing
    IdentityNode,
    GateNode,
    ReshapeNode,
    FlattenNode,
    ConcatNode,
    PermuteNode,
    SliceNode,
    RowSliceScatterNode,
    IndexCopyNode,
    TypeCastNode,
    # element-wise
    AddNode,
    MulNode,
    GatedMulNode,
    EleAffineNode,
    ActivationNode,
    # vision
    ConvolutionNode,
    MaxPool2dNode,
    AvgPool2dNode,
    NearestUpsampleNode,
    # linear / normalization
    LinearNode,
    LayerNormNode,
    RmsNormNode,
    BatchNormNode,
    # transformer
    TokenEmbeddingNode,
    TimeEmbeddingNode,
    CyclicTableReaderNode,
    RopeNode,
    AttentionNode,
    # output heads
    ArgmaxNode,
    TopKSamplingNode,
)


__all__ = [
    "IdentityNode",
    "GateNode",
    "ReshapeNode",
    "FlattenNode",
    "ConcatNode",
    "PermuteNode",
    "SliceNode",
    "RowSliceScatterNode",
    "IndexCopyNode",
    "TypeCastNode",
    "AddNode",
    "MulNode",
    "GatedMulNode",
    "EleAffineNode",
    "ActivationNode",
    "ConvolutionNode",
    "MaxPool2dNode",
    "AvgPool2dNode",
    "NearestUpsampleNode",
    "LinearNode",
    "LayerNormNode",
    "RmsNormNode",
    "BatchNormNode",
    "TokenEmbeddingNode",
    "TimeEmbeddingNode",
    "CyclicTableReaderNode",
    "RopeNode",
    "AttentionNode",
    "ArgmaxNode",
    "TopKSamplingNode",
]
