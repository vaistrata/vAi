# Using vAi

The components needed to build and run a neural network with vAi.

|                                            |                        |
| ------------------------------------------ | ------------------------ |
| [Node](node.md)                             | A single operation. Passes tensors through slots |
| [NodeGroup](node-group.md)                   | A subgraph that groups multiple nodes into one unit |
| [NeuralNet](neural-net.md)                   | The entire graph. Manages execution and buffers |
| [Tensor](tensor.md)                         | Data that flows between nodes |
| [Execution pipeline](execution-pipeline.md)  | What happens in one `model(...)` call |
| [Profiling](profiling.md)                    | Timing individual sections |
