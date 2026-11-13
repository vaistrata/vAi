# Profiling

vAi can measure each node's **GPU execution time**. It records timestamps before and after the node's `run()`, then reads and accumulates them after the forward pass ends.

## Enabling profiling

Profiling is enabled when at least one node has a tag. With no tags, there is no profiling overhead.

```cpp
net.decoder.layers[i].mhAttn.attn.setProfileTag("Attn");
```


## Nodes with the same tag are combined

Times for nodes with the same tag are summed.

```cpp
for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
{
    net.decoder.layers[i].norm1.setProfileTag("LayerNorm");
    net.decoder.layers[i].norm2.setProfileTag("LayerNorm");
}
```

With 12 blocks, the times of 24 nodes appear in one `LayerNorm` row.

A `.` in a tag groups entries into a hierarchy.

```cpp
b.mhAttn.projQKV.setProfileTag("Linear.[768x2304]");
b.mhAttn.projOut.setProfileTag("Linear.[768x768]");
```

In this case, times are listed by shape under `Linear`, and the parent shows the sum of its children.

## Multiple forward passes are averaged

Measurements accumulate after each forward pass, and `printProfile()` prints the average across the recorded forward passes.

```cpp
for (int i = 0; i < 5; ++i)   // Warmup
    net(inDict);

net.resetProfileRecords();   // Discard warmup records

for (int i = 0; i < 10; ++i)
    net(inDict);

net.printProfile();          // Average over 10 passes
```

Initial runs can include one-time costs such as shader compilation or buffer preparation, so measure after a warmup. Use `resetProfileRecords()` to clear all accumulated records and start measuring again.

## Output and resetting records

```text
=== Per-node GPU profile (avg over 10 iters) ===
  ├── Conv1x1bn                 :  28.1411 ms  (15.05%)  ██████
  │   ├── [1024x256@80]         :   3.5974 ms  ( 1.92%)  █
  │   └── [1280x512@40]         :   4.7626 ms  ( 2.55%)  █
  ├── Conv3x3s1                 : 129.6934 ms  (69.36%)  ████████████████████████████
  OTHER (untagged)              :   6.1002 ms  ( 3.25%)
  TOTAL (full forward)          : 187.5394 ms
```

| Item    | Meaning |
| ------- | ---------------------- |
| `ms`    | Average GPU time per forward pass |
| `%`     | Percentage of `TOTAL` |
| `TOTAL` | GPU time for the entire forward pass |
| `OTHER` | Time spent in untagged sections |

A large `OTHER` value means much of the time is spent in sections that have not been tagged yet.

By default, `printProfile()` resets accumulated records after printing. Use `printProfile(false)` to inspect intermediate results while keeping the records.

The default behavior makes it easy to measure two phases with the same tags separately, such as prefill and decode.

```cpp
net(inDict);
net.printProfile();          // Print prefill results and reset records

for (int i = 0; i < maxNewTokens; ++i)
    net(inDict);

net.printProfile();          // Print only the decode phase
```

Because the first print resets the records, the second print includes only the measurements accumulated during decode.

## Changing tags

Tags can be changed during measurement. The changes take effect when the command buffer is re-recorded on the next forward pass.

After inspecting the full profile, clear the existing tags and tag the desired nodes to examine a particular bottleneck in detail.

```cpp
net.clearProfileTags();

net.decoder.layers[0].ffn.fc1.setProfileTag("fc1");
net.decoder.layers[0].ffn.fc2.setProfileTag("fc2");
```

Pass an empty string to remove the tag from a specific node.

```cpp
net.lmHead.setProfileTag("");
```

Changing the tag configuration resets existing records so measurements of different targets are not mixed into one average.

## Reading a value

The average GPU time for a specific tag can also be read as a number without printing.

```cpp
double ms = net.getGpuTimeMs("Linear.[768x2304]");
```

`getGpuTimeMs()` **looks up the full tag as a single name**, without interpreting `.` as a hierarchy. For a hierarchical tag, pass its exact full name, such as `"Linear.[768x2304]"`, rather than `"Linear"`.

It returns `0.0` if the tag does not exist or no forward pass has been recorded yet.

## Notes

`printProfile()`, `getGpuTimeMs()`, and `resetProfileRecords()` internally wait for the previous forward pass to finish before reading GPU timestamps. Calling them on every iteration of a measurement loop can serialize execution, so call them outside the loop where possible.

Profiling adds timestamp commands before and after each tagged node. Measurements are at node granularity: even if a node dispatches multiple kernels, the entire node is measured as one section.

## API summary

| API                                         | Role |
| ------------------------------------------- | ----------------------- |
| `Node::setProfileTag(tag)`                  | Set a tag; an empty string removes it |
| `NeuralNet::clearProfileTags()`             | Remove all tags |
| `NeuralNet::resetProfileRecords()`          | Reset accumulated records |
| `NeuralNet::printProfile(withResetRecords)` | Print averages; reset records after printing by default |
| `NeuralNet::getGpuTimeMs(tag)`              | Return the average GPU time for a specific tag |
