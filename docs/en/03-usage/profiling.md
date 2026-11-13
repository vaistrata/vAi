# Profiling

vAi can measure the **GPU execution time** of each node. It records timestamps before and after the node's `run()`, then reads and accumulates those values after the forward ends.

## Enabling profiling

Profiling is enabled when at least one tag is assigned to a node. Conversely, when no tags are assigned, profiling has no overhead.

```cpp
net.layers[i].mhAttn.attn.setProfileTag("Attn");
```


## Identical tags are combined

The times for nodes with the same tag are summed into a single value.

```cpp
for (uint32_t i = 0; i < net.modelCfg.numLayers; ++i)
{
    net.layers[i].norm1.setProfileTag("LayerNorm");
    net.layers[i].norm2.setProfileTag("LayerNorm");
}
```

If there are 12 layers, the times for all 24 nodes are collected on the single `LayerNorm` line.

Tags can be grouped into a hierarchy by including `.` in them.

```cpp
b.mhAttn.projQKV.setProfileTag("Linear.[768x2304]");
b.mhAttn.projOut.setProfileTag("Linear.[768x768]");
```

In this case, the times under `Linear` are divided by shape, and the parent shows the sum of the times for its child entries.

## Multiple forwards accumulate as an average

Measurements accumulate after each forward, and `printProfile()` prints the average of all forwards recorded so far.

```cpp
for (int i = 0; i < 5; ++i)   // Warm-up
    net(inDict);

net.resetProfileRecords();   // Discard the warm-up records

for (int i = 0; i < 10; ++i)
    net(inDict);

net.printProfile();          // Average over 10 runs
```

Initial runs can include one-time costs such as shader compilation or buffer preparation, so it is better to measure after a warm-up. `resetProfileRecords()` is used to reset all records accumulated so far before measuring again.

## Output and record reset

```text
=== Per-node GPU profile (avg over 10 iters) ===
  ├── Conv1x1bn                 :  28.1411 ms  (15.05%)  ██████
  │   ├── [1024x256@80]         :   3.5974 ms  ( 1.92%)  █
  │   └── [1280x512@40]         :   4.7626 ms  ( 2.55%)  █
  ├── Conv3x3s1                 : 129.6934 ms  (69.36%)  ████████████████████████████
  OTHER (untagged)              :   6.1002 ms  ( 3.25%)
  TOTAL (full forward)          : 187.5394 ms
```

| Item | Meaning |
|---|---|
| `ms` | Average GPU time per forward |
| `%` | Percentage of `TOTAL` |
| `TOTAL` | GPU time for the entire forward |
| `OTHER` | Time in untagged sections |

A large `OTHER` means that substantial time is being spent in sections that have not yet been tagged.

By default, `printProfile()` resets the accumulated records after printing. To inspect only the intermediate results while retaining the records, use `printProfile(false)`.

This default behavior makes it easy to measure two sections with the same tag configuration separately, such as prefill and decode.

```cpp
net(inDict);
net.printProfile();          // Print prefill and reset the records

for (int i = 0; i < maxNewTokens; ++i)
    net(inDict);

net.printProfile();          // Print only the decode section
```

The records are reset when the first output is printed, so the second output includes only the records newly accumulated during decode.

## Changing tags

Tags can be changed during measurement. The changed tags take effect when the command buffer is recorded again for the next forward.

If you want to examine a specific bottleneck in detail after checking the full profile, you can clear the existing tags and apply them again to the desired nodes.

```cpp
net.clearProfileTags();

net.layers[0].ffn.fc1.setProfileTag("fc1");
net.layers[0].ffn.fc2.setProfileTag("fc2");
```

Pass an empty string to remove the tag from a specific node.

```cpp
net.lmHead.setProfileTag("");
```

Existing records are reset when the tag configuration changes. This prevents measurements with different targets from being mixed into one average.

## Reading values

The average GPU time of a specific tag can also be read as a number without printing it.

```cpp
double ms = net.getGpuTimeMs("Linear.[768x2304]");
```

`getGpuTimeMs()` does not interpret `.` as a hierarchy and **looks up the entire tag as a single name**. Therefore, if a hierarchical tag is used, the exact full tag, such as `"Linear.[768x2304]"`, must be passed instead of `"Linear"`.

If the tag does not exist or no forward has been recorded yet, it returns `0.0`.

## Notes

`printProfile()`, `getGpuTimeMs()`, and `resetProfileRecords()` internally wait for the previous forward to complete in order to read the GPU timestamps. Calling them every time inside a measurement loop can serialize execution, so call them outside the loop whenever possible.

During profiling, timestamp commands are added before and after tagged nodes. Measurement granularity is at the node level, and even if one node dispatches multiple kernels, the entire node is measured as one section.

## API summary

| API | Role |
|---|---|
| `Node::setProfileTag(tag)` | Assigns a tag. An empty string removes it |
| `NeuralNet::clearProfileTags()` | Removes all tags |
| `NeuralNet::resetProfileRecords()` | Resets the accumulated records |
| `NeuralNet::printProfile(withResetRecords)` | Prints the average. By default, resets the records after printing |
| `NeuralNet::getGpuTimeMs(tag)` | Returns the average GPU time for a specific tag |
