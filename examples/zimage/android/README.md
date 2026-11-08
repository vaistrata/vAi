# Z-Image on Android

One screen with a seed field and a Generate button. Tapping the button runs
Z-Image-Turbo on the phone's GPU through vAi: 9 DiT steps, then the VAE
decode. The 512x512 result is shown and saved to `Pictures/ZImage`.

This app was written for the Galaxy S25 Ultra (Adreno 830). The kernels are
tuned for Adreno, and every forward is split into one submission per node,
because Adreno's watchdog resets the GPU when a submission runs too long.

On the S25 Ultra (seed 42, starting cool):

| Stage | Time |
|---|---|
| Load the weights (on launch) | 3.3 s for the DiT, 0.5 s for the VAE |
| Steps 1–9 | 6.0 s per step while cool, rising to 8.6 s as the GPU throttles |
| VAE decode | 4.4 s |
| Total, from button press to image | 73 s |

## What is not on the phone yet

The caption features come from the Qwen3 text encoder, which still runs on
a desktop. `examples/zimage/tools.py prepare "<prompt>"` writes
`demo-input.safetensors` and `prompt.txt`. The app generates that one
prompt, and the seed field changes the noise.

The noise comes from `std::mt19937_64`, not from torch's generator, so a
given seed gives a different image here than in diffusers.

## Build

You need:
- the Android SDK (platform 35)
- the NDK (`ANDROID_NDK_HOME`)
- JDK 17 or newer
- Gradle 9
- `glslc`, which comes with the NDK's shader-tools or the Vulkan SDK

```bash
cd examples/zimage/android
echo "sdk.dir=$ANDROID_HOME" > local.properties
ANDROID_NDK_HOME=/path/to/ndk gradle assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

Gradle runs `build-native.sh` first. That script builds `libzimage-jni.so`
and the shaders with the vAi CMake build, the same one the command-line tools
use. The release APK is signed with the debug key so it installs as is.

## Weights

The app reads its weights from its own external files directory. Open the app
once so the directory exists, then push the files with adb:

```bash
D=/sdcard/Android/data/com.vaistrata.zimage/files/data/zimage
adb shell mkdir -p $D/Z-Image-Turbo/vae
adb push data/zimage/Z-Image-Turbo-q4-split.safetensors $D/
adb push data/zimage/Z-Image-Turbo/vae/. $D/Z-Image-Turbo/vae/
adb push data/zimage/demo-input.safetensors data/zimage/prompt.txt $D/
```

`Z-Image-Turbo-q4-split.safetensors` is the prepared DiT, 3.7 GB. A desktop
run of `demo-zimage --q4 --split-ffn-up` writes it from the checkpoint (see
`models/zimage/tongyi-weights.h`). The app does not convert the 23 GB
checkpoint itself.

## Running from adb

```bash
adb shell am start -S -n com.vaistrata.zimage/.MainActivity --el seed 42
adb logcat -s zimage
```

With `seed` set, the app generates as soon as the weights are loaded. It also
turns the screen on and shows over the lock screen. Without that, the system
freezes a stopped app partway through the run: Samsung's FreecessController
does it within a minute.
