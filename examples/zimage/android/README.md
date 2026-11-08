# Z-Image on Android

One screen with a prompt, a seed and a Generate button. Tapping the button
runs all of Z-Image-Turbo on the phone's GPU through vAi: the Qwen3 caption
encoder, 9 DiT steps, then the VAE decode. The 512x512 result is shown and
saved to `Pictures/ZImage`.

This app was written for the Galaxy S25 Ultra (Adreno 830). The kernels are
tuned for Adreno, and every forward is split into one submission per node,
because Adreno's watchdog resets the GPU when a submission runs too long.

On the S25 Ultra (seed 42, starting cool; DiT and VAE as measured before
the encoder came in):

| Stage | Time |
|---|---|
| On launch: read the VAE | 0.5–0.8 s |
| New prompt: read the text encoder, encode, read the DiT | 2.2 s + 0.1–0.7 s + 3.0–3.3 s |
| Steps 1–9 | 6.0 s per step while cool, rising to 8.6 s as the GPU throttles |
| VAE decode | 4.4 s |
| Total, from button press to image | about 80 s for a new prompt, 73 s for a new seed |

The DiT (3.7 GB) and the text encoder (2.0 GB) are never in memory together.
Holding both with the VAE took 6.2 GB of GPU memory, and with other apps open
the low-memory killer took the app during the VAE decode. A new prompt
instead drops the DiT, builds the encoder, encodes and drops it, then reads
the DiT again. That costs about 5 s and keeps the app at 4.3 GB. A new seed
for the same prompt reuses both the features and the DiT.

The text is NFC-normalized first (`java.text.Normalizer`), as Qwen3's
tokenizer expects, and cut off at 512 tokens, as the pipeline does. Without
the encoder's files the app falls back to the fixed caption features of
`demo-input.safetensors` (written by `examples/zimage/tools.py prepare
"<prompt>"`, with `prompt.txt`), and only the seed can change.

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
adb shell mkdir -p $D/Z-Image-Turbo/vae $D/Z-Image-Turbo/tokenizer
adb push data/zimage/Z-Image-Turbo-q4-split.safetensors $D/
adb push data/zimage/Z-Image-Turbo-text-encoder-q4.safetensors $D/
adb push data/zimage/Z-Image-Turbo/tokenizer/tokenizer.json $D/Z-Image-Turbo/tokenizer/
adb push data/zimage/Z-Image-Turbo/vae/. $D/Z-Image-Turbo/vae/
```

The app does not convert the checkpoints itself; the prepared files are
written on a desktop:

- `Z-Image-Turbo-q4-split.safetensors` is the prepared DiT, 3.7 GB. A run of
  `demo-zimage --q4 --split-ffn-up` writes it from the 23 GB checkpoint (see
  `models/zimage/tongyi-weights.h`).
- `Z-Image-Turbo-text-encoder-q4.safetensors` is the prepared caption
  encoder, 2.8 GB: `python3 examples/zimage/tools.py text-encoder` writes it
  from the 8 GB `text_encoder/` checkpoint, without a GPU.

## Running from adb

```bash
adb shell am start -S -n com.vaistrata.zimage/.MainActivity --el seed 42 --es prompt "'A red fox in the snow'"
adb logcat -s zimage
```

With `seed` set, the app generates (from `prompt` if given, else the one in
the field) as soon as the weights are loaded. It also
turns the screen on and shows over the lock screen. Without that, the system
freezes a stopped app partway through the run: Samsung's FreecessController
does it within a minute.
