# Z-Image on Android

One dark screen: the image on top, then the prompt, the size (512, 768 or
1024 square), the seed, and the images made so far, with Generate pinned to
the bottom. Tapping it runs all of Z-Image-Turbo on the phone's GPU through
vAi: the Qwen3 caption encoder, 8 DiT steps, then the VAE decode. The result
is shown and saved to `Pictures/ZImage`.

- While it runs, the image card shows the stage ("Step 4 of 8"), the elapsed
  time and, from the second step on, about how long is left: the pace of the
  steps after the first (which also uploads the weights), with the decode
  counted as 0.6 of a step.
- Generate turns into Cancel. A cancelled run stops at its next stage, which
  is at most one DiT step away: no GPU work is in flight between stages. A
  finished decode is never thrown away.
- Recent is a strip of the images in `Pictures/ZImage`, newest first. Tapping
  one shows it. If the app made it, the tap also puts its prompt, seed and
  size back, from `history.tsv` in the app's files, so Generate makes it
  again. A long press opens it in the system viewer.
- Details, folded, has each stage's time and the thermal state.

This app was written for the Galaxy S25 Ultra (Adreno 830). The kernels are
tuned for Adreno, and every forward is split into one submission per node,
because Adreno's watchdog resets the GPU when a submission runs too long.

On the S25 Ultra (seed 42, starting cool; DiT and VAE as measured before
the encoder came in):

| Stage | Time |
|---|---|
| On launch: read the VAE | 0.5–0.8 s |
| New prompt: read the text encoder, encode, read the DiT | 2.2 s + 0.1–0.7 s + 3.0–3.3 s |
| Each of the 8 steps | 6.0 s while cool, rising to 8.6 s as the GPU throttles |
| VAE decode | 4.4 s |
| Total, from button press to image | about 72 s for a new prompt, 65 s for a new seed |

Since the attention kernel was rewritten (12.6 ms per layer instead of 34 ms;
see `benchmarks/zimage/README.md`), the steps run faster. On a phone already
warm from benchmarks they took 7.5 s each, where they had taken 8–12 s warm.

The app samples 8 steps, as diffusers' `ZImagePipeline` example and ComfyUI
do. It used to sample 9; on the same noise the image is of the same quality,
and the run is one step (6–9 s) shorter. The totals above are the 9-step
measurements less one throttled step.

The DiT (3.7 GB) and the text encoder (2.0 GB) are never in memory together.
Holding both with the VAE took 6.2 GB of GPU memory, and with other apps open
the low-memory killer took the app during the VAE decode. A new prompt
instead drops the DiT, builds the encoder, encodes and drops it, then reads
the DiT again. That costs about 5 s and keeps the app at 4.3 GB. A new seed
for the same prompt reuses both the features and the DiT.

Until now, dropping the DiT freed nothing. The caption features, once fed to
the DiT, held the device buffer they had been uploaded to, and that buffer
kept the DiT's whole buffer pool alive. A host-only copy of them now replaces
them first (`dropDit`), and the app's resident memory falls from 3.8 GB to
0.3 GB when the DiT goes.

## Larger images

`generate` takes the image size: a multiple of 16 from 256 to 1024. The
screen offers 512, 768 and 1024; from adb, `--ei size 768` sets it. Past 512, the decode's feature maps do not fit beside
the DiT. At 768 the VAE alone peaks at 1.8 GB of device memory, against 1.1 GB
at 512, and the low-memory killer took the app mid-decode. So the decode runs
alone: the DiT is dropped before it, and the VAE after it, along with the
feature maps its buffer pool keeps. The next image reads both again (about
4.5 s).

On the same phone, new prompt, from a phone warm from earlier runs:

| Size | DiT tokens | Each step | VAE decode | Total |
|---|---|---|---|---|
| 512 | 1056 | 7–8 s | 5.4 s | about 70 s |
| 768 | 2336 | 12–16 s | 10.3 s | about 2.4 min |
| 1024 | 4128 | 24–32 s | 17.0 s | about 4.5 min |

During the 1024 steps the phone had about 1.1 GB of memory left, with the
usual apps in the background; `generate` refuses larger sizes.

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
