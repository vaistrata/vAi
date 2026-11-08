#!/bin/sh
# Builds libzimage-jni.so for arm64-v8a with the NDK and copies it, with the
# shaders it loads, to where Gradle packages them:
#   app/src/main/jniLibs/arm64-v8a/libzimage-jni.so
#   app/src/main/assets/shaders/{vai,zimage}/*.comp.spv
# The NDK is $ANDROID_NDK_HOME, $ANDROID_NDK_ROOT or the newest under
# $ANDROID_HOME/ndk. Run by Gradle before every build (ninja keeps it cheap).
set -e
here=$(cd "$(dirname "$0")" && pwd)
build=$here/native/build

ndk=${ANDROID_NDK_HOME:-$ANDROID_NDK_ROOT}
if [ -z "$ndk" ] && [ -n "$ANDROID_HOME" ] && [ -d "$ANDROID_HOME/ndk" ]; then
    ndk=$ANDROID_HOME/ndk/$(ls "$ANDROID_HOME/ndk" | sort -V | tail -1)
fi
if [ ! -f "$ndk/build/cmake/android.toolchain.cmake" ]; then
    echo "build-native.sh: no NDK found; set ANDROID_NDK_HOME" >&2
    exit 1
fi

if [ ! -f "$build/build.ninja" ]; then
    cmake -S "$here/native" -B "$build" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 \
        -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$build" --target zimage-jni

libs=$here/app/src/main/jniLibs/arm64-v8a
assets=$here/app/src/main/assets/shaders
mkdir -p "$libs" "$assets/vai" "$assets/zimage"
# Stripped here: Gradle strips only with an NDK of its own SDK.
"$(ls "$ndk"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)" --strip-unneeded \
    -o "$libs/libzimage-jni.so" "$build/libzimage-jni.so"
# One .spv per shader source: the build directory keeps those of deleted ones.
root=$here/../../..
rm -f "$assets"/vai/*.spv "$assets"/zimage/*.spv
for src in "$root"/vai/shaders/*.comp; do
    cp "$build/vai/shaders/$(basename "$src").spv" "$assets/vai/"
done
for src in "$root"/models/zimage/shaders/*.comp; do
    cp "$build/vai/models/zimage/shaders/$(basename "$src").spv" "$assets/zimage/"
done
