# syntax=docker/dockerfile:1
#==============================================================================
# vAi environment image: run the C++ demos and pyvai.
#
#   docker build -f docker/Dockerfile -t vai .      # from the repo root
#
# Inside: `cmake --build build` rebuilds the C++ demos, and
# `uv pip install -e .` refreshes the bindings.
#==============================================================================

# Pin the digest for a tagged release: ubuntu:24.04@sha256:...
ARG BASE_IMAGE=ubuntu:24.04

# The LunarG SDK release.
ARG VULKAN_SDK_VERSION=1.4.357.0
ARG UV_VERSION=0.12.7


# Named so the pin above drives it; uv ships as a static binary.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


################################################################################
# sdk — the LunarG tarball, cut down to what the image uses
################################################################################
FROM ${BASE_IMAGE} AS sdk
ARG VULKAN_SDK_VERSION
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        binutils ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

# The whole SDK is 1.8 GB unpacked. runtime/ goes to `base`, dev/ to `dev`.
WORKDIR /sdk
RUN file="vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
    && url="https://sdk.lunarg.com/sdk" \
    && root="${VULKAN_SDK_VERSION}/x86_64" \
    && curl -fsSL --retry 3 "$url/download/${VULKAN_SDK_VERSION}/linux/$file" -o "$file" \
    && curl -fsSL --retry 3 "$url/sha/${VULKAN_SDK_VERSION}/linux/$file.txt" | sha256sum -c - \
    && mkdir runtime dev \
    && tar -xJf "$file" -C runtime --strip-components=2 --wildcards \
        "$root/lib/VulkanLoader/lib/libvulkan.so*" \
        "$root/bin/vulkaninfo" \
    && tar -xJf "$file" -C dev --strip-components=2 --wildcards \
        "$root/include" \
        "$root/bin/glslc" "$root/bin/glslang*" "$root/bin/spirv-*" \
        "$root/lib/libVkLayer_khronos_validation.so" \
        "$root/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json" \
    && strip --strip-debug dev/lib/libVkLayer_khronos_validation.so \
    && rm "$file"


################################################################################
# base — Vulkan runtime: loader + in-image ICD
################################################################################
FROM ${BASE_IMAGE} AS base
ARG DEBIAN_FRONTEND=noninteractive

# Mesa is the AMD/Intel ICD (NVIDIA's is injected at run time). The x11/egl
# libs are what NVIDIA's ICD needs headless — missing, it silently falls
# back to llvmpipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers \
        libx11-6 libxext6 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Mesa pulls in Ubuntu's libvulkan1; LD_LIBRARY_PATH puts the SDK loader first.
COPY --from=sdk /sdk/runtime/ /opt/vulkan-sdk/

ENV PATH=/opt/vulkan-sdk/bin:$PATH \
    LD_LIBRARY_PATH=/opt/vulkan-sdk/lib/VulkanLoader/lib

# `graphics` is required, or the toolkit skips the Vulkan ICD.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics


################################################################################
# dev — C++ demos + pyvai, built from the source in the image
################################################################################
FROM base AS dev
ARG DEBIAN_FRONTEND=noninteractive

# GCC 14 is the compiler CI builds with; the distro's default, 13, is too old.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        build-essential gcc-14 g++-14 cmake ninja-build \
        python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 \
    CXX=g++-14

# CMAKE_PREFIX_PATH is how FindVulkan finds the SDK's loader to link.
COPY --from=sdk /sdk/dev/ /opt/vulkan-sdk/

ENV VULKAN_SDK=/opt/vulkan-sdk \
    VK_ADD_LAYER_PATH=/opt/vulkan-sdk/share/vulkan/explicit_layer.d \
    CMAKE_PREFIX_PATH=/opt/vulkan-sdk:/opt/vulkan-sdk/lib/VulkanLoader

COPY --from=uv /uv /uvx /bin/

# The venv the bindings install into (PEP 668 blocks the system one). Built on
# the distro python3, not a uv-managed one: nanobind compiles against the
# headers python3-dev provides.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy
RUN uv venv --python python3 /opt/venv

# /opt/vai is baked into VAI_DATA_DIR / SHADER_OUTPUT_DIR at configure time:
# shaders load from build/, checkpoints mount at data/.
WORKDIR /opt/vai
COPY . .

# The cache mount keeps uv's downloads across builds, so a source change does
# not re-fetch the build backend.
RUN --mount=type=cache,target=/root/.cache/uv \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DVAI_BUILD_EXAMPLES=ON \
    && cmake --build build \
    && uv pip install -e .

ENV PATH=/opt/vai/build/bin:$PATH

CMD ["bash"]
