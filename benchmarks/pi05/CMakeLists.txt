cmake_minimum_required(VERSION 3.21)

project(bench-pi05)

# The model owns its definition (net.h) and its checkpoint loader (lerobot-weights.h,
# which resolves data/ckpts/pi05/lerobot/...); the comparison benches take only its
# prompt and noise helpers (processing.h), so every framework gets the same input.
set(MODEL_DIR ${CMAKE_SOURCE_DIR}/models/experimental/pi05)

# vla.cpp and TensorRT Edge-LLM ship no CMake package. Their setup scripts link the
# install at <cache>/vai/vla.cpp and <cache>/vai/TensorRT-Edge-LLM, used here unless
# vla_ROOT or TensorRTEdgeLLM_ROOT names another.
if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_benchCache "$ENV{XDG_CACHE_HOME}/vai")
else()
    set(_benchCache "$ENV{HOME}/.cache/vai")
endif()
if(NOT DEFINED vla_ROOT AND NOT DEFINED ENV{vla_ROOT} AND IS_DIRECTORY "${_benchCache}/vla.cpp")
    set(vla_ROOT "${_benchCache}/vla.cpp")
endif()
if(NOT DEFINED TensorRTEdgeLLM_ROOT)
    set(TensorRTEdgeLLM_ROOT "$ENV{TensorRTEdgeLLM_ROOT}")
    if(NOT TensorRTEdgeLLM_ROOT)
        set(TensorRTEdgeLLM_ROOT "${_benchCache}/TensorRT-Edge-LLM")
    endif()
endif()
unset(_benchCache)

# vla.cpp and TensorRT Edge-LLM run on CUDA: their benches list the CUDA devices to pick one.
find_package(CUDAToolkit QUIET)

# The comparison benches share only the CPU input preparation with vAi: its image
# processor and tokenizer, compiled in without linking vAi itself.
function(bench_pi05_configure_target target_name)
    target_sources(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
        ${CMAKE_SOURCE_DIR}/vai/utils/tokenizer.cpp)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}                    # vai/utils/image-processor.h, tokenizer.h
        ${MODEL_DIR}                           # processing.h
        ${CMAKE_SOURCE_DIR}/benchmarks/common) # bench-inputs.h, bench-utils.h
endfunction()

################################################################################
# vai
################################################################################
add_executable(bench-pi05-vai
    vai/bench.cpp
)

target_include_directories(bench-pi05-vai PRIVATE
    ${MODEL_DIR}                           # net.h, lerobot-weights.h, processing.h
    ${CMAKE_SOURCE_DIR}/benchmarks/common  # bench-inputs.h, bench-utils.h
)

target_link_libraries(bench-pi05-vai PRIVATE
    vAi
)

################################################################################
# vla.cpp
################################################################################
# Optional, Linux only. vla.cpp installs libvla and vla.h but no record of its release,
# so the release is read from the converter setup-vlacpp.sh keeps next to the install.
find_path(vla_INCLUDE_DIR NAMES vla.h HINTS "${vla_ROOT}" ENV vla_ROOT PATH_SUFFIXES include)
find_library(vla_LIBRARY NAMES vla HINTS "${vla_ROOT}" ENV vla_ROOT PATH_SUFFIXES lib lib64)
mark_as_advanced(vla_INCLUDE_DIR vla_LIBRARY)

set(_vlaVersion "")
if(vla_INCLUDE_DIR AND vla_LIBRARY)
    get_filename_component(_vlaPrefix "${vla_INCLUDE_DIR}" DIRECTORY)
    set(_vlaVersion "of unknown release")
    if(EXISTS "${_vlaPrefix}/convert/pyproject.toml")
        file(STRINGS "${_vlaPrefix}/convert/pyproject.toml" _vlaLine REGEX "^version *= *\"[^\"]+\"" LIMIT_COUNT 1)
        if(_vlaLine)
            string(REGEX REPLACE "^version *= *\"([^\"]+)\".*" "v\\1" _vlaVersion "${_vlaLine}")
        endif()
        unset(_vlaLine)
    endif()
endif()

if(_vlaVersion AND NOT _vlaVersion STREQUAL "v0.4.0")
    # The bench reads the pinned release's config fields, and its numbers are recorded
    # against that release.
    message(WARNING "[vAi-bench] vla.cpp ${_vlaVersion} at ${_vlaPrefix} != pinned v0.4.0: "
                    "skipping bench-pi05-vlacpp. Run benchmarks/scripts/setup-vlacpp.sh, "
                    "or point vla_ROOT at the v0.4.0 install it makes.")
elseif(_vlaVersion AND NOT CUDAToolkit_FOUND)
    message(STATUS "[vAi-bench] vla.cpp found at ${_vlaPrefix}, but not the CUDA toolkit: skipping bench-pi05-vlacpp. "
                   "Point CUDAToolkit_ROOT at the CUDA that vla.cpp was built with.")
elseif(_vlaVersion)
    message(STATUS "[vAi-bench] vla.cpp found: ${_vlaPrefix} (${_vlaVersion})")

    if(NOT TARGET vla::vla)
        add_library(vla::vla UNKNOWN IMPORTED)
        set_target_properties(vla::vla PROPERTIES
            IMPORTED_LOCATION "${vla_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${vla_INCLUDE_DIR}")
    endif()

    find_package(Threads REQUIRED)
    add_executable(bench-pi05-vlacpp
        vla.cpp/bench.cpp
    )
    bench_pi05_configure_target(bench-pi05-vlacpp)
    # The bench's runtime config, which sets FP16 weights.
    target_compile_definitions(bench-pi05-vlacpp PRIVATE
        VLACPP_RUNTIME_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/vla.cpp/runtime.json")
    target_link_libraries(bench-pi05-vlacpp PRIVATE
        vla::vla
        CUDA::cudart
        Threads::Threads
    )
else()
    message(STATUS "[vAi-bench] vla.cpp not found, skipping bench-pi05-vlacpp. "
                   "For it run benchmarks/scripts/setup-vlacpp.sh, or point vla_ROOT at an existing install.")
endif()
unset(_vlaVersion)
unset(_vlaPrefix)

################################################################################
# MIGraphX
################################################################################
# Optional, Linux only. setup-migraphx.sh registers its install in CMake's package
# registry. ROCm's own folder is searched first, so the pinned release is asked for by
# version: a MIGraphX of another release that ROCm carries is passed over for it.
if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

set(MIGRAPHX_GPU_HIP_FLAGS "-Wno-gcc-install-dir-libstdcxx" CACHE STRING
    "Default native MIGraphX HIP compiler flags")

# The pinned release first; else any, for the warning below to name.
find_package(migraphx 2.18...<2.19 QUIET CONFIG)
if(NOT migraphx_FOUND)
    find_package(migraphx QUIET CONFIG)
endif()
find_package(hip QUIET CONFIG)

# The GEMM backend the bench compiles its graphs with: mlir fuses the dots into MLIR
# kernels; hipblaslt lowers every dot to hipBLASLt with MLIR off. Either way a dot left
# outside MLIR goes to hipBLASLt. mlir is the faster on the bundle export-onnx.py writes
# by default, with its constants folded; hipblaslt on one exported with --no-constant-folding.
set(MIGRAPHX_GEMM_PROVIDER "mlir" CACHE STRING "GEMM backend of the MIGraphX bench: mlir or hipblaslt")
set_property(CACHE MIGRAPHX_GEMM_PROVIDER PROPERTY STRINGS mlir hipblaslt)
if(NOT MIGRAPHX_GEMM_PROVIDER MATCHES "^(hipblaslt|mlir)$")
    message(FATAL_ERROR "[vAi-bench] MIGRAPHX_GEMM_PROVIDER must be mlir or hipblaslt, not '${MIGRAPHX_GEMM_PROVIDER}'")
endif()

set(_migraphxFound OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET migraphx::c AND (TARGET hip::amdhip64 OR TARGET hip::host))
    set(_migraphxFound ON)
    # The release, from the package version or else from the header the bench prints it from.
    set(_migraphxVersion "${migraphx_VERSION_MAJOR}.${migraphx_VERSION_MINOR}")
    if(_migraphxVersion STREQUAL ".")
        get_target_property(_migraphxIncludes migraphx::c INTERFACE_INCLUDE_DIRECTORIES)
        find_file(_migraphxVersionHeader migraphx/version.h HINTS ${_migraphxIncludes} NO_DEFAULT_PATH NO_CACHE)
        set(_migraphxVersion "of unknown release")
        if(_migraphxVersionHeader)
            file(STRINGS "${_migraphxVersionHeader}" _migraphxLines REGEX "MIGRAPHX_VERSION_(MAJOR|MINOR) ")
            string(REGEX REPLACE ".*MIGRAPHX_VERSION_MAJOR ([0-9]+).*" "\\1" _major "${_migraphxLines}")
            string(REGEX REPLACE ".*MIGRAPHX_VERSION_MINOR ([0-9]+).*" "\\1" _minor "${_migraphxLines}")
            set(_migraphxVersion "${_major}.${_minor}")
            unset(_major)
            unset(_minor)
            unset(_migraphxLines)
        endif()
        unset(_migraphxVersionHeader)
        unset(_migraphxIncludes)
    endif()
endif()

if(_migraphxFound AND NOT _migraphxVersion STREQUAL "2.18")
    # The bench reads the graphs of the pinned release's export and binds its output
    # names, and its numbers are recorded against that release.
    message(WARNING "[vAi-bench] MIGraphX ${_migraphxVersion} at ${migraphx_DIR} != pinned 2.18 (rocm-10.1): "
                    "skipping bench-pi05-migraphx. Run benchmarks/scripts/setup-migraphx.sh, "
                    "or point migraphx_DIR at an install of 2.18.")
elseif(_migraphxFound)
    message(STATUS "[vAi-bench] MIGraphX found: ${migraphx_DIR} (${_migraphxVersion}), GEMM through ${MIGRAPHX_GEMM_PROVIDER}")

    add_executable(bench-pi05-migraphx
        migraphx/bench.cpp
    )
    bench_pi05_configure_target(bench-pi05-migraphx)
    target_link_libraries(bench-pi05-migraphx PRIVATE migraphx::c)
    # setup-migraphx.sh puts the ROCm runtime MIGraphX was built for next to libmigraphx_c.
    # The bench then links that HIP rather than the system's, so its RUNPATH names no
    # system ROCm folder and every ROCm library in the process comes from one release;
    # the system HIP provides the headers.
    get_filename_component(_migraphxLibDir "${migraphx_DIR}/../.." ABSOLUTE)
    if(EXISTS "${_migraphxLibDir}/libamdhip64.so.7" AND TARGET hip::amdhip64)
        # The system HIP's include folder also holds that ROCm's own MIGraphX headers. Linked
        # after migraphx::c, its folder comes after MIGraphX's on the command line, so
        # <migraphx/...> resolves to the pinned release.
        add_library(bench-pi05-hip-headers INTERFACE)
        get_target_property(_hipIncludes hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(_hipDefinitions hip::amdhip64 INTERFACE_COMPILE_DEFINITIONS)
        if(_hipIncludes)
            target_include_directories(bench-pi05-hip-headers SYSTEM INTERFACE ${_hipIncludes})
        endif()
        if(_hipDefinitions)
            target_compile_definitions(bench-pi05-hip-headers INTERFACE ${_hipDefinitions})
        endif()
        target_link_libraries(bench-pi05-migraphx PRIVATE
            bench-pi05-hip-headers "${_migraphxLibDir}/libamdhip64.so.7")
        message(STATUS "[vAi-bench] bench-pi05-migraphx runs on the ROCm runtime in ${_migraphxLibDir}")
        unset(_hipIncludes)
        unset(_hipDefinitions)
    elseif(TARGET hip::amdhip64)
        target_link_libraries(bench-pi05-migraphx PRIVATE hip::amdhip64)
    else()
        target_link_libraries(bench-pi05-migraphx PRIVATE hip::host)
    endif()
    unset(_migraphxLibDir)
    target_compile_definitions(bench-pi05-migraphx PRIVATE
        __HIP_PLATFORM_AMD__=1
        MIGRAPHX_GPU_HIP_FLAGS_DEFAULT="${MIGRAPHX_GPU_HIP_FLAGS}"
        MIGRAPHX_GEMM_PROVIDER_DEFAULT="${MIGRAPHX_GEMM_PROVIDER}"
    )
else()
    message(STATUS "[vAi-bench] MIGraphX or HIP not found on Linux, skipping bench-pi05-migraphx. "
                   "For it run benchmarks/scripts/setup-migraphx.sh, or point migraphx_DIR at an existing install.")
endif()
unset(_migraphxFound)
unset(_migraphxVersion)

################################################################################
# TensorRT Edge-LLM (pi0.5 runtime)
################################################################################
# Optional, Linux only. Edge-LLM installs no CMake package, so this reads the layout
# setup-trt-edge-llm.sh installs: headers under include/, libraries under lib/, and the
# SM its kernels were built for in .cuda-arch.
set(_edgellmVersion "")
set(_edgellmVersionHeader "${TensorRTEdgeLLM_ROOT}/include/tensorrt_edgellm/common/version.h")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${_edgellmVersionHeader}"
   AND EXISTS "${TensorRTEdgeLLM_ROOT}/.cuda-arch")
    set(_edgellmVersion "of unknown release")
    file(STRINGS "${_edgellmVersionHeader}" _edgellmLine REGEX "kRUNTIME_VERSION = \"[^\"]+\"" LIMIT_COUNT 1)
    if(_edgellmLine)
        string(REGEX REPLACE ".*kRUNTIME_VERSION = \"([^\"]+)\".*" "\\1" _edgellmVersion "${_edgellmLine}")
    endif()
    unset(_edgellmLine)
endif()
unset(_edgellmVersionHeader)

find_path(TensorRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}" ENV TENSORRT_ROOT PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY nvinfer HINTS "${TENSORRT_ROOT}" ENV TENSORRT_ROOT PATH_SUFFIXES lib lib64)
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)

if(_edgellmVersion AND NOT _edgellmVersion STREQUAL "0.11.1")
    # The bench calls the pinned release's experimental pi0.5 runtime, whose API other
    # releases change, and its numbers are recorded against that release.
    message(WARNING "[vAi-bench] TensorRT Edge-LLM ${_edgellmVersion} at ${TensorRTEdgeLLM_ROOT} != pinned 0.11.1: "
                    "skipping bench-pi05-trt. Run benchmarks/scripts/setup-trt-edge-llm.sh, "
                    "or point TensorRTEdgeLLM_ROOT at the v0.11.1 install it makes.")
elseif(_edgellmVersion AND NOT (CUDAToolkit_FOUND AND TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY))
    message(STATUS "[vAi-bench] TensorRT Edge-LLM found at ${TensorRTEdgeLLM_ROOT}, but not the CUDA toolkit "
                   "or the TensorRT SDK it was built with: skipping bench-pi05-trt. "
                   "Point CUDAToolkit_ROOT and TENSORRT_ROOT at them.")
elseif(_edgellmVersion)
    message(STATUS "[vAi-bench] TensorRT Edge-LLM found: ${TensorRTEdgeLLM_ROOT} (${_edgellmVersion})")

    # edgellmCore holds relocatable device code, which a C++ link leaves unresolved: an
    # empty CUDA source gives the bench a device link, for the SM the setup built for.
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
    endif()
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()
    enable_language(CUDA)
    file(STRINGS "${TensorRTEdgeLLM_ROOT}/.cuda-arch" _edgellmArch LIMIT_COUNT 1)
    file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/edgellm-device-link.cu" CONTENT "")

    find_package(Threads REQUIRED)
    add_executable(bench-pi05-trt
        trt/bench.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/edgellm-device-link.cu
    )
    bench_pi05_configure_target(bench-pi05-trt)
    set_target_properties(bench-pi05-trt PROPERTIES
        CUDA_ARCHITECTURES "${_edgellmArch}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared     # the libcudart Edge-LLM links, not a static copy
        CUDA_STANDARD 17)               # nvcc has no C++23, the project's CXX_STANDARD
    target_include_directories(bench-pi05-trt PRIVATE
        ${TensorRTEdgeLLM_ROOT}/include                   # runtime/pi05Policy.h, nlohmann/json.hpp
        ${TensorRTEdgeLLM_ROOT}/include/tensorrt_edgellm  # common/trtUtils.h
        ${TensorRT_INCLUDE_DIR})                          # NvInfer.h
    # vAi's image processor keeps its stb implementation local; Edge-LLM defines stb symbols too.
    target_compile_definitions(bench-pi05-trt PRIVATE
        STB_IMAGE_STATIC
        VAI_EDGELLM_PLUGIN_PATH="${TensorRTEdgeLLM_ROOT}/lib/libNvInfer_edgellm_plugin.so"
    )
    # The libraries upstream's pi05_policy_inference links, without those the bench never calls.
    target_link_libraries(bench-pi05-trt PRIVATE
        ${TensorRTEdgeLLM_ROOT}/lib/libpi05Experimental.a
        ${TensorRTEdgeLLM_ROOT}/lib/libedgellmCore.a
        ${TensorRTEdgeLLM_ROOT}/lib/libedgellmChatTemplate.a
        ${TensorRTEdgeLLM_ROOT}/lib/libxgrammarCore.a
        ${TensorRT_LIBRARY}
        CUDA::cudart
        ${CMAKE_DL_LIBS}
        Threads::Threads
    )
    unset(_edgellmArch)
else()
    message(STATUS "[vAi-bench] TensorRT Edge-LLM not found, skipping bench-pi05-trt. "
                   "For it run benchmarks/scripts/setup-trt-edge-llm.sh, or point TensorRTEdgeLLM_ROOT at the install it makes.")
endif()
unset(_edgellmVersion)
