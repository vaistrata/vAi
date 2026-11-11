cmake_minimum_required(VERSION 3.21)  # for $<TARGET_RUNTIME_DLLS>

project(bench-yolox)

set(YOLOX_BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(YOLOX_MODEL_DIR ${CMAKE_SOURCE_DIR}/models/experimental/yolox)
set(YOLOX_WEIGHTS_DIR ${CMAKE_SOURCE_DIR}/data/benchmarks/yolox CACHE PATH
    "Directory containing YOLOX benchmark artifacts")
if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

function(bench_yolox_configure_target target_name)
    target_include_directories(${target_name} PRIVATE
        ${YOLOX_BENCH_DIR}/common
        ${YOLOX_MODEL_DIR}
        ${CMAKE_SOURCE_DIR})
    target_compile_definitions(${target_name} PRIVATE
        YOLOX_WEIGHTS_DIR="${YOLOX_WEIGHTS_DIR}")
endfunction()

################################################################################
# vai
################################################################################
add_executable(bench-yolox-vai
    vai/bench.cpp
    common/processing.cpp
)

add_dependencies(bench-yolox-vai experimental-yolox-shaders)

bench_yolox_configure_target(bench-yolox-vai)

target_link_libraries(bench-yolox-vai PRIVATE
    vAi
)

################################################################################
# ncnn
################################################################################
# Optional: set ncnn_DIR to the directory holding ncnnConfig.cmake
# (typically <install>/lib/cmake/ncnn).
find_package(ncnn QUIET CONFIG)

if(ncnn_FOUND)
    message(STATUS "[vAi-bench] ncnn found: ${ncnn_DIR}")

    add_executable(bench-yolox-ncnn
        ncnn/bench.cpp
        common/processing.cpp
        ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
    )

    bench_yolox_configure_target(bench-yolox-ncnn)

    target_link_libraries(bench-yolox-ncnn PRIVATE
        ncnn
    )

    # Windows: copy ncnn (and Vulkan layer) DLLs next to the exe so the binary runs
    # without manually setting PATH. Skips cleanly when ncnn is built as a static
    # library (TARGET_RUNTIME_DLLS empty) by switching the command to "cmake -E true".
    if(WIN32)
        add_custom_command(TARGET bench-yolox-ncnn POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
                $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:bench-yolox-ncnn>>,copy_if_different,true>
                $<TARGET_RUNTIME_DLLS:bench-yolox-ncnn>
                $<TARGET_FILE_DIR:bench-yolox-ncnn>
            COMMAND_EXPAND_LISTS
        )
    endif()
else()
    message(STATUS "[vAi-bench] ncnn not found — skipping bench-yolox-ncnn "
                   "(set ncnn_DIR to a Vulkan-enabled ncnn install)")
endif()

################################################################################
# ONNX Runtime
################################################################################
set(ONNX_MIGRAPHX_GPU_HIP_FLAGS "-Wno-gcc-install-dir-libstdcxx" CACHE STRING
    "Default ONNX Runtime MIGraphX EP HIP compiler flags")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "[vAi-bench] ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")

    add_executable(bench-yolox-onnxruntime
        onnxruntime/bench.cpp
        common/processing.cpp
        ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
    )
    bench_yolox_configure_target(bench-yolox-onnxruntime)
    target_include_directories(bench-yolox-onnxruntime SYSTEM PRIVATE
        "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(bench-yolox-onnxruntime PRIVATE
        "${ONNXRUNTIME_LIBRARY}")

    if(ONNX_MIGRAPHX_GPU_HIP_FLAGS)
        target_compile_definitions(bench-yolox-onnxruntime PRIVATE
            ONNX_MIGRAPHX_GPU_HIP_FLAGS="${ONNX_MIGRAPHX_GPU_HIP_FLAGS}")
    endif()

    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cudart)
        target_link_libraries(bench-yolox-onnxruntime PRIVATE CUDA::cudart)
        target_compile_definitions(bench-yolox-onnxruntime PRIVATE
            VAI_YOLOX_HAVE_CUDA_RUNTIME=1)
    endif()

    find_package(hip QUIET CONFIG)
    if(TARGET hip::amdhip64)
        target_sources(bench-yolox-onnxruntime PRIVATE
            onnxruntime/hip_runtime.cpp)
        target_link_libraries(bench-yolox-onnxruntime PRIVATE
            hip::amdhip64)
        target_compile_definitions(bench-yolox-onnxruntime PRIVATE
            VAI_YOLOX_HAVE_HIP_RUNTIME=1
            __HIP_PLATFORM_AMD__=1)
    endif()
else()
    message(STATUS "[vAi-bench] ONNX Runtime not found — skipping bench-yolox-onnxruntime "
                   "(set ONNXRUNTIME_ROOT)")
endif()

################################################################################
# MIGraphX
################################################################################
set(MIGRAPHX_GPU_HIP_FLAGS "-Wno-gcc-install-dir-libstdcxx" CACHE STRING
    "Default native MIGraphX HIP compiler flags")

find_package(migraphx QUIET CONFIG)

if(migraphx_FOUND AND TARGET migraphx::c)
    message(STATUS "[vAi-bench] MIGraphX found: ${migraphx_DIR}")

    add_executable(bench-yolox-migraphx
        migraphx/bench.cpp
        common/processing.cpp
        ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
    )
    bench_yolox_configure_target(bench-yolox-migraphx)
    target_link_libraries(bench-yolox-migraphx PRIVATE migraphx::c)

    find_package(hip QUIET CONFIG)
    if(TARGET hip::amdhip64)
        target_link_libraries(bench-yolox-migraphx PRIVATE
            hip::amdhip64)
        target_compile_definitions(bench-yolox-migraphx PRIVATE
            VAI_YOLOX_HAVE_HIP_RUNTIME=1
            __HIP_PLATFORM_AMD__=1)
    endif()

    if(MIGRAPHX_GPU_HIP_FLAGS)
        target_compile_definitions(bench-yolox-migraphx PRIVATE
            MIGRAPHX_GPU_HIP_FLAGS_DEFAULT="${MIGRAPHX_GPU_HIP_FLAGS}")
    endif()
else()
    message(STATUS "[vAi-bench] MIGraphX not found — skipping bench-yolox-migraphx "
                   "(set migraphx_DIR or CMAKE_PREFIX_PATH)")
endif()

################################################################################
# TensorRT
################################################################################
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS "${TENSORRT_ROOT}" ENV TensorRT_ROOT
    PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY
    NAMES nvinfer nvinfer_10
    HINTS "${TENSORRT_ROOT}" ENV TensorRT_ROOT
    PATH_SUFFIXES lib lib64 lib/x64 targets/x86_64-linux/lib)
find_package(CUDAToolkit QUIET)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TARGET CUDA::cudart)
    message(STATUS "[vAi-bench] TensorRT found: ${TENSORRT_LIBRARY}")

    add_executable(bench-yolox-tensorrt
        tensorrt/bench.cpp
        common/processing.cpp
        ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
    )
    bench_yolox_configure_target(bench-yolox-tensorrt)
    target_include_directories(bench-yolox-tensorrt SYSTEM PRIVATE
        "${TENSORRT_INCLUDE_DIR}")
    target_link_libraries(bench-yolox-tensorrt PRIVATE
        "${TENSORRT_LIBRARY}"
        CUDA::cudart)
else()
    message(STATUS "[vAi-bench] TensorRT not found — skipping bench-yolox-tensorrt "
                   "(set TENSORRT_ROOT and CUDAToolkit_ROOT)")
endif()

################################################################################
# ExecuTorch
################################################################################
find_package(executorch QUIET CONFIG)

if(executorch_FOUND AND TARGET executorch AND TARGET vulkan_backend)
    find_package(Threads REQUIRED)
    find_package(Vulkan REQUIRED)

    add_executable(bench-yolox-executorch
        executorch/bench.cpp
        common/processing.cpp
        ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
    )
    bench_yolox_configure_target(bench-yolox-executorch)
    target_link_libraries(bench-yolox-executorch PRIVATE
        executorch
        extension_module_static
        extension_runner_util
        extension_tensor
        extension_threadpool
        optimized_native_cpu_ops_lib
        vulkan_backend
        Threads::Threads
        Vulkan::Vulkan
        ${CMAKE_DL_LIBS}
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(bench-yolox-executorch PRIVATE m)
    endif()

    message(STATUS "[vAi-bench] ExecuTorch found: ${executorch_DIR}")
else()
    message(STATUS "[vAi-bench] ExecuTorch not found — skipping bench-yolox-executorch "
                   "(set executorch_DIR to a Vulkan-enabled ExecuTorch install)")
endif()
