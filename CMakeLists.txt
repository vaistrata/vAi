cmake_minimum_required(VERSION 3.16)

project(vAi VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable folder organization in Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Output directories (build directory already contains config name via cmake.buildDirectory)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build options
option(VAI_BUILD_MODEL_DEMO "Build each model's demo (demo-<model>)" ON)
option(VAI_BUILD_EXAMPLES "Build vAi examples (examples/: mnist, chat-*)" OFF)
option(VAI_BUILD_TESTS "Build vAi test projects" OFF)
option(VAI_BUILD_BENCH "Build vAi benchmark projects" OFF)
option(VAI_BUILD_PYTHON "Build vAi Python bindings (nanobind)" OFF)

# Default data root — models resolve their checkpoints under here.
add_definitions(
    -DVAI_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)


# External dependencies directory
if(NOT DEFINED VAI_EXTERNAL_DIR)
    # Use vAi's own external directory
    set(VAI_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
endif()

# Set position independent code for Python bindings (required for nanobind)
if(VAI_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


################################################################################
# Easy Vulkan API Library
################################################################################
if(NOT TARGET eva)
    message(STATUS "[vAi] Configuring EVA library...")
    add_subdirectory(${VAI_EXTERNAL_DIR}/eva)
    message(STATUS "[vAi] EVA library configuration completed.")
    target_compile_options(eva PRIVATE -w)
else()
    message(STATUS "[vAi] Using existing EVA target")
endif()

# Put vai/shaders on glslc's include path so a shader kept outside it (a
# test-local kernel, say) can still pull in common/*.h by the same path its
# neighbours use. eva_compile_shaders takes no include-path argument, but it
# splices EVA_GLSLC straight into the command line.
list(APPEND EVA_GLSLC -I ${CMAKE_CURRENT_SOURCE_DIR}/vai/shaders)


################################################################################
# vAi Library
################################################################################
# Collect shader files
file(GLOB VAI_SHADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vai/shaders/*.comp)

# Create static library
add_library(vAi STATIC
    vai/engine/neural-net.h
    vai/engine/graph-flow.h
    vai/engine/graph-controller.h
    vai/engine/node-profiler.h
    vai/engine/tensor.h
    vai/engine/buffer-pool.h
    vai/engine/error.h
    vai/engine/safetensors-parser.h
    vai/engine/weights.h
    vai/utils/image-processor.h
    vai/utils/tokenizer.h

    vai/engine/neural-net.cpp
    vai/engine/safetensors-parser.cpp
    vai/engine/tensor.cpp
    vai/utils/image-processor.cpp
    vai/utils/tokenizer.cpp

    ${VAI_SHADERS}
)

# Auto-detect and compile shaders
eva_auto_compile_shaders(vAi ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# Define shader output directory for runtime
target_compile_definitions(vAi
PUBLIC
    SHADER_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

# Include directories. Only the repo root, so a header in another folder is spelled in full
# (vai/nodes/linear.h, models/<impl>/<model>/net.h) and no two folders can shadow each other.
# A header in the same folder or below is included by its relative path.
target_include_directories(vAi
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link eva (PUBLIC so external projects can access eva too)
target_link_libraries(vAi
PUBLIC
    eva
)


################################################################################
# Models with kernels of their own
################################################################################
add_subdirectory(models)
add_compile_definitions(${VAI_MODEL_SHADER_DEFINITIONS})


################################################################################
# Examples
################################################################################
if(VAI_BUILD_EXAMPLES)
    message(STATUS "[vAi] Building example demos...")
    add_subdirectory(examples)
    message(STATUS "[vAi] Example demo build configuration completed.")
endif()


################################################################################
# Benchmarks
################################################################################
if(VAI_BUILD_BENCH)
    message(STATUS "[vAi] Building benchmark projects...")

    # Vision benchmarks
    # add_subdirectory(benchmarks/yolox)

    # LLM benchmarks
    add_subdirectory(benchmarks/gpt2)
    add_subdirectory(benchmarks/llama3)

    # VLM benchmarks
    add_subdirectory(benchmarks/gemma3)

    message(STATUS "[vAi] Benchmark projects build configuration completed.")
endif()


################################################################################
# Python Bindings (nanobind)
################################################################################
if(VAI_BUILD_PYTHON)
    add_subdirectory(python)
endif()


################################################################################
# Tests
################################################################################
if(VAI_BUILD_TESTS)
    message(STATUS "[vAi] Building test projects...")
    add_subdirectory(tests)
    message(STATUS "[vAi] Test projects build configuration completed.")
endif()
