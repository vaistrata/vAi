cmake_minimum_required(VERSION 3.21)

project(bench-llama3)

# The model owns its definition (net.h) and its checkpoint loader
# (meta-weights.h, which resolves data/ckpts/llama3/meta/...), so the bench and models/production/llama3/demo.cpp
# can never end up on different weights.
set(MODEL_DIR ${CMAKE_SOURCE_DIR}/models/production/llama3)

################################################################################
# vai
################################################################################
add_executable(bench-llama3-vai
    vai/bench.cpp
)

target_include_directories(bench-llama3-vai PRIVATE
    ${MODEL_DIR}                           # net.h, meta-weights.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common            # bench-prompts.h
)

target_link_libraries(bench-llama3-vai PRIVATE
    vAi
)

################################################################################
# llama.cpp
################################################################################
# Optional. benchmarks/scripts/setup-llamacpp.sh installs into the user cache and
# registers it in CMake's package registry, so no path is needed here; an
# explicit -Dllama_DIR or CMAKE_PREFIX_PATH still takes precedence.
# NO_SYSTEM_ENVIRONMENT_PATH: CMake searches prefixes derived from $PATH
# (~/.local/bin -> ~/.local) before the registry, so any other llama.cpp
# installed there would shadow the pinned one.
find_package(llama QUIET CONFIG NO_SYSTEM_ENVIRONMENT_PATH)

if(llama_FOUND)
    message(STATUS "[vAi-bench] llama.cpp found: ${llama_DIR} (build ${LLAMA_BUILD_NUMBER})")

    # The installed llama.cpp is a Release (/MD) static library; a Debug (/MDd)
    # bench fails to link against it with LNK2038 on MSVC.
    if(CMAKE_CONFIGURATION_TYPES)
        message(STATUS "[vAi-bench] llama.cpp was built Release: build bench-llama3-llamacpp with --config Release")
    endif()

    if(NOT LLAMA_BUILD_NUMBER EQUAL 10426)
        message(WARNING "[vAi-bench] llama.cpp build ${LLAMA_BUILD_NUMBER} != pinned b10426 — "
                        "recorded numbers are not comparable, and the [EXPERIMENTAL] "
                        "sampling API this bench uses may break the build or silently "
                        "change what gets sampled")
    endif()

    add_executable(bench-llama3-llamacpp
        llama.cpp/bench.cpp
    )

    target_include_directories(bench-llama3-llamacpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common      # bench-prompts.h
    )

    target_link_libraries(bench-llama3-llamacpp PRIVATE
        llama
    )
else()
    message(STATUS "[vAi-bench] llama.cpp not found — building bench-llama3-vai only. "
                   "For the llama.cpp counterpart run benchmarks/scripts/setup-llamacpp.sh, "
                   "or point llama_DIR at an existing install.")
endif()
