cmake_minimum_required(VERSION 3.21)

project(bench-gpt2)

# The model owns its definition (net.h) and its checkpoint loader
# (openai-weights.h, which resolves data/ckpts/gpt2/openai/...), so the bench and models/production/gpt2/demo.cpp
# can never end up on different weights.
set(MODEL_DIR ${CMAKE_SOURCE_DIR}/models/production/gpt2)

################################################################################
# vai
################################################################################
add_executable(bench-gpt2-vai
    vai/bench.cpp
)

target_include_directories(bench-gpt2-vai PRIVATE
    ${MODEL_DIR}                           # net.h, openai-weights.h
    ${CMAKE_SOURCE_DIR}/benchmarks/common  # bench-inputs.h, bench-utils.h
)

target_link_libraries(bench-gpt2-vai PRIVATE
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

if(llama_FOUND AND NOT LLAMA_BUILD_NUMBER EQUAL 11960)
    # The bench calls the pinned release's batch and backend-sampling API, which other
    # builds lack or change, and its numbers are recorded against that release.
    message(WARNING "[vAi-bench] llama.cpp build ${LLAMA_BUILD_NUMBER} at ${llama_DIR} != pinned v0.8.0 (b11960): "
                    "skipping bench-gpt2-llamacpp. Run benchmarks/scripts/setup-llamacpp.sh, "
                    "or point llama_DIR at a v0.8.0 install.")
elseif(llama_FOUND)
    message(STATUS "[vAi-bench] llama.cpp found: ${llama_DIR} (build ${LLAMA_BUILD_NUMBER})")

    # The installed llama.cpp is a Release (/MD) static library; a Debug (/MDd)
    # bench fails to link against it with LNK2038 on MSVC.
    if(CMAKE_CONFIGURATION_TYPES)
        message(STATUS "[vAi-bench] llama.cpp was built Release: build bench-gpt2-llamacpp with --config Release")
    endif()

    add_executable(bench-gpt2-llamacpp
        llama.cpp/bench.cpp
    )

    target_include_directories(bench-gpt2-llamacpp PRIVATE
        ${CMAKE_SOURCE_DIR}/benchmarks/common   # bench-inputs.h, bench-utils.h
    )

    target_link_libraries(bench-gpt2-llamacpp PRIVATE
        llama
    )
else()
    message(STATUS "[vAi-bench] llama.cpp not found, building bench-gpt2-vai only. "
                   "For the llama.cpp counterpart run benchmarks/scripts/setup-llamacpp.sh, "
                   "or point llama_DIR at an existing install.")
endif()
