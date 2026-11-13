cmake_minimum_required(VERSION 3.21)

project(bench-gemma3)

# The model owns its definition (net.h) and checkpoint loader, so the bench
# and models/production/gemma3/demo.cpp always use the same model implementation and weights.
set(MODEL_DIR ${CMAKE_SOURCE_DIR}/models/production/gemma3)

################################################################################
# vai
################################################################################
add_executable(bench-gemma3-vai
    vai/bench.cpp
)

target_include_directories(bench-gemma3-vai PRIVATE
    ${MODEL_DIR}                         # net.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common   # bench-prompts.h
)

target_link_libraries(bench-gemma3-vai PRIVATE
    vAi
)

################################################################################
# llama.cpp
################################################################################
# Optional. benchmarks/scripts/setup-llamacpp.sh installs into the user cache
# (with LLAMA_BUILD_MTMD=ON) and registers it in CMake's package registry, so no
# path is needed here; an explicit -Dllama_DIR or CMAKE_PREFIX_PATH still takes
# precedence.
# NO_SYSTEM_ENVIRONMENT_PATH: CMake searches prefixes derived from $PATH
# (~/.local/bin -> ~/.local) before the registry, so any other llama.cpp
# installed there would shadow the pinned one.
find_package(llama QUIET CONFIG NO_SYSTEM_ENVIRONMENT_PATH)

if(llama_FOUND)
    message(STATUS "[vAi-bench] llama.cpp found: ${llama_DIR} (build ${LLAMA_BUILD_NUMBER})")

    # mtmd is installed alongside llama.cpp, but currently does not provide its
    # own CMake package config, so resolve it from llama's install lib directory.
    find_library(MTMD_LIBRARY
        NAMES mtmd
        HINTS ${LLAMA_LIB_DIR}
    )

    if(MTMD_LIBRARY)
        message(STATUS "[vAi-bench] mtmd found: ${MTMD_LIBRARY}")

        add_library(mtmd UNKNOWN IMPORTED)

        set_target_properties(mtmd PROPERTIES
            IMPORTED_LOCATION "${MTMD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "llama"
        )

        # The installed llama.cpp is a Release (/MD) static library; a Debug
        # (/MDd) bench may fail to link against it with LNK2038 on MSVC.
        if(CMAKE_CONFIGURATION_TYPES)
            message(STATUS "[vAi-bench] llama.cpp was built Release: build bench-gemma3-llamacpp with --config Release")
        endif()

        add_executable(bench-gemma3-llamacpp
            llama.cpp/bench.cpp
            ${CMAKE_SOURCE_DIR}/vai/utils/image-processor.cpp
        )

        target_link_libraries(bench-gemma3-llamacpp PRIVATE
            llama
            mtmd
        )

        target_include_directories(bench-gemma3-llamacpp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_SOURCE_DIR}
        )
    else()
        message(STATUS "[vAi-bench] llama.cpp found, but mtmd was not found — "
                       "build/install llama.cpp with -DLLAMA_BUILD_MTMD=ON.")
    endif()
else()
    message(STATUS "[vAi-bench] llama.cpp not found — building bench-gemma3-vai only. "
                   "For the llama.cpp counterpart run benchmarks/scripts/setup-llamacpp.sh, "
                   "or point llama_DIR at an existing install.")
endif()
