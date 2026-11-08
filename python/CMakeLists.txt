message(STATUS "[vAi] Configuring Python bindings...")

################################################################################
# Compiler — GCC 14 or newer
################################################################################
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR
        "[vAi] GCC ${CMAKE_CXX_COMPILER_VERSION} is too old: the core needs GCC 14 or newer (C++23 deducing `this`).\n"
        "  Remove ${CMAKE_BINARY_DIR} and re-run with the newer compiler, e.g.:\n"
        "  CC=gcc-14 CXX=g++-14 pip install -e ."
    )
endif()

################################################################################
# Toolchain — Python + nanobind
################################################################################
if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9
    COMPONENTS Interpreter ${DEV_MODULE}
    REQUIRED
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE NB_RESULT
    ERROR_QUIET
)

if(NOT NB_RESULT EQUAL 0)
    message(FATAL_ERROR
        "[vAi] nanobind not found in ${Python_EXECUTABLE}.\n"
        "  Install with: \"${Python_EXECUTABLE}\" -m pip install nanobind\n"
        "  (or re-run cmake with -DPython_EXECUTABLE=<path to python with nanobind installed>)"
    )
endif()

find_package(nanobind CONFIG REQUIRED)

message(STATUS "[vAi] Python  : ${Python_EXECUTABLE} (${Python_VERSION})")
message(STATUS "[vAi] nanobind: ${nanobind_ROOT}")

################################################################################
# Package output — <build>/python/pyvai
################################################################################
# A direct CMake build assembles the package here (extension, stub, .py files):
# import it with `PYTHONPATH=<build>/python`. pip goes through the install rules.
set(PYVAI_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pyvai)

################################################################################
# Extension module (_pyvai)
################################################################################
file(GLOB_RECURSE PYVAI_SRCS
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp
)

nanobind_add_module(_pyvai
    NB_STATIC
    LTO
    ${PYVAI_SRCS}
)

target_link_libraries(_pyvai
PRIVATE
    vAi
)

set_target_properties(_pyvai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYVAI_PACKAGE_DIR}"
)

# Model-local shaders are compiled once under models/ (see models/CMakeLists.txt).
# Every model's: the bindings reach model-local nodes, not only the bound models.
add_dependencies(_pyvai ${VAI_MODEL_SHADER_TARGETS})

################################################################################
# Package assembly — pure-Python files
################################################################################
# Per file, so the copy is incremental and never picks up __pycache__.
file(GLOB_RECURSE PYVAI_PY_FILES
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/pyvai
    ${CMAKE_CURRENT_SOURCE_DIR}/pyvai/*.py
)
list(APPEND PYVAI_PY_FILES py.typed)

set(PYVAI_PKG_OUTPUTS)
foreach(REL_PATH ${PYVAI_PY_FILES})
    add_custom_command(
        OUTPUT ${PYVAI_PACKAGE_DIR}/${REL_PATH}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/pyvai/${REL_PATH}
                ${PYVAI_PACKAGE_DIR}/${REL_PATH}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pyvai/${REL_PATH}
    )
    list(APPEND PYVAI_PKG_OUTPUTS ${PYVAI_PACKAGE_DIR}/${REL_PATH})
endforeach()

add_custom_target(pyvai-package ALL DEPENDS ${PYVAI_PKG_OUTPUTS})

################################################################################
# Type stub (_pyvai.pyi)
################################################################################
nanobind_add_stub(_pyvai_stub
    MODULE _pyvai
    OUTPUT ${PYVAI_PACKAGE_DIR}/_pyvai.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_pyvai>
    DEPENDS _pyvai
)

################################################################################
# Install rules — pip install -e . (scikit-build-core)
################################################################################
# The .py files come from the source tree (pyproject wheel.packages), so only
# the extension and the stub install here.
if(SKBUILD)
    install(TARGETS _pyvai LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
            COMPONENT python)
    install(FILES ${PYVAI_PACKAGE_DIR}/_pyvai.pyi
            DESTINATION ${SKBUILD_PROJECT_NAME}
            COMPONENT python)
endif()

message(STATUS "[vAi] Python bindings configuration completed.")
