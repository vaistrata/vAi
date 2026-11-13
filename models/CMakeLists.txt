# <impl>-<model>-shaders — every models/<impl>/<model>/shaders compiled once, named by its folders.
# <IMPL>_<MODEL>_SHADER_OUTPUT_DIR tells the implementation's nodes where the .spv files land.
file(GLOB MODEL_SHADER_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*/*/shaders)

foreach(shaderDir ${MODEL_SHADER_DIRS})
    get_filename_component(modelDir ${shaderDir} DIRECTORY)
    get_filename_component(model ${modelDir} NAME)
    get_filename_component(implDir ${modelDir} DIRECTORY)
    get_filename_component(impl ${implDir} NAME)
    string(TOUPPER ${impl} IMPL)
    string(TOUPPER ${model} MODEL)

    file(GLOB shaders CONFIGURE_DEPENDS ${shaderDir}/*.comp)
    set(outDir ${CMAKE_CURRENT_BINARY_DIR}/${impl}/${model}/shaders)

    add_custom_target(${impl}-${model}-shaders)
    eva_compile_shaders(TARGET ${impl}-${model}-shaders OUTPUT_DIR ${outDir} SHADERS ${shaders})

    list(APPEND definitions ${IMPL}_${MODEL}_SHADER_OUTPUT_DIR="${outDir}")
    list(APPEND targets ${impl}-${model}-shaders)
endforeach()

set(VAI_MODEL_SHADER_DEFINITIONS ${definitions} PARENT_SCOPE)
set(VAI_MODEL_SHADER_TARGETS ${targets} PARENT_SCOPE)

# Each models/<impl>/<model>/CMakeLists.txt builds that implementation's demo. The shader
# macros are set here first: definitions reach only the directories added after them.
if(VAI_BUILD_MODEL_DEMO)
    add_compile_definitions(${definitions})
    link_libraries(vAi)

    file(GLOB DEMO_LISTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/*/CMakeLists.txt)
    foreach(demoList ${DEMO_LISTS})
        get_filename_component(demoDir ${demoList} DIRECTORY)
        add_subdirectory(${demoDir})
    endforeach()
endif()
