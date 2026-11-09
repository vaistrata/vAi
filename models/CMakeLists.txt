# <model>-shaders — every models/<model>/shaders compiled once, named by its folder.
# <MODEL>_SHADER_OUTPUT_DIR tells the model's nodes where the .spv files land.
file(GLOB MODEL_SHADER_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*/shaders)

foreach(shaderDir ${MODEL_SHADER_DIRS})
    get_filename_component(modelDir ${shaderDir} DIRECTORY)
    get_filename_component(model ${modelDir} NAME)
    string(TOUPPER ${model} MODEL)

    file(GLOB shaders CONFIGURE_DEPENDS ${shaderDir}/*.comp)
    set(outDir ${CMAKE_CURRENT_BINARY_DIR}/${model}/shaders)

    add_custom_target(${model}-shaders)
    eva_compile_shaders(TARGET ${model}-shaders OUTPUT_DIR ${outDir} SHADERS ${shaders})

    list(APPEND definitions ${MODEL}_SHADER_OUTPUT_DIR="${outDir}")
    list(APPEND targets ${model}-shaders)
endforeach()

set(VAI_MODEL_SHADER_DEFINITIONS ${definitions} PARENT_SCOPE)
set(VAI_MODEL_SHADER_TARGETS ${targets} PARENT_SCOPE)
