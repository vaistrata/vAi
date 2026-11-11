# Every example links vAi and nothing else, so the dependency is declared once for
# the whole sub-directory.

link_libraries(vAi)

# Every folder with a CMakeLists.txt is an example, so a new one needs no entry here.
file(GLOB EXAMPLE_LISTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt)

foreach(exampleList ${EXAMPLE_LISTS})
    get_filename_component(exampleDir ${exampleList} DIRECTORY)
    add_subdirectory(${exampleDir})
endforeach()
