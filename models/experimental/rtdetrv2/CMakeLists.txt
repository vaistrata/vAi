cmake_minimum_required(VERSION 3.16)

project(demo-rtdetrv2)

# demo-rtdetrv2 — detects objects in one image and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} experimental-rtdetrv2-shaders)
