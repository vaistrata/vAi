cmake_minimum_required(VERSION 3.16)

project(demo-zimage)

# demo-zimage — Turbo flow-matching sampling on the DiT, timings per step
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} zimage-shaders)
