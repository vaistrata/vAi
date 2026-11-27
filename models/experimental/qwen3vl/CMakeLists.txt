cmake_minimum_required(VERSION 3.20)

project(demo-qwen3vl)

# demo-qwen3vl — answers one prompt about one image and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} experimental-qwen3vl-shaders)
