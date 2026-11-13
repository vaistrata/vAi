cmake_minimum_required(VERSION 3.20)

project(demo-yolox)

# demo-yolox — detects objects in one image and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} experimental-yolox-shaders)
