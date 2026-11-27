cmake_minimum_required(VERSION 3.20)

project(demo-yolox)

# demo-yolox — detects objects in a batch of images and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} experimental-yolox-shaders)
