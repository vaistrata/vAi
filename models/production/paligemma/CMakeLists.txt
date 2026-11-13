cmake_minimum_required(VERSION 3.16)

project(demo-paligemma)

# demo-paligemma — answers one prompt about one image and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)
