cmake_minimum_required(VERSION 3.16)

project(demo-pi0)

# demo-pi0 — turns three cameras and a task string into one action chunk
add_executable(${PROJECT_NAME}
    demo.cpp
)
