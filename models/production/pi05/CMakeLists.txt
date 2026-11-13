cmake_minimum_required(VERSION 3.16)

project(demo-pi05)

# demo-pi05 — turns three cameras, a task and the robot's state into one action chunk
add_executable(${PROJECT_NAME}
    demo.cpp
)
