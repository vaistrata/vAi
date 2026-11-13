cmake_minimum_required(VERSION 3.16)

project(demo-llama3)

# demo-llama3 — one-prompt completion with timings
add_executable(${PROJECT_NAME}
    demo.cpp
)
