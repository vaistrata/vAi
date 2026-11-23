cmake_minimum_required(VERSION 3.20)

project(demo-experimental-pi05)

# demo-experimental-pi05 — pi0.5 on two nets, so a prefill - decode loop re-records nothing
add_executable(${PROJECT_NAME}
    demo.cpp
)
