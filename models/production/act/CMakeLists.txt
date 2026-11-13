cmake_minimum_required(VERSION 3.20)

project(demo-act)

# demo-act — turns one camera view and the robot's state into an action chunk
add_executable(${PROJECT_NAME}
    demo.cpp
)
