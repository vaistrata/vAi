cmake_minimum_required(VERSION 3.20)

project(demo-diffusion-policy)

# demo-diffusion-policy — turns two pushT observation steps into the next actions
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} experimental-diffusion-policy-shaders)
