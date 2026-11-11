cmake_minimum_required(VERSION 3.16)

project(demo-whisper)

# demo-whisper — transcribes a WAV file and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)

add_dependencies(${PROJECT_NAME} whisper-shaders)
