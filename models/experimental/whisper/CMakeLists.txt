cmake_minimum_required(VERSION 3.20)

project(demo-whisper)

# demo-whisper — transcribes a WAV file in 30 s windows and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
)
