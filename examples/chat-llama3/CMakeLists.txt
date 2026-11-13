cmake_minimum_required(VERSION 3.16)

project(chat-llama3)

# chat-llama3 — multi-turn interactive chat on the KV cache
add_executable(${PROJECT_NAME}
    main.cpp
)
