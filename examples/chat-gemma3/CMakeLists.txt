cmake_minimum_required(VERSION 3.16)

project(chat-gemma3)

# chat-gemma3 — multi-turn chat on the KV cache, with an image on 4B
add_executable(${PROJECT_NAME}
    main.cpp
)
