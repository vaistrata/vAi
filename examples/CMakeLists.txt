# Every demo links vAi and nothing else, so the dependency is declared once for
# the whole sub-directory.

link_libraries(vAi)

add_subdirectory(mnist)
add_subdirectory(gpt2)
add_subdirectory(llama3)
add_subdirectory(gemma3)
add_subdirectory(paligemma)
add_subdirectory(pi0)
add_subdirectory(pi05)
add_subdirectory(yolox)
add_subdirectory(whisper)
