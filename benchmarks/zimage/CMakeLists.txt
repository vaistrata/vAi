cmake_minimum_required(VERSION 3.16)

project(bench-zimage)

# The model owns its definition (model.h) and its checkpoint loader
# (tongyi-weights.h, which resolves data/zimage/...), so the bench and
# examples/zimage can never end up on different weights. The PyTorch
# counterpart (torch/bench.py) writes the reference this bench reads.

################################################################################
# vai
################################################################################
add_executable(bench-zimage-vai
    vai/bench.cpp
)

add_dependencies(bench-zimage-vai zimage-shaders)

target_link_libraries(bench-zimage-vai PRIVATE
    vAi
)

# The VAE decoder against torch/vae.py's reference.
add_executable(bench-zimage-vae-vai
    vai/vae-bench.cpp
)

add_dependencies(bench-zimage-vae-vai zimage-shaders)

target_link_libraries(bench-zimage-vae-vai PRIVATE
    vAi
)

# The caption encoder (Qwen3) against torch/text_encoder.py's reference.
add_executable(bench-zimage-text-encoder-vai
    vai/text-encoder-bench.cpp
)

target_link_libraries(bench-zimage-text-encoder-vai PRIVATE
    vAi
)
