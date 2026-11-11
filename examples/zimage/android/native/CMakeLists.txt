# libzimage-jni.so — the Z-Image pipeline behind the app's Java side, plus the
# shaders it loads; build-native.sh configures this for arm64-v8a.
cmake_minimum_required(VERSION 3.24)
project(zimage-android CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VAI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. CACHE PATH "vAi checkout")
set(APP_ID com.vaistrata.zimage)

# vAi bakes its shader and data directories in as string literals. The app's
# are re-defined through CMAKE_CXX_FLAGS, which land after the targets' own
# definitions, so these -U/-D win:
#   shaders  the app's internal files, unpacked there from the APK's assets
#   data     the app's external files, where adb can push the weights
set(APP_FILES /data/data/${APP_ID}/files)
set(APP_DATA /storage/emulated/0/Android/data/${APP_ID}/files/data)
string(APPEND CMAKE_CXX_FLAGS
    " -USHADER_OUTPUT_DIR -DSHADER_OUTPUT_DIR=\\\"${APP_FILES}/shaders/vai\\\""
    " -UZIMAGE_SHADER_OUTPUT_DIR -DZIMAGE_SHADER_OUTPUT_DIR=\\\"${APP_FILES}/shaders/zimage\\\""
    " -UVAI_DATA_DIR -DVAI_DATA_DIR=\\\"${APP_DATA}\\\"")

add_subdirectory(${VAI_ROOT} vai)

add_library(zimage-jni SHARED zimage-jni.cpp)
target_link_libraries(zimage-jni PRIVATE vAi android log)
add_dependencies(zimage-jni zimage-shaders)
