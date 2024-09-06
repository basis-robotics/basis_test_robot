cmake_minimum_required(VERSION 3.28)

project(BasisTestRobot LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add basis, use current binary directory as root, to avoid pollution
add_subdirectory(/basis/ basis)

add_subdirectory(lib/image_conversion)

add_subdirectory(unit/v4l2_camera_driver)
add_subdirectory(unit/yuyv_to_rgb)