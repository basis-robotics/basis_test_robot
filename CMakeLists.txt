cmake_minimum_required(VERSION 3.26)

project(BasisTestRobot LANGUAGES CXX)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    add_compile_definitions(BASIS_HAS_CUDA=1)
else()
    add_compile_definitions(BASIS_HAS_CUDA=0)
    message(WARNING "No CUDA support detected, disabling CUDA features")
endif()

# TODO
# https://github.com/descampsa/yuv2rgb

#set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add basis, use current binary directory as root, to avoid pollution
add_subdirectory(../basis/ basis)

add_subdirectory(lib/image_conversion)

add_subdirectory(unit/v4l2_camera_driver)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(unit/perception_demo)
    add_subdirectory(unit/yuyv_to_rgb)
endif()