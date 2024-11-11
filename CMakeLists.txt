cmake_minimum_required(VERSION 3.26)

project(BasisTestRobot LANGUAGES CXX)
include(CheckLanguage)
check_language(CUDA)

function(multi_option variable docstring value)
  set(possible_values "${value}")
  if(ARGC GREATER "3")
    list(LENGTH ARGN length)
    list(PREPEND ARGN "${value}")
    list(GET ARGN "${length}" value)
    set(possible_values "${ARGN}")
  endif()

  option("${variable}" "${docstring}" "${value}")
  set_property(CACHE "${variable}" PROPERTY STRINGS "${possible_values}")
  if(NOT "${${variable}}" IN_LIST possible_values)
    message(FATAL_ERROR "The variable \"${variable}\" has an unknown value: ${${variable}}\nPossible values are: ${possible_values}")
  endif()
endfunction()


multi_option(BASIS_PLATFORM "Platform to compile on" "UNKNOWN" "ORIN" "PI")
message(STATUS "BASIS_PLATFORM='${BASIS_PLATFORM}'")

if(CMAKE_CUDA_COMPILER)
    add_compile_definitions(BASIS_HAS_CUDA=1)
else()
    add_compile_definitions(BASIS_HAS_CUDA=0)
    message(WARNING "No CUDA support detected, disabling CUDA features")
endif()

# TODO?
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

if(BASIS_PLATFORM STREQUAL "PI")
    add_subdirectory(unit/rpi_libcamera_driver)
    add_subdirectory(unit/rpi_freenove_servo_driver)
endif()


if(CMAKE_CUDA_COMPILER)
    add_subdirectory(unit/perception_demo)
    add_subdirectory(unit/yuyv_to_rgb)
endif()