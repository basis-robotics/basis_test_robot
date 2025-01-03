cmake_minimum_required(VERSION 3.25.1)

set(CMAKE_POLICY_DEFAULT_CMP0174 NEW)

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

multi_option(BASIS_PLATFORM "Platform to compile on" "UNKNOWN" "ORIN" "PI" "OFF")

if((BASIS_PLATFORM STREQUAL "UNKNOWN" OR BASIS_PLATFORM STREQUAL "OFF") AND DEFINED ENV{BASIS_PLATFORM})
  set(BASIS_PLATFORM $ENV{BASIS_PLATFORM})
  message(STATUS "BASIS_PLATFORM not set, pulled BASIS_PLATFORM='${BASIS_PLATFORM}' from environment")
endif()

message(STATUS "BASIS_PLATFORM='${BASIS_PLATFORM}'")

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_compile_definitions(BASIS_HAS_CUDA=1)
else()
  add_compile_definitions(BASIS_HAS_CUDA=0)
  message(WARNING "No CUDA support detected, disabling CUDA features")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add basis, use current binary directory as root, to avoid pollution
add_subdirectory(../basis/ basis)

if(EXISTS "../geometry2")
  add_subdirectory(../geometry2/tf2_basis geometry2)
else()
  FetchContent_Declare(
      geometry2
      GIT_REPOSITORY https://github.com/basis-robotics/geometry2.git
      GIT_TAG basis
      SOURCE_SUBDIR tf2_basis
  )
  FetchContent_MakeAvailable(geometry2)
endif()

add_subdirectory(lib/image_conversion)

add_subdirectory(proto)

add_subdirectory(unit/joystick_driver)
add_subdirectory(unit/v4l2_camera_driver)
add_subdirectory(unit/rplidar_driver)
add_subdirectory(unit/static_transform_publisher)

if(BASIS_PLATFORM STREQUAL "PI")
  FetchContent_Declare(
    PiPCA9685 
    GIT_REPOSITORY https://github.com/basis-robotics/PiPCA9685.git
    GIT_TAG main
  )
  FetchContent_MakeAvailable(PiPCA9685)

  add_subdirectory(unit/rpi_libcamera_driver)
  add_subdirectory(unit/rpi_freenove_mecanum_driver)
  add_subdirectory(unit/rpi_freenove_servo_driver)
endif()

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(unit/perception_demo)
    add_subdirectory(unit/yuyv_to_rgb)
endif()