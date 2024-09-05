cmake_minimum_required(VERSION 3.28)

project(BasisTestRobot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Add basis, use current binary directory as root, to avoid pollution
add_subdirectory(/basis/ basis)
add_subdirectory(unit/v4l2_camera_driver)