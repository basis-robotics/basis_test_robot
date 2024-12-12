generate_unit(joystick_driver)

find_package(PkgConfig REQUIRED)
pkg_search_module(LIBEVDEV REQUIRED libevdev)

target_link_libraries(unit_joystick_driver basis_robot_input_proto ${LIBEVDEV_LIBRARIES})
target_include_directories(unit_joystick_driver PUBLIC ${LIBEVDEV_INCLUDE_DIRS})