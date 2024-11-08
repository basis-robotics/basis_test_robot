# Only build the rpi camera driver if we have the required dependencies
find_library(CAMERA_LIBRARY NAMES libcamera.so HINTS /usr/lib/arm-linux-aarch64-linux-gnu)

if(CAMERA_LIBRARY)
    generate_unit(rpi_libcamera_driver)
    
    target_link_libraries(unit_rpi_libcamera_driver image_conversion foxglove_schemas_protobuf ${CAMERA_LIBRARY})
    target_include_directories(unit_rpi_libcamera_driver PUBLIC /usr/include/libcamera)
endif()