generate_unit(rplidar_driver)

FetchContent_Declare(
    rplidar_sdk
    GIT_REPOSITORY https://github.com/Slamtec/rplidar_sdk.git
    GIT_TAG 99478e5fb90de3b4a6db0080acacd373f8b36869
    PATCH_COMMAND sed -i -e "s/<nullptr_t>/<std::nullptr_t>/g" sdk/src/sl_async_transceiver.cpp
)
FetchContent_MakeAvailable(rplidar_sdk)

set(RPLIDAR_DIR ${rplidar_sdk_SOURCE_DIR}/sdk)

add_library(rplidar_sdk 
    ${RPLIDAR_DIR}/src/sl_lidar_driver.cpp 
    ${RPLIDAR_DIR}/src/hal/thread.cpp
    ${RPLIDAR_DIR}/src/sl_crc.cpp
    ${RPLIDAR_DIR}/src/sl_serial_channel.cpp
    ${RPLIDAR_DIR}/src/sl_lidarprotocol_codec.cpp
    ${RPLIDAR_DIR}/src/sl_async_transceiver.cpp
    ${RPLIDAR_DIR}/src/sl_tcp_channel.cpp
    ${RPLIDAR_DIR}/src/sl_udp_channel.cpp 

    ${RPLIDAR_DIR}/src/dataunpacker/dataunpacker.cpp 
    ${RPLIDAR_DIR}/src/dataunpacker/unpacker/handler_capsules.cpp 
    ${RPLIDAR_DIR}/src/dataunpacker/unpacker/handler_hqnode.cpp 
    ${RPLIDAR_DIR}/src/dataunpacker/unpacker/handler_normalnode.cpp 
    # Linux only
    ${RPLIDAR_DIR}/src/arch/linux/net_serial.cpp
    ${RPLIDAR_DIR}/src/arch/linux/net_socket.cpp
    ${RPLIDAR_DIR}/src/arch/linux/timer.cpp 
    )

target_include_directories(rplidar_sdk PUBLIC ${RPLIDAR_DIR}/src ${RPLIDAR_DIR}/include)

target_link_libraries(unit_rplidar_driver foxglove_schemas_protobuf rplidar_sdk)
