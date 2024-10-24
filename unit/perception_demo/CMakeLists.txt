generate_unit(perception_demo)

find_package(CUDAToolkit)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(lib)


target_link_libraries(unit_perception_demo image_conversion foxglove_schemas_protobuf 
    

    inference
    )
