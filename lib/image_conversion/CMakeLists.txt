add_library(image_conversion src/image_conversion.cpp)
target_include_directories(image_conversion PUBLIC include)

find_package(CUDAToolkit)
# nppig probably isn't needed
target_link_libraries(image_conversion CUDA::cudart nppicc npp_plus_c npp_plus_icc foxglove_schemas_protobuf basis::core::time)