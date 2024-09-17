add_library(image_conversion SHARED src/image_conversion.cpp)
target_include_directories(image_conversion PUBLIC include)

find_package(CUDAToolkit)

target_link_libraries(image_conversion  CUDA::cudart  CUDA::nppicc foxglove_schemas_protobuf basis::core::time)

install(TARGETS image_conversion DESTINATION lib/)
