add_library(image_conversion SHARED src/image_conversion.cpp)
target_include_directories(image_conversion PUBLIC include)

# Unfortuantely we depend on transport due to the helpers for converting from a variant
# This enum/helper should be hoisted out
target_link_libraries(image_conversion foxglove_schemas_protobuf basis::core::time basis::core::transport)

if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
    target_link_libraries(image_conversion  CUDA::cudart  CUDA::nppicc)
endif()

install(TARGETS image_conversion DESTINATION lib/)
