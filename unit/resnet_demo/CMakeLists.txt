generate_unit(resnet_demo)

find_package(CUDAToolkit)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(prepare_resnet_image src/prepare_image.cu)
target_include_directories(prepare_resnet_image PUBLIC include)

target_link_libraries(unit_resnet_demo image_conversion foxglove_schemas_protobuf 
    onnxruntime
    # onnxruntime_providers_cuda
    onnxruntime_providers_shared
    # onnxruntime_providers_tensorrt
    nvinfer
    nvonnxparser


    prepare_resnet_image
    )
