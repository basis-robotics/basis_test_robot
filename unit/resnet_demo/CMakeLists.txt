generate_unit(resnet_demo)

find_package(CUDAToolkit)

target_link_libraries(unit_resnet_demo image_conversion foxglove_schemas_protobuf 
    onnxruntime
    # onnxruntime_providers_cuda
    onnxruntime_providers_shared
    # onnxruntime_providers_tensorrt
    nvinfer
    nvonnxparser

    )
