add_library(inference src/inference.cpp)
target_include_directories(inference PUBLIC include)
target_link_libraries(inference
    prepare_image
    image_conversion
    foxglove_schemas_protobuf
    spdlog

    onnxruntime
    
    onnxruntime_providers_shared
    
    nvinfer
    nvonnxparser

    CUDA::nppisu
    CUDA::nppig

    CUDA::nppial
    CUDA::nppidei)