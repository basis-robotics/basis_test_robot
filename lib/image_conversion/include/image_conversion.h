#pragma once

#include <span>
#include <string.h>
#include <memory>
#include <basis/core/time.h>

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                                \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

void CheckCudaError();

/*
 Note: this is how it starts - someone writes really basic CUDA code to "just get images working" 
 and then two years later it's still in use, your robot has blocking cuda calls all over the place 
 and your performance metrics are sad because nothing is pipelined.
*/
namespace foxglove {
    class RawImage;
}
namespace image_conversion {

enum class PixelFormat {
    Invalid,
    YUV422,
    RGB,
};


struct CudaManagedImage {
    CudaManagedImage(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::byte* data = nullptr);
    ~CudaManagedImage();
    CudaManagedImage(const CudaManagedImage&) = delete;
    CudaManagedImage& operator=(const CudaManagedImage&) = delete;
    size_t StepSize() const {
        switch (pixel_format) {
        case PixelFormat::YUV422:
            return 2 * width;
        case PixelFormat::RGB:
            return 3 * width;
        default:
            return 0;
        }
    }

    size_t ImageSize() const {
        return StepSize() * height;
    }

    std::shared_ptr<foxglove::RawImage> ToFoxglove() const;


    const PixelFormat pixel_format;
    const int width;
    const int height;
    basis::core::MonotonicTime time;
    std::byte* buffer = nullptr;
};




std::unique_ptr<CudaManagedImage> YUYV_to_RGB(const CudaManagedImage& image_in);


}