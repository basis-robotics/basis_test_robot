#pragma once

#include <span>
#include <string.h>
#include <memory>
#include <basis/core/time.h>
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
    std::byte* buffer;
};




std::unique_ptr<CudaManagedImage> YUYV_to_RGB(const CudaManagedImage& image_in);


}