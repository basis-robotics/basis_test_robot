#pragma once

#include <span>
#include <string.h>
#include <memory>
#include <basis/core/time.h>
#include <variant>

#if BASIS_HAS_CUDA

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                                \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        raise(SIGTRAP); \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)
#endif

/*
 Note: this is how it starts - someone writes really basic CUDA code to "just get images working" 
 and then two years later it's still in use, your robot has blocking cuda calls all over the place 
 and your performance metrics are sad because nothing is pipelined.
*/
namespace foxglove {
    class RawImage;
}
namespace image_conversion {

#if BASIS_HAS_CUDA
void CheckCudaError();
#endif

enum class PixelFormat {
    Invalid,
    YUV422,
    RGB,
};

struct Image {
    Image(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::string_view frame_id);
    virtual ~Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
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

    std::shared_ptr<foxglove::RawImage> ToMessage() const;
    // static std::shared_ptr<Image> FromMessage(const foxglove::RawImage* message);
    static std::shared_ptr<const Image> FromVariant(
        const std::variant<std::monostate,
            std::shared_ptr<const foxglove::RawImage>,
            std::shared_ptr<const Image>>& variant);

    virtual void CopyToCPUBuffer(std::byte* out) const = 0;

    virtual std::byte* GetGPUBuffer() const { return nullptr; }

    const PixelFormat pixel_format;
    const int width;
    const int height;
    basis::core::MonotonicTime time;
    std::string frame_id;
};

struct CpuImage : public Image {
    CpuImage(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::string_view frame_id, const std::byte* data = nullptr);

    virtual void CopyToCPUBuffer(std::byte* out) const override;

    static std::shared_ptr<image_conversion::CpuImage> FromMessage(const foxglove::RawImage* message);
    static std::shared_ptr<const image_conversion::CpuImage> FromVariant(
        const std::variant<std::monostate,
            std::shared_ptr<const foxglove::RawImage>,
            std::shared_ptr<const Image>>& variant);

    std::unique_ptr<std::byte[]> buffer;
};

#if BASIS_HAS_CUDA
struct CudaManagedImage : public Image {
    CudaManagedImage(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::string_view frame_id, const std::byte* data = nullptr);
    ~CudaManagedImage();
    CudaManagedImage(const CudaManagedImage&) = delete;
    CudaManagedImage& operator=(const CudaManagedImage&) = delete;

    virtual void CopyToCPUBuffer(std::byte* out) const override;

    virtual std::byte* GetGPUBuffer() const override {
        return buffer;
    }

    static std::shared_ptr<image_conversion::CudaManagedImage> FromMessage(const foxglove::RawImage* message);
    static std::shared_ptr<const image_conversion::CudaManagedImage> FromVariant(
        const std::variant<std::monostate,
            std::shared_ptr<const foxglove::RawImage>,
            std::shared_ptr<const image_conversion::Image>>& variant);

    std::byte* buffer = nullptr;
};

std::unique_ptr<CudaManagedImage> YUYV_to_RGB(const Image& image_in);
#endif


}