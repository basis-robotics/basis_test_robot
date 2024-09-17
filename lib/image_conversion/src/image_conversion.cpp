#include <foxglove/RawImage.pb.h>
#include <image_conversion.h>
#include <nppi_color_conversion.h>

void CheckCudaError() {
  auto e = cudaGetLastError();
  if (e != 0) {
    std::cout << cudaGetErrorString(e) << std::endl;
    throw e;
  }
}

namespace image_conversion {
CudaManagedImage::CudaManagedImage(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time,
                                   std::byte *data)
    : pixel_format(pixel_format), width(width), height(height), time(time) {
  const size_t size = ImageSize();
  // cudaMalloc3D??
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc(&buffer, size));

  if (data) {
    cudaMemcpy(buffer, data, size, cudaMemcpyHostToDevice);
  }
}

CudaManagedImage::~CudaManagedImage() { CUDA_SAFE_CALL_NO_SYNC(cudaFree(buffer)); }
std::unique_ptr<CudaManagedImage> YUYV_to_RGB(const CudaManagedImage &image_in) {
  auto rgb = std::make_unique<CudaManagedImage>(PixelFormat::RGB, image_in.width, image_in.height, image_in.time);

  auto status = nppiYUV422ToRGB_8u_C2C3R((const Npp8u *)image_in.buffer, image_in.StepSize(), (Npp8u *)rgb->buffer,
                                         rgb->StepSize(), {image_in.width, image_in.height});
  // TODO: logger
  if (status != 0) {
    std::cout << "bad status " << status << std::endl;
    return nullptr;
  }
  return std::move(rgb);
}

std::shared_ptr<foxglove::RawImage> CudaManagedImage::ToFoxglove() const {

  auto image_msg = std::make_shared<foxglove::RawImage>();

  const timespec ts = time.ToTimespec();

  image_msg->mutable_timestamp()->set_seconds(ts.tv_sec);
  image_msg->mutable_timestamp()->set_nanos(ts.tv_nsec);
  image_msg->set_frame_id("webcam");
  switch (pixel_format) {
  case PixelFormat::YUV422:
    image_msg->set_encoding("yuyv");
    break;
  case PixelFormat::RGB:
    image_msg->set_encoding("rgb8");
    break;
  default:
    // BASIS_LOG_ERROR("No conversion possible");
    return nullptr;
  }

  image_msg->set_width(width);
  image_msg->set_height(height);
  image_msg->set_step(StepSize());
  const auto &data = image_msg->mutable_data();
  data->resize(ImageSize());

  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(data->data(), buffer, ImageSize(), cudaMemcpyDeviceToHost));

  return image_msg;
}

} // namespace image_conversion