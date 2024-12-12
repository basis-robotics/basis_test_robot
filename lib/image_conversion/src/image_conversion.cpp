#include "basis/core/time.h"
#include <foxglove/RawImage.pb.h>
#include <image_conversion.h>

#include <basis/core/transport/convertable_inproc.h>

#if BASIS_HAS_CUDA
#include <nppi_color_conversion.h>
#endif

#include <google/protobuf/util/time_util.h>

namespace image_conversion {

template<typename T_IMAGE_TYPE>
std::shared_ptr<T_IMAGE_TYPE> ImageFromMessage(const foxglove::RawImage *message) {
  PixelFormat pixel_format;
  if (message->encoding() == "yuyv") {
    pixel_format = PixelFormat::YUV422;
  } else if (message->encoding() == "rgb8") {
    pixel_format = PixelFormat::RGB;
  } else {
    return nullptr;
  }
  auto out = std::make_shared<T_IMAGE_TYPE>(
      pixel_format, message->width(), message->height(),
      basis::core::MonotonicTime::FromSecondsNanoseconds(message->timestamp().seconds(), message->timestamp().nanos()),
      message->frame_id(), (const std::byte *)message->data().data());
  return out;
}

template<typename T_IMAGE_TYPE>
std::shared_ptr<const Image>
ImageFromVariant(const std::variant<std::monostate, std::shared_ptr<const foxglove::RawImage>,
                                                 std::shared_ptr<const Image>> &variant) {
  switch (variant.index()) {
  case basis::MessageVariant::NO_MESSAGE:
    return {};
  case basis::MessageVariant::TYPE_MESSAGE: {
    auto foxglove = std::get<basis::MessageVariant::TYPE_MESSAGE>(variant);
    return ImageFromMessage<T_IMAGE_TYPE>(foxglove.get());
  };
  case basis::MessageVariant::INPROC_TYPE_MESSAGE:
    return std::get<basis::MessageVariant::INPROC_TYPE_MESSAGE>(variant);
  }

  return {};
}


Image::Image(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::string_view frame_id)
    : pixel_format(pixel_format), width(width), height(height), time(time), frame_id(frame_id) {
  
}

std::shared_ptr<foxglove::RawImage> Image::ToMessage() const {
  auto image_msg = std::make_shared<foxglove::RawImage>();

  *image_msg->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(time.nsecs);
  image_msg->set_frame_id(frame_id);

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

  CopyToCPUBuffer((std::byte*)data->data());

  return image_msg;
}

CpuImage::CpuImage(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::string_view frame_id,
                                   const std::byte *cpu_data)
    : Image(pixel_format, width, height, time, frame_id) {
  const size_t size = ImageSize();
  // cudaMalloc3D??
  //CUDA_SAFE_CALL_NO_SYNC(cudaMalloc(&buffer, size));
  buffer = std::make_unique<std::byte[]>(size);

  if (cpu_data) {
    memcpy(buffer.get(), cpu_data, size);
  }
}

std::shared_ptr<CpuImage> CpuImage::FromMessage(const foxglove::RawImage* message) {
  return ImageFromMessage<CpuImage>(message);
}

std::shared_ptr<const CpuImage>
CpuImage::FromVariant(const std::variant<std::monostate, std::shared_ptr<const foxglove::RawImage>,
                                                 std::shared_ptr<const Image>> &variant) {
  auto image = ImageFromVariant<CpuImage>(variant);
  if(image->GetGPUBuffer()) {
    // TODO: conversions
    std::cout << "non CPU image passed to CudaManagedImage::FromVariant" << std::endl;
    return nullptr;
  }
  else {
    return std::dynamic_pointer_cast<const CpuImage>(image);
  }
}

void CpuImage::CopyToCPUBuffer(std::byte* out) const {
  memcpy(out, buffer.get(), ImageSize());
}

#if BASIS_HAS_CUDA

void CheckCudaError() {
  auto e = cudaGetLastError();
  if (e != 0) {
    std::cout << cudaGetErrorString(e) << std::endl;
    throw e;
  }
}

CudaManagedImage::CudaManagedImage(PixelFormat pixel_format, int width, int height, basis::core::MonotonicTime time, std::string_view frame_id,
                                   const std::byte *cpu_data)
    : Image(pixel_format, width, height, time, frame_id) {
  const size_t size = ImageSize();
  // cudaMalloc3D??
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc(&buffer, size));

  if (cpu_data) {
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(buffer, cpu_data, size, cudaMemcpyHostToDevice));
  }
}

CudaManagedImage::~CudaManagedImage() { CUDA_SAFE_CALL_NO_SYNC(cudaFree(buffer)); }

void CudaManagedImage::CopyToCPUBuffer(std::byte* out) const {
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(out, buffer, ImageSize(), cudaMemcpyDeviceToHost));
}

std::unique_ptr<CudaManagedImage> YUYV_to_RGB(const Image &image_in) {
  std::byte* buffer = image_in.GetGPUBuffer();
  if(buffer == nullptr) {
    // TODO: conversions
    std::cout << "non GPU image passed into YUYV_to_RGB" << std::endl;
    return nullptr;
  }

  auto rgb = std::make_unique<CudaManagedImage>(PixelFormat::RGB, image_in.width, image_in.height, image_in.time, image_in.frame_id);

  auto status = nppiYUV422ToRGB_8u_C2C3R((const Npp8u *)buffer, image_in.StepSize(), (Npp8u *)rgb->buffer,
                                         rgb->StepSize(), {image_in.width, image_in.height});
  // TODO: logger
  if (status != 0) {
    std::cout << "bad status " << status << std::endl;
    return nullptr;
  }
  return std::move(rgb);
}

std::shared_ptr<CudaManagedImage> CudaManagedImage::FromMessage(const foxglove::RawImage *message) {
  return ImageFromMessage<CudaManagedImage>(message);
}

std::shared_ptr<const CudaManagedImage>
CudaManagedImage::FromVariant(const std::variant<std::monostate, std::shared_ptr<const foxglove::RawImage>,
                                                 std::shared_ptr<const Image>> &variant) {
  auto image = ImageFromVariant<CudaManagedImage>(variant);
  if(!image) {
    return nullptr;
  }
  if(image->GetGPUBuffer()) {
    return std::dynamic_pointer_cast<const CudaManagedImage>(image);
  }
  else {
    // TODO: conversions
    std::cout << "non GPU image passed to CudaManagedImage::FromVariant" << std::endl;
    return nullptr;
  }
}

#endif
} // namespace image_conversion