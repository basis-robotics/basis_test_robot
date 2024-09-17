#include <string.h>
#include <filesystem>
#include <vector>
#include <image_conversion.h>

struct Detection {
  std::string det_class;
  std::array<std::array<int, 2>, 4> bbox;
};

enum class InferenceRunMode {
  INFERENCE,
  DUMP_CACHE,
};

class Inference {
public:
  Inference(const std::filesystem::path& model_dir, const std::string_view platform, InferenceRunMode mode, int inference_height, int inference_width);
  virtual ~Inference();

  virtual std::vector<Detection> Infer(const image_conversion::CudaManagedImage& image) = 0;

  void DetectInputsOutputs();

  size_t BufferSize() {
    return inference_width * inference_height * 3 * sizeof(float);
  }

  std::array<std::shared_ptr<foxglove::RawImage>, 3> DumpInferenceBuffer();

  Ort::Value CreateInputInferenceTensor();

protected:

  float* inference_buffer = nullptr;
  int inference_width = 1280;
  int inference_height = 800;
  std::unique_ptr<Ort::Session> session;
  basis::core::MonotonicTime last_input_time;
};


class InferenceYoloV9 : public Inference {
public:
  InferenceYoloV9(const std::filesystem::path& model_dir, const std::string_view platform, InferenceRunMode mode);
  ~InferenceYoloV9();

  virtual std::vector<Detection> Infer(const image_conversion::CudaManagedImage& image) override;

  unsigned char* resize_buffer;
  int resize_step;
};

class InferenceDetrResnet : public Inference {
public:
  InferenceDetrResnet(const std::filesystem::path& model_dir, const std::string_view platform, InferenceRunMode mode);

  virtual std::vector<Detection> Infer(const image_conversion::CudaManagedImage& image) override;


  std::vector<Detection> Infer();

};