#include <stdio.h>
__global__ void RGBToTensorKernel(const unsigned char *in, const int in_step, float *out, const int out_w, const int out_h) {
  // screw it, get someone in the know to explain this
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  
  const int source_pos = in_step * pixel_y + pixel_x * 3;

  const int plane_size = out_w * out_h;
  const int dest_pos_r = out_w * pixel_y + pixel_x;
  const int dest_pos_g = dest_pos_r + plane_size;
  const int dest_pos_b = dest_pos_r + plane_size * 2;

  unsigned char in_r = in[source_pos];
  unsigned char in_g = in[source_pos + 1];
  unsigned char in_b = in[source_pos + 2];

  out[dest_pos_r] = (float)in_r / 255.0f;
  out[dest_pos_g] = (float)in_g / 255.0f;
  out[dest_pos_b] = (float)in_b / 255.0f;
  // normalize
  // out[dest_pos_r] = (out[dest_pos_r] - 0.485) / 0.229;
  // out[dest_pos_g] = (out[dest_pos_g] - 0.456) / 0.224;
  // out[dest_pos_b] = (out[dest_pos_b] - 0.406) / 0.225;
}

__host__ void RGBToTensor(const unsigned char *in, const int in_w, const int in_h, const int in_step, float *out, const int out_w, const int out_h) {
  constexpr int BLOCK_SIZE = 8;
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid_dim((in_w - 1) / BLOCK_SIZE + 1, (in_h - 1) / BLOCK_SIZE + 1, 1);

  RGBToTensorKernel<<<grid_dim, block_dim>>>(in, in_step, out, out_w, out_h);
}