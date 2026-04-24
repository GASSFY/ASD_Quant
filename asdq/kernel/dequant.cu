#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void dequantize_4bit_kernel(
    const uint8_t* __restrict__ qweight,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ zeros,
    scalar_t* __restrict__ out,
    int out_features,
    int in_features,
    int group_size,
    int n_groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = out_features * in_features;
  if (idx >= total) return;

  int row = idx / in_features;
  int col = idx % in_features;
  int packed_col = col >> 1;
  uint8_t packed = qweight[row * ((in_features + 1) / 2) + packed_col];
  uint8_t qv = (col & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);

  int g = col / group_size;
  if (g >= n_groups) return;
  scalar_t scale = scales[row * n_groups + g];
  scalar_t zero = zeros[row * n_groups + g];
  out[idx] = (static_cast<scalar_t>(qv) - zero) * scale;
}

torch::Tensor dequantize_4bit_cuda(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size,
    int64_t in_features) {
  const auto out_features = static_cast<int>(qweight.size(0));
  const auto n_groups = static_cast<int>(scales.size(1));
  auto out = torch::zeros(
      {out_features, static_cast<int>(in_features)},
      scales.options());

  const int total = out_features * static_cast<int>(in_features);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      scales.scalar_type(),
      "dequantize_4bit_cuda",
      ([&] {
        dequantize_4bit_kernel<scalar_t><<<blocks, threads>>>(
            qweight.data_ptr<uint8_t>(),
            scales.data_ptr<scalar_t>(),
            zeros.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            out_features,
            static_cast<int>(in_features),
            static_cast<int>(group_size),
            n_groups);
      }));

  return out;
}
