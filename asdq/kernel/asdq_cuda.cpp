#include <torch/extension.h>
#include <vector>

torch::Tensor dequantize_4bit_cuda(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size,
    int64_t in_features);

torch::Tensor dequantize_4bit(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size,
    int64_t in_features) {
  TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be CUDA tensor");
  TORCH_CHECK(qweight.dtype() == torch::kUInt8, "qweight must be uint8");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D [out_features, n_groups]");
  TORCH_CHECK(zeros.sizes() == scales.sizes(), "zeros shape must match scales");
  TORCH_CHECK(group_size > 0, "group_size must be > 0");
  TORCH_CHECK(in_features > 0, "in_features must be > 0");
  TORCH_CHECK(in_features % group_size == 0, "in_features must be divisible by group_size");

  return dequantize_4bit_cuda(qweight, scales, zeros, group_size, in_features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequantize_4bit", &dequantize_4bit, "Dequantize int4 packed weights (CUDA)");
}
