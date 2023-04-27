#pragma once
// #include "cuda/afm.h"
#include <torch/extension.h>

extern "C" std::tuple<at::Tensor, at::Tensor> afm_cuda(
    const at::Tensor& lines,
    const at::Tensor& shape_info,
    const int height,
    const int width);

extern "C" std::tuple<at::Tensor,at::Tensor> afm(
    const at::Tensor& lines,
    const at::Tensor& shape_info,
    const int height,
    const int width)
{
    return afm_cuda(lines,shape_info,height,width);
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("afm", &afm, "attraction field map generation");
// }