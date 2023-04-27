#include "afm.h"
// #include <torch/all.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("afm", &afm, "attraction field map generation");
}