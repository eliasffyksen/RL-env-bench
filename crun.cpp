#include <iostream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>

torch::jit::script::Module model;

bool load(std::string path) {
  try {
    model = torch::jit::load(path);
    return true;
  }
  catch (const c10::Error& e) {
    return false;
  }
}

torch::Tensor run(int batch_size) {
  return model.forward({batch_size}).toTensor();
}

torch::Tensor run_model(
  torch::jit::script::Module m,
  int batch_size
) {
  return m.forward({batch_size}).toTensor();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_model", &run_model, "Run jit script from C++");
  m.def("load", &load, "Load model from file");
  m.def("run", &run, "Run loaded file");
}
