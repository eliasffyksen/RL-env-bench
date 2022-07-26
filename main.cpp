#include <torch/script.h>
#include <iostream>
#include <chrono>

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <rounds>\n";
    return -1;
  }

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  int rounds = atoi(argv[2]);
  for (int i = 0; i < 10; i++) {
    model.forward({10});
  }

  std::cout << "[" << std::endl;
  int batch_size = 2;
  for (int i = 0; i < rounds; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    model.forward({batch_size});
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "\t" << duration.count() << "," << std::endl;
    batch_size *= 2;
  }
  std::cout << "]" << std::endl;
}


