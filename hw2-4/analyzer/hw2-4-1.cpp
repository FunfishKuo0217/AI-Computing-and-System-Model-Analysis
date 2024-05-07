#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

    std::cout << "ok\n";
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    size_t total_params = 0;
    for(const auto& p : module.parameters()){
      size_t layer_size = 1;
      // std::cout << p.sizes() << std::endl;
      for( auto iter = p.sizes().begin(); iter != p.sizes().end(); iter++){
        layer_size *= *iter;
      }	
      total_params += layer_size;    
    }
    std::cout << "total size: " << total_params*4 <<" bytes" <<std::endl;
}

