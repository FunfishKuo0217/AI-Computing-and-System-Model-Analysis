#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <ATen/core/symbol.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>

void ChildModulePrint(torch::jit::script::Module module, std::string prefix, std::vector<torch::jit::IValue> & CurrInputs, long long *activations){
  prefix += "|---";
  for (const auto& s : module.named_children()) {
      // s.value 也是一個小 module
      // std::cout << prefix <<  "Submodule Name: " << s.name << std::endl;
      ChildModulePrint(s.value, prefix, CurrInputs, activations);
      // 一定要確保沒有下層 submodules 才是正確的 child!!!
      if (prefix == " |---|---" || s.name == "avgpool"){
        torch::jit::script::Module CurrModule;
        at::Tensor output;
        if (s.name == "avgpool"){
          CurrModule = s.value;
          output = CurrModule.forward(CurrInputs).toTensor();
          // std::cout << " output = " << output.sizes() << std::endl;
          output = output.view({1, 9216});
        }
        else{
          CurrModule = s.value;
          output = CurrModule.forward(CurrInputs).toTensor();
        }
        CurrInputs.pop_back();
        CurrInputs.push_back(output);
        // std::cout << " output size = " << output.sizes() << std::endl;
        size_t activation_curr = 1;
        for( auto iter = output.sizes().begin(); iter != output.sizes().end(); iter++){
          activation_curr *= *iter;
        }	
        *activations += activation_curr;
      } 
  } 
}

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
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  // at::Tensor output = module.forward(inputs).toTensor();
  std::cout << "Input Size: " << std::endl;
  std::cout << inputs[0].toTensor().sizes() << std::endl;
  // ChildModule(module, " ", inputs);
  long long activations = 0;
  ChildModulePrint(module, " ", inputs, &activations);
  std::cout << "Activations = " << activations*4 << " bytes" << std::endl;
}



