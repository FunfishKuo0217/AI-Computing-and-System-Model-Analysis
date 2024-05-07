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

void MacCalculation(torch::jit::script::Module module, std::string prefix, 
                        std::vector<torch::jit::IValue> & CurrInputs, 
                        long long *macs,
                        std::string parent_module){
  prefix += "|---";
  for (const auto& s : module.named_children()) {
      // s.value is the submodule
      MacCalculation(s.value, prefix, CurrInputs, macs, s.name);
      // 一定要確保沒有下層 submodules 才是正確的 child!!!
      if (prefix == " |---|---" || s.name == "avgpool"){
        torch::jit::script::Module CurrModule;
        at::Tensor output;
        at::Tensor input = CurrInputs[0].toTensor();
        if (s.name == "avgpool"){
            CurrModule = s.value;
            output = CurrModule.forward(CurrInputs).toTensor();
            // reshape for avgpool
            output = output.view({1, 9216});
        }
        else{
            CurrModule = s.value;
            output = CurrModule.forward(CurrInputs).toTensor();
        }
        // print parameters & mac 計算
        for (const auto& p : CurrModule.named_parameters(/*recurse*/false)){
            if (p.name ==  "weight" && (parent_module == "features" || parent_module == "classifier")){
                // Conv2D
                if(parent_module == "features"){
                    // Macs = kernel_size*kernel_size*in_channel*out_channel*output_element
                    *macs += p.value.sizes()[2] * p.value.sizes()[3] * output.sizes()[1] * output.sizes()[2] * output.sizes()[3] * input.sizes()[1];
                }
                // Linear
                else if(parent_module == "classifier"){
                    // Macs = in_feature * out_features
                    *macs += p.value.sizes()[0] * p.value.sizes()[1];
                }
            }
        }
        CurrInputs.pop_back();
        CurrInputs.push_back(output);
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
  long long macs = 0;
  MacCalculation(module, " ", inputs, &macs, " ");
  std::cout << "MACs = " << macs << std::endl;
}



