#include <torch/torch.h>
#include <iostream>
#include <unordered_map>

// 定义一个全局变量，用于存储激活值
std::unordered_map<std::string, torch::Tensor> activation;

// 定义一个函数，用于创建钩子
torch::autograd::tensor_list hook(torch::Tensor input, torch::Tensor output) {
    // 将激活值存储到全局变量中
    activation["layer_output"] = output;
    // 返回空的 tensor_list
    return {};
}

int main() {
    // 创建一个简单的模型
    torch::nn::Sequential model(
        torch::nn::Linear(10, 5),
        torch::nn::Linear(5, 1)
    );

    // 注册钩子到模型的指定层
    auto layer = model[0]; // 这里假设我们想要获取第一层的激活值
    layer->register_forward_hook(&hook);

    // 输入数据
    torch::Tensor input = torch::randn({1, 10});

    // 前向传播计算模型输出
    torch::Tensor output = model->forward(input);

    // 获取激活值
    torch::Tensor layer_output = activation["layer_output"];
    std::cout << "Layer output: " << layer_output << std::endl;

    return 0;
}
