import torch
import torchvision.models as models
import torch.nn as nn
# 加載 GoogLeNet 模型
model = models.googlenet(pretrained=True)
input_shape = (3, 224, 224)
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(get_activation(name))

# Model inference
data = torch.randn(1, 3, 224, 224)
output = model(data)


for layer in activation:
    print(f"Activation from layer {layer}: {activation[layer].shape}")
