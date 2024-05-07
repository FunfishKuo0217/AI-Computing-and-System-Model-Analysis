import torchvision.models as models
import torch
activation = {}
# Define a hook function
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Load a pre-trained AlexNet model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Dictionary to store activations from each layer
activation = {}

# Register hook to each linear layer
for layer_name, layer in model.named_modules():
    layer.register_forward_hook(get_activation(layer_name))

# Run model inference
data = torch.randn(1, 3, 224, 224)
output = model(data)

# Access the saved activations
local_memory = 0
for layer in activation:
    # 所有 layer 的 tensor 皆為 float32，因此以 4 byte 計算
    local_memory += torch.numel(activation[layer])*4
    # print(f"Activation from layer {layer}: {activation[layer].shape}")

print(f"Activation memory storage requirement: {local_memory} byte ({round(local_memory/1048576, 2)}MB)")
