# AIAS 2024 Lab 2 Model Analysis


[toc]


## HW 2-1 Model Analysis Using Pytorch

### 2-1-1. Calculate the number of model parameters：

#### Code
```python=
import torchvision.models as models

# 加載 GoogLeNet 模型
model = models.googlenet(pretrained=True)
print(model)

input_shape = (3, 224, 224)


total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)
```

#### Execution Result
```
Total number of parameters:  6624904
```

### 2-1-2. Calculate memory requirements for storing the model weights.
#### Code
```python=
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print("Total memory for parameters: ", param_size)
```

#### Execution Result
```
Total memory for parameters:  26499616
```



### 2-1-3. Use Torchinfo to print model architecture including the number of parameters and the output activation size of each layer 
#### Code
```python=
import torchinfo
# input_shape = (3, 224, 224)
torchinfo.summary(model, input_shape, batch_dim = 0, col_names=("output_size", "num_params"), verbose=0)
```

#### Execution Result
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GoogLeNet                                [1, 1000]                 --
├─BasicConv2d: 1-1                       [1, 64, 112, 112]         --
│    └─Conv2d: 2-1                       [1, 64, 112, 112]         9,408
│    └─BatchNorm2d: 2-2                  [1, 64, 112, 112]         128
├─MaxPool2d: 1-2                         [1, 64, 56, 56]           --
├─BasicConv2d: 1-3                       [1, 64, 56, 56]           --
│    └─Conv2d: 2-3                       [1, 64, 56, 56]           4,096
│    └─BatchNorm2d: 2-4                  [1, 64, 56, 56]           128
├─BasicConv2d: 1-4                       [1, 192, 56, 56]          --
│    └─Conv2d: 2-5                       [1, 192, 56, 56]          110,592
│    └─BatchNorm2d: 2-6                  [1, 192, 56, 56]          384
├─MaxPool2d: 1-5                         [1, 192, 28, 28]          --
├─Inception: 1-6                         [1, 256, 28, 28]          --
│    └─BasicConv2d: 2-7                  [1, 64, 28, 28]           --
│    │    └─Conv2d: 3-1                  [1, 64, 28, 28]           12,288
│    │    └─BatchNorm2d: 3-2             [1, 64, 28, 28]           128
│    └─Sequential: 2-8                   [1, 128, 28, 28]          --
│    │    └─BasicConv2d: 3-3             [1, 96, 28, 28]           18,624
│    │    └─BasicConv2d: 3-4             [1, 128, 28, 28]          110,848
│    └─Sequential: 2-9                   [1, 32, 28, 28]           --
│    │    └─BasicConv2d: 3-5             [1, 16, 28, 28]           3,104
│    │    └─BasicConv2d: 3-6             [1, 32, 28, 28]           4,672
│    └─Sequential: 2-10                  [1, 32, 28, 28]           --
│    │    └─MaxPool2d: 3-7               [1, 192, 28, 28]          --
│    │    └─BasicConv2d: 3-8             [1, 32, 28, 28]           6,208
├─Inception: 1-7                         [1, 480, 28, 28]          --
│    └─BasicConv2d: 2-11                 [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-9                  [1, 128, 28, 28]          32,768
│    │    └─BatchNorm2d: 3-10            [1, 128, 28, 28]          256
│    └─Sequential: 2-12                  [1, 192, 28, 28]          --
│    │    └─BasicConv2d: 3-11            [1, 128, 28, 28]          33,024
│    │    └─BasicConv2d: 3-12            [1, 192, 28, 28]          221,568
│    └─Sequential: 2-13                  [1, 96, 28, 28]           --
│    │    └─BasicConv2d: 3-13            [1, 32, 28, 28]           8,256
│    │    └─BasicConv2d: 3-14            [1, 96, 28, 28]           27,840
│    └─Sequential: 2-14                  [1, 64, 28, 28]           --
│    │    └─MaxPool2d: 3-15              [1, 256, 28, 28]          --
│    │    └─BasicConv2d: 3-16            [1, 64, 28, 28]           16,512
├─MaxPool2d: 1-8                         [1, 480, 14, 14]          --
├─Inception: 1-9                         [1, 512, 14, 14]          --
│    └─BasicConv2d: 2-15                 [1, 192, 14, 14]          --
│    │    └─Conv2d: 3-17                 [1, 192, 14, 14]          92,160
│    │    └─BatchNorm2d: 3-18            [1, 192, 14, 14]          384
│    └─Sequential: 2-16                  [1, 208, 14, 14]          --
│    │    └─BasicConv2d: 3-19            [1, 96, 14, 14]           46,272
│    │    └─BasicConv2d: 3-20            [1, 208, 14, 14]          180,128
│    └─Sequential: 2-17                  [1, 48, 14, 14]           --
│    │    └─BasicConv2d: 3-21            [1, 16, 14, 14]           7,712
│    │    └─BasicConv2d: 3-22            [1, 48, 14, 14]           7,008
│    └─Sequential: 2-18                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-23              [1, 480, 14, 14]          --
│    │    └─BasicConv2d: 3-24            [1, 64, 14, 14]           30,848
├─Inception: 1-10                        [1, 512, 14, 14]          --
│    └─BasicConv2d: 2-19                 [1, 160, 14, 14]          --
│    │    └─Conv2d: 3-25                 [1, 160, 14, 14]          81,920
│    │    └─BatchNorm2d: 3-26            [1, 160, 14, 14]          320
│    └─Sequential: 2-20                  [1, 224, 14, 14]          --
│    │    └─BasicConv2d: 3-27            [1, 112, 14, 14]          57,568
│    │    └─BasicConv2d: 3-28            [1, 224, 14, 14]          226,240
│    └─Sequential: 2-21                  [1, 64, 14, 14]           --
│    │    └─BasicConv2d: 3-29            [1, 24, 14, 14]           12,336
│    │    └─BasicConv2d: 3-30            [1, 64, 14, 14]           13,952
│    └─Sequential: 2-22                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-31              [1, 512, 14, 14]          --
│    │    └─BasicConv2d: 3-32            [1, 64, 14, 14]           32,896
├─Inception: 1-11                        [1, 512, 14, 14]          --
│    └─BasicConv2d: 2-23                 [1, 128, 14, 14]          --
│    │    └─Conv2d: 3-33                 [1, 128, 14, 14]          65,536
│    │    └─BatchNorm2d: 3-34            [1, 128, 14, 14]          256
│    └─Sequential: 2-24                  [1, 256, 14, 14]          --
│    │    └─BasicConv2d: 3-35            [1, 128, 14, 14]          65,792
│    │    └─BasicConv2d: 3-36            [1, 256, 14, 14]          295,424
│    └─Sequential: 2-25                  [1, 64, 14, 14]           --
│    │    └─BasicConv2d: 3-37            [1, 24, 14, 14]           12,336
│    │    └─BasicConv2d: 3-38            [1, 64, 14, 14]           13,952
│    └─Sequential: 2-26                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-39              [1, 512, 14, 14]          --
│    │    └─BasicConv2d: 3-40            [1, 64, 14, 14]           32,896
├─Inception: 1-12                        [1, 528, 14, 14]          --
│    └─BasicConv2d: 2-27                 [1, 112, 14, 14]          --
│    │    └─Conv2d: 3-41                 [1, 112, 14, 14]          57,344
│    │    └─BatchNorm2d: 3-42            [1, 112, 14, 14]          224
│    └─Sequential: 2-28                  [1, 288, 14, 14]          --
│    │    └─BasicConv2d: 3-43            [1, 144, 14, 14]          74,016
│    │    └─BasicConv2d: 3-44            [1, 288, 14, 14]          373,824
│    └─Sequential: 2-29                  [1, 64, 14, 14]           --
│    │    └─BasicConv2d: 3-45            [1, 32, 14, 14]           16,448
│    │    └─BasicConv2d: 3-46            [1, 64, 14, 14]           18,560
│    └─Sequential: 2-30                  [1, 64, 14, 14]           --
│    │    └─MaxPool2d: 3-47              [1, 512, 14, 14]          --
│    │    └─BasicConv2d: 3-48            [1, 64, 14, 14]           32,896
├─Inception: 1-13                        [1, 832, 14, 14]          --
│    └─BasicConv2d: 2-31                 [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-49                 [1, 256, 14, 14]          135,168
│    │    └─BatchNorm2d: 3-50            [1, 256, 14, 14]          512
│    └─Sequential: 2-32                  [1, 320, 14, 14]          --
│    │    └─BasicConv2d: 3-51            [1, 160, 14, 14]          84,800
│    │    └─BasicConv2d: 3-52            [1, 320, 14, 14]          461,440
│    └─Sequential: 2-33                  [1, 128, 14, 14]          --
│    │    └─BasicConv2d: 3-53            [1, 32, 14, 14]           16,960
│    │    └─BasicConv2d: 3-54            [1, 128, 14, 14]          37,120
│    └─Sequential: 2-34                  [1, 128, 14, 14]          --
│    │    └─MaxPool2d: 3-55              [1, 528, 14, 14]          --
│    │    └─BasicConv2d: 3-56            [1, 128, 14, 14]          67,840
├─MaxPool2d: 1-14                        [1, 832, 7, 7]            --
├─Inception: 1-15                        [1, 832, 7, 7]            --
│    └─BasicConv2d: 2-35                 [1, 256, 7, 7]            --
│    │    └─Conv2d: 3-57                 [1, 256, 7, 7]            212,992
│    │    └─BatchNorm2d: 3-58            [1, 256, 7, 7]            512
│    └─Sequential: 2-36                  [1, 320, 7, 7]            --
│    │    └─BasicConv2d: 3-59            [1, 160, 7, 7]            133,440
│    │    └─BasicConv2d: 3-60            [1, 320, 7, 7]            461,440
│    └─Sequential: 2-37                  [1, 128, 7, 7]            --
│    │    └─BasicConv2d: 3-61            [1, 32, 7, 7]             26,688
│    │    └─BasicConv2d: 3-62            [1, 128, 7, 7]            37,120
│    └─Sequential: 2-38                  [1, 128, 7, 7]            --
│    │    └─MaxPool2d: 3-63              [1, 832, 7, 7]            --
│    │    └─BasicConv2d: 3-64            [1, 128, 7, 7]            106,752
├─Inception: 1-16                        [1, 1024, 7, 7]           --
│    └─BasicConv2d: 2-39                 [1, 384, 7, 7]            --
│    │    └─Conv2d: 3-65                 [1, 384, 7, 7]            319,488
│    │    └─BatchNorm2d: 3-66            [1, 384, 7, 7]            768
│    └─Sequential: 2-40                  [1, 384, 7, 7]            --
│    │    └─BasicConv2d: 3-67            [1, 192, 7, 7]            160,128
│    │    └─BasicConv2d: 3-68            [1, 384, 7, 7]            664,320
│    └─Sequential: 2-41                  [1, 128, 7, 7]            --
│    │    └─BasicConv2d: 3-69            [1, 48, 7, 7]             40,032
│    │    └─BasicConv2d: 3-70            [1, 128, 7, 7]            55,552
│    └─Sequential: 2-42                  [1, 128, 7, 7]            --
│    │    └─MaxPool2d: 3-71              [1, 832, 7, 7]            --
│    │    └─BasicConv2d: 3-72            [1, 128, 7, 7]            106,752
├─AdaptiveAvgPool2d: 1-17                [1, 1024, 1, 1]           --
├─Dropout: 1-18                          [1, 1024]                 --
├─Linear: 1-19                           [1, 1000]                 1,025,000
==========================================================================================
Total params: 6,624,904
Trainable params: 6,624,904
Non-trainable params: 0
Total mult-adds (G): 1.50
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 51.63
Params size (MB): 26.50
Estimated Total Size (MB): 78.73
==========================================================================================
```


### 2-1-4. Calculate computation requirements
#### Code
```python=
def calculate_output_shape(input_shape, layer):
    # Calculate the output shape for Conv2d, MaxPool2d, and Linear layers
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        kernel_size = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size, layer.kernel_size)
        )
        stride = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride, layer.stride)
        )
        padding = (
            layer.padding
            if isinstance(layer.padding, tuple)
            else (layer.padding, layer.padding)
        )
        dilation = (
            layer.dilation
            if isinstance(layer.dilation, tuple)
            else (layer.dilation, layer.dilation)
        )

        output_height = (
            input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        output_width = (
            input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
        return (
            layer.out_channels if hasattr(layer, "out_channels") else input_shape[0],
            output_height,
            output_width,
        )
    elif isinstance(layer, nn.Linear):
        # For Linear layers, the output shape is simply the layer's output features
        return (layer.out_features,)
    else:
        return input_shape


def calculate_macs(layer, input_shape, output_shape):
    # Calculate MACs for Conv2d and Linear layers
    if isinstance(layer, nn.Conv2d):
        kernel_ops = (
            layer.kernel_size[0]
            * layer.kernel_size[1]
            * (layer.in_channels / layer.groups)
        )
        output_elements = output_shape[1] * output_shape[2]
        macs = int(kernel_ops * output_elements * layer.out_channels)
        return macs
    elif isinstance(layer, nn.Linear):
        # For Linear layers, MACs are the product of input features and output features
        macs = int(layer.in_features * layer.out_features)
        return macs
    else:
        return 0

# Initial input shape
input_shape = (3, 224, 224)
total_macs = 0

# Iterate through the layers of the model
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear)):
        output_shape = calculate_output_shape(input_shape, layer)
        macs = calculate_macs(layer, input_shape, output_shape)
        total_macs += macs
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: {macs}"
            )
        elif isinstance(layer, nn.MaxPool2d):
            # Also print shape transformation for MaxPool2d layers (no MACs calculated)
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: N/A"
            )
        input_shape = output_shape  # Update the input shape for the next layer

print(f"Total MACs: {total_macs}")
```

#### Execution Result
```
Layer: conv1.conv, Type: Conv2d, Input Shape: (3, 224, 224), Output Shape: (64, 112, 112), MACs: 118013952
Layer: maxpool1, Type: MaxPool2d, Input Shape: (64, 112, 112), Output Shape: (64, 55, 55), MACs: N/A
Layer: conv2.conv, Type: Conv2d, Input Shape: (64, 55, 55), Output Shape: (64, 55, 55), MACs: 12390400
Layer: conv3.conv, Type: Conv2d, Input Shape: (64, 55, 55), Output Shape: (192, 55, 55), MACs: 334540800
Layer: maxpool2, Type: MaxPool2d, Input Shape: (192, 55, 55), Output Shape: (192, 27, 27), MACs: N/A
Layer: inception3a.branch1.conv, Type: Conv2d, Input Shape: (192, 27, 27), Output Shape: (64, 27, 27), MACs: 8957952
Layer: inception3a.branch2.0.conv, Type: Conv2d, Input Shape: (64, 27, 27), Output Shape: (96, 27, 27), MACs: 13436928
Layer: inception3a.branch2.1.conv, Type: Conv2d, Input Shape: (96, 27, 27), Output Shape: (128, 27, 27), MACs: 80621568
Layer: inception3a.branch3.0.conv, Type: Conv2d, Input Shape: (128, 27, 27), Output Shape: (16, 27, 27), MACs: 2239488
Layer: inception3a.branch3.1.conv, Type: Conv2d, Input Shape: (16, 27, 27), Output Shape: (32, 27, 27), MACs: 3359232
Layer: inception3a.branch4.0, Type: MaxPool2d, Input Shape: (32, 27, 27), Output Shape: (32, 27, 27), MACs: N/A
Layer: inception3a.branch4.1.conv, Type: Conv2d, Input Shape: (32, 27, 27), Output Shape: (32, 27, 27), MACs: 4478976
Layer: inception3b.branch1.conv, Type: Conv2d, Input Shape: (32, 27, 27), Output Shape: (128, 27, 27), MACs: 23887872
Layer: inception3b.branch2.0.conv, Type: Conv2d, Input Shape: (128, 27, 27), Output Shape: (128, 27, 27), MACs: 23887872
Layer: inception3b.branch2.1.conv, Type: Conv2d, Input Shape: (128, 27, 27), Output Shape: (192, 27, 27), MACs: 161243136
Layer: inception3b.branch3.0.conv, Type: Conv2d, Input Shape: (192, 27, 27), Output Shape: (32, 27, 27), MACs: 5971968
Layer: inception3b.branch3.1.conv, Type: Conv2d, Input Shape: (32, 27, 27), Output Shape: (96, 27, 27), MACs: 20155392
Layer: inception3b.branch4.0, Type: MaxPool2d, Input Shape: (96, 27, 27), Output Shape: (96, 27, 27), MACs: N/A
Layer: inception3b.branch4.1.conv, Type: Conv2d, Input Shape: (96, 27, 27), Output Shape: (64, 27, 27), MACs: 11943936
Layer: maxpool3, Type: MaxPool2d, Input Shape: (64, 27, 27), Output Shape: (64, 13, 13), MACs: N/A
Layer: inception4a.branch1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (192, 13, 13), MACs: 15575040
Layer: inception4a.branch2.0.conv, Type: Conv2d, Input Shape: (192, 13, 13), Output Shape: (96, 13, 13), MACs: 7787520
Layer: inception4a.branch2.1.conv, Type: Conv2d, Input Shape: (96, 13, 13), Output Shape: (208, 13, 13), MACs: 30371328
Layer: inception4a.branch3.0.conv, Type: Conv2d, Input Shape: (208, 13, 13), Output Shape: (16, 13, 13), MACs: 1297920
Layer: inception4a.branch3.1.conv, Type: Conv2d, Input Shape: (16, 13, 13), Output Shape: (48, 13, 13), MACs: 1168128
Layer: inception4a.branch4.0, Type: MaxPool2d, Input Shape: (48, 13, 13), Output Shape: (48, 13, 13), MACs: N/A
Layer: inception4a.branch4.1.conv, Type: Conv2d, Input Shape: (48, 13, 13), Output Shape: (64, 13, 13), MACs: 5191680
Layer: inception4b.branch1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (160, 13, 13), MACs: 13844480
Layer: inception4b.branch2.0.conv, Type: Conv2d, Input Shape: (160, 13, 13), Output Shape: (112, 13, 13), MACs: 9691136
Layer: inception4b.branch2.1.conv, Type: Conv2d, Input Shape: (112, 13, 13), Output Shape: (224, 13, 13), MACs: 38158848
Layer: inception4b.branch3.0.conv, Type: Conv2d, Input Shape: (224, 13, 13), Output Shape: (24, 13, 13), MACs: 2076672
Layer: inception4b.branch3.1.conv, Type: Conv2d, Input Shape: (24, 13, 13), Output Shape: (64, 13, 13), MACs: 2336256
Layer: inception4b.branch4.0, Type: MaxPool2d, Input Shape: (64, 13, 13), Output Shape: (64, 13, 13), MACs: N/A
Layer: inception4b.branch4.1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (64, 13, 13), MACs: 5537792
Layer: inception4c.branch1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (128, 13, 13), MACs: 11075584
Layer: inception4c.branch2.0.conv, Type: Conv2d, Input Shape: (128, 13, 13), Output Shape: (128, 13, 13), MACs: 11075584
Layer: inception4c.branch2.1.conv, Type: Conv2d, Input Shape: (128, 13, 13), Output Shape: (256, 13, 13), MACs: 49840128
Layer: inception4c.branch3.0.conv, Type: Conv2d, Input Shape: (256, 13, 13), Output Shape: (24, 13, 13), MACs: 2076672
Layer: inception4c.branch3.1.conv, Type: Conv2d, Input Shape: (24, 13, 13), Output Shape: (64, 13, 13), MACs: 2336256
Layer: inception4c.branch4.0, Type: MaxPool2d, Input Shape: (64, 13, 13), Output Shape: (64, 13, 13), MACs: N/A
Layer: inception4c.branch4.1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (64, 13, 13), MACs: 5537792
Layer: inception4d.branch1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (112, 13, 13), MACs: 9691136
Layer: inception4d.branch2.0.conv, Type: Conv2d, Input Shape: (112, 13, 13), Output Shape: (144, 13, 13), MACs: 12460032
Layer: inception4d.branch2.1.conv, Type: Conv2d, Input Shape: (144, 13, 13), Output Shape: (288, 13, 13), MACs: 63078912
Layer: inception4d.branch3.0.conv, Type: Conv2d, Input Shape: (288, 13, 13), Output Shape: (32, 13, 13), MACs: 2768896
Layer: inception4d.branch3.1.conv, Type: Conv2d, Input Shape: (32, 13, 13), Output Shape: (64, 13, 13), MACs: 3115008
Layer: inception4d.branch4.0, Type: MaxPool2d, Input Shape: (64, 13, 13), Output Shape: (64, 13, 13), MACs: N/A
Layer: inception4d.branch4.1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (64, 13, 13), MACs: 5537792
Layer: inception4e.branch1.conv, Type: Conv2d, Input Shape: (64, 13, 13), Output Shape: (256, 13, 13), MACs: 22843392
Layer: inception4e.branch2.0.conv, Type: Conv2d, Input Shape: (256, 13, 13), Output Shape: (160, 13, 13), MACs: 14277120
Layer: inception4e.branch2.1.conv, Type: Conv2d, Input Shape: (160, 13, 13), Output Shape: (320, 13, 13), MACs: 77875200
Layer: inception4e.branch3.0.conv, Type: Conv2d, Input Shape: (320, 13, 13), Output Shape: (32, 13, 13), MACs: 2855424
Layer: inception4e.branch3.1.conv, Type: Conv2d, Input Shape: (32, 13, 13), Output Shape: (128, 13, 13), MACs: 6230016
Layer: inception4e.branch4.0, Type: MaxPool2d, Input Shape: (128, 13, 13), Output Shape: (128, 13, 13), MACs: N/A
Layer: inception4e.branch4.1.conv, Type: Conv2d, Input Shape: (128, 13, 13), Output Shape: (128, 13, 13), MACs: 11421696
Layer: maxpool4, Type: MaxPool2d, Input Shape: (128, 13, 13), Output Shape: (128, 6, 6), MACs: N/A
Layer: inception5a.branch1.conv, Type: Conv2d, Input Shape: (128, 6, 6), Output Shape: (256, 6, 6), MACs: 7667712
Layer: inception5a.branch2.0.conv, Type: Conv2d, Input Shape: (256, 6, 6), Output Shape: (160, 6, 6), MACs: 4792320
Layer: inception5a.branch2.1.conv, Type: Conv2d, Input Shape: (160, 6, 6), Output Shape: (320, 6, 6), MACs: 16588800
Layer: inception5a.branch3.0.conv, Type: Conv2d, Input Shape: (320, 6, 6), Output Shape: (32, 6, 6), MACs: 958464
Layer: inception5a.branch3.1.conv, Type: Conv2d, Input Shape: (32, 6, 6), Output Shape: (128, 6, 6), MACs: 1327104
Layer: inception5a.branch4.0, Type: MaxPool2d, Input Shape: (128, 6, 6), Output Shape: (128, 6, 6), MACs: N/A
Layer: inception5a.branch4.1.conv, Type: Conv2d, Input Shape: (128, 6, 6), Output Shape: (128, 6, 6), MACs: 3833856
Layer: inception5b.branch1.conv, Type: Conv2d, Input Shape: (128, 6, 6), Output Shape: (384, 6, 6), MACs: 11501568
Layer: inception5b.branch2.0.conv, Type: Conv2d, Input Shape: (384, 6, 6), Output Shape: (192, 6, 6), MACs: 5750784
Layer: inception5b.branch2.1.conv, Type: Conv2d, Input Shape: (192, 6, 6), Output Shape: (384, 6, 6), MACs: 23887872
Layer: inception5b.branch3.0.conv, Type: Conv2d, Input Shape: (384, 6, 6), Output Shape: (48, 6, 6), MACs: 1437696
Layer: inception5b.branch3.1.conv, Type: Conv2d, Input Shape: (48, 6, 6), Output Shape: (128, 6, 6), MACs: 1990656
Layer: inception5b.branch4.0, Type: MaxPool2d, Input Shape: (128, 6, 6), Output Shape: (128, 6, 6), MACs: N/A
Layer: inception5b.branch4.1.conv, Type: Conv2d, Input Shape: (128, 6, 6), Output Shape: (128, 6, 6), MACs: 3833856
Layer: fc, Type: Linear, Input Shape: (128, 6, 6), Output Shape: (1000,), MACs: 1024000
Total MACs: 1356857600

```

### 2-1-5. Use forward hooks to extract the output activations of  the Conv2d layers.
#### Code
```python=
import torch
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
```

#### Execution Result
```
Activation from layer conv1.conv: torch.Size([1, 64, 112, 112])
Activation from layer conv2.conv: torch.Size([1, 64, 56, 56])
Activation from layer conv3.conv: torch.Size([1, 192, 56, 56])
Activation from layer inception3a.branch1.conv: torch.Size([1, 64, 28, 28])
Activation from layer inception3a.branch2.0.conv: torch.Size([1, 96, 28, 28])
Activation from layer inception3a.branch2.1.conv: torch.Size([1, 128, 28, 28])
Activation from layer inception3a.branch3.0.conv: torch.Size([1, 16, 28, 28])
Activation from layer inception3a.branch3.1.conv: torch.Size([1, 32, 28, 28])
Activation from layer inception3a.branch4.1.conv: torch.Size([1, 32, 28, 28])
Activation from layer inception3b.branch1.conv: torch.Size([1, 128, 28, 28])
Activation from layer inception3b.branch2.0.conv: torch.Size([1, 128, 28, 28])
Activation from layer inception3b.branch2.1.conv: torch.Size([1, 192, 28, 28])
Activation from layer inception3b.branch3.0.conv: torch.Size([1, 32, 28, 28])
Activation from layer inception3b.branch3.1.conv: torch.Size([1, 96, 28, 28])
Activation from layer inception3b.branch4.1.conv: torch.Size([1, 64, 28, 28])
Activation from layer inception4a.branch1.conv: torch.Size([1, 192, 14, 14])
Activation from layer inception4a.branch2.0.conv: torch.Size([1, 96, 14, 14])
Activation from layer inception4a.branch2.1.conv: torch.Size([1, 208, 14, 14])
Activation from layer inception4a.branch3.0.conv: torch.Size([1, 16, 14, 14])
Activation from layer inception4a.branch3.1.conv: torch.Size([1, 48, 14, 14])
Activation from layer inception4a.branch4.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4b.branch1.conv: torch.Size([1, 160, 14, 14])
Activation from layer inception4b.branch2.0.conv: torch.Size([1, 112, 14, 14])
Activation from layer inception4b.branch2.1.conv: torch.Size([1, 224, 14, 14])
Activation from layer inception4b.branch3.0.conv: torch.Size([1, 24, 14, 14])
Activation from layer inception4b.branch3.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4b.branch4.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4c.branch1.conv: torch.Size([1, 128, 14, 14])
Activation from layer inception4c.branch2.0.conv: torch.Size([1, 128, 14, 14])
Activation from layer inception4c.branch2.1.conv: torch.Size([1, 256, 14, 14])
Activation from layer inception4c.branch3.0.conv: torch.Size([1, 24, 14, 14])
Activation from layer inception4c.branch3.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4c.branch4.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4d.branch1.conv: torch.Size([1, 112, 14, 14])
Activation from layer inception4d.branch2.0.conv: torch.Size([1, 144, 14, 14])
Activation from layer inception4d.branch2.1.conv: torch.Size([1, 288, 14, 14])
Activation from layer inception4d.branch3.0.conv: torch.Size([1, 32, 14, 14])
Activation from layer inception4d.branch3.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4d.branch4.1.conv: torch.Size([1, 64, 14, 14])
Activation from layer inception4e.branch1.conv: torch.Size([1, 256, 14, 14])
Activation from layer inception4e.branch2.0.conv: torch.Size([1, 160, 14, 14])
Activation from layer inception4e.branch2.1.conv: torch.Size([1, 320, 14, 14])
Activation from layer inception4e.branch3.0.conv: torch.Size([1, 32, 14, 14])
Activation from layer inception4e.branch3.1.conv: torch.Size([1, 128, 14, 14])
Activation from layer inception4e.branch4.1.conv: torch.Size([1, 128, 14, 14])
Activation from layer inception5a.branch1.conv: torch.Size([1, 256, 7, 7])
Activation from layer inception5a.branch2.0.conv: torch.Size([1, 160, 7, 7])
Activation from layer inception5a.branch2.1.conv: torch.Size([1, 320, 7, 7])
Activation from layer inception5a.branch3.0.conv: torch.Size([1, 32, 7, 7])
Activation from layer inception5a.branch3.1.conv: torch.Size([1, 128, 7, 7])
Activation from layer inception5a.branch4.1.conv: torch.Size([1, 128, 7, 7])
Activation from layer inception5b.branch1.conv: torch.Size([1, 384, 7, 7])
Activation from layer inception5b.branch2.0.conv: torch.Size([1, 192, 7, 7])
Activation from layer inception5b.branch2.1.conv: torch.Size([1, 384, 7, 7])
Activation from layer inception5b.branch3.0.conv: torch.Size([1, 48, 7, 7])
Activation from layer inception5b.branch3.1.conv: torch.Size([1, 128, 7, 7])
Activation from layer inception5b.branch4.1.conv: torch.Size([1, 128, 7, 7])
```

## HW 2-2 Add more statistics to analyze the an ONNX model Using sclblonnx

### 2-2-1. model characteristics
#### Code
```python=
import onnx 
import json
onnx_model = onnx.load('./mobilenetv2-10.onnx')
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

## List all tensor names in the graph
input_nlist = [k.name for k in onnx_model.graph.input]
initializer_nlist = [k.name for k in onnx_model.graph.initializer]
value_info_nlist = [k.name for k in onnx_model.graph.value_info]

##### Collect Attribute ####
def OperatorAttr(op_type):
    JSON_list = []
    for i in onnx_model.graph.node:
        if i.op_type == op_type:
            JSON = {}
            JSON[i.name] = {}
            if i.op_type == 'Conv':
                for j in i.input:
                    if j in input_nlist:
                        idx = input_nlist.index(j)
                        (dims, size) = get_size(onnx_model.graph.input[idx].type.tensor_type.shape)
                        c = dims[1]
                        h = dims[2]
                        w = dims[3]
                        
                    elif j in initializer_nlist:
                        idx = initializer_nlist.index(j)
                        dims = onnx_model.graph.initializer[idx].dims
                    elif j in value_info_nlist:
                        idx = value_info_nlist.index(j)
                        (dims, size) = get_size(onnx_model.graph.value_info[idx].type.tensor_type.shape)
                        c = dims[1]
                        h = dims[2]
                        w = dims[3]
                    JSON[i.name]['channel'] = c
                    JSON[i.name]['height'] = h
                    JSON[i.name]['width'] = w
            for attr in i.attribute:
                JSON[i.name][attr.name] = attr.i if len(attr.ints) == 0 else attr.ints
                # str.replace(old, new[, max])
            JSON_list.append(JSON)
    return JSON_list

def Print_JSON(JSON):
    key = list(JSON.keys())[0]
    print(key)
    for ckey in JSON[key]:
        print(f'|- {ckey} : {JSON[key][ckey]}')
    

op_dict = {}
op_operator = set()

# Compute the each op_type
for i in onnx_model.graph.node:
    if i.op_type in op_dict:
        op_dict[i.op_type] += 1
    else:
        op_dict[i.op_type] = 1
    op_operator.add(i.op_type)



print(f'Total operators: {sum(op_dict.values())}')
print(f'Unique Operator: {op_operator}')
print('=====')
for op in op_dict:
    print(f'Operator[{op}]: {op_dict[op]}')
    JSON_list = OperatorAttr(op)
    if len(JSON_list) == 0:
        print('No attribute')
    for item in JSON_list:
        Print_JSON(item)
    print('====')
```

#### Execution Result
```
Total operators: 105
Unique Operator: {'Concat', 'Conv', 'Reshape', 'Shape', 'Add', 'Unsqueeze', 'Gemm', 'Gather', 'Clip', 'Constant', 'GlobalAveragePool'}
=====
Operator[Conv]: 52
Conv_0
|- channel : 3
|- height : 224
|- width : 224
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [2, 2]
Conv_2
|- channel : 32
|- height : 112
|- width : 112
|- dilations : [1, 1]
|- group : 32
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_4
|- channel : 32
|- height : 112
|- width : 112
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_5
|- channel : 16
|- height : 112
|- width : 112
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_7
|- channel : 96
|- height : 112
|- width : 112
|- dilations : [1, 1]
|- group : 96
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [2, 2]
Conv_9
|- channel : 96
|- height : 56
|- width : 56
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_10
|- channel : 24
|- height : 56
|- width : 56
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_12
|- channel : 144
|- height : 56
|- width : 56
|- dilations : [1, 1]
|- group : 144
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_14
|- channel : 144
|- height : 56
|- width : 56
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_16
|- channel : 24
|- height : 56
|- width : 56
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_18
|- channel : 144
|- height : 56
|- width : 56
|- dilations : [1, 1]
|- group : 144
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [2, 2]
Conv_20
|- channel : 144
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_21
|- channel : 32
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_23
|- channel : 192
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 192
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_25
|- channel : 192
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_27
|- channel : 32
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_29
|- channel : 192
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 192
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_31
|- channel : 192
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_33
|- channel : 32
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_35
|- channel : 192
|- height : 28
|- width : 28
|- dilations : [1, 1]
|- group : 192
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [2, 2]
Conv_37
|- channel : 192
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_38
|- channel : 64
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_40
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 384
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_42
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_44
|- channel : 64
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_46
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 384
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_48
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_50
|- channel : 64
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_52
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 384
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_54
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_56
|- channel : 64
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_58
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 384
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_60
|- channel : 384
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_61
|- channel : 96
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_63
|- channel : 576
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 576
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_65
|- channel : 576
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_67
|- channel : 96
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_69
|- channel : 576
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 576
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_71
|- channel : 576
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_73
|- channel : 96
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_75
|- channel : 576
|- height : 14
|- width : 14
|- dilations : [1, 1]
|- group : 576
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [2, 2]
Conv_77
|- channel : 576
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_78
|- channel : 160
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_80
|- channel : 960
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 960
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_82
|- channel : 960
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_84
|- channel : 160
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_86
|- channel : 960
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 960
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_88
|- channel : 960
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_90
|- channel : 160
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_92
|- channel : 960
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 960
|- kernel_shape : [3, 3]
|- pads : [1, 1, 1, 1]
|- strides : [1, 1]
Conv_94
|- channel : 960
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
Conv_95
|- channel : 320
|- height : 7
|- width : 7
|- dilations : [1, 1]
|- group : 1
|- kernel_shape : [1, 1]
|- pads : [0, 0, 0, 0]
|- strides : [1, 1]
====
Operator[Clip]: 35
Clip_1
|- max : 0
|- min : 0
Clip_3
|- max : 0
|- min : 0
Clip_6
|- max : 0
|- min : 0
Clip_8
|- max : 0
|- min : 0
Clip_11
|- max : 0
|- min : 0
Clip_13
|- max : 0
|- min : 0
Clip_17
|- max : 0
|- min : 0
Clip_19
|- max : 0
|- min : 0
Clip_22
|- max : 0
|- min : 0
Clip_24
|- max : 0
|- min : 0
Clip_28
|- max : 0
|- min : 0
Clip_30
|- max : 0
|- min : 0
Clip_34
|- max : 0
|- min : 0
Clip_36
|- max : 0
|- min : 0
Clip_39
|- max : 0
|- min : 0
Clip_41
|- max : 0
|- min : 0
Clip_45
|- max : 0
|- min : 0
Clip_47
|- max : 0
|- min : 0
Clip_51
|- max : 0
|- min : 0
Clip_53
|- max : 0
|- min : 0
Clip_57
|- max : 0
|- min : 0
Clip_59
|- max : 0
|- min : 0
Clip_62
|- max : 0
|- min : 0
Clip_64
|- max : 0
|- min : 0
Clip_68
|- max : 0
|- min : 0
Clip_70
|- max : 0
|- min : 0
Clip_74
|- max : 0
|- min : 0
Clip_76
|- max : 0
|- min : 0
Clip_79
|- max : 0
|- min : 0
Clip_81
|- max : 0
|- min : 0
Clip_85
|- max : 0
|- min : 0
Clip_87
|- max : 0
|- min : 0
Clip_91
|- max : 0
|- min : 0
Clip_93
|- max : 0
|- min : 0
Clip_96
|- max : 0
|- min : 0
====
Operator[Add]: 10
Add_15
Add_26
Add_32
Add_43
Add_49
Add_55
Add_66
Add_72
Add_83
Add_89
====
Operator[GlobalAveragePool]: 1
GlobalAveragePool_97
====
Operator[Shape]: 1
Shape_98
====
Operator[Constant]: 1
Constant_99
|- value : 0
====
Operator[Gather]: 1
Gather_100
|- axis : 0
====
Operator[Unsqueeze]: 1
Unsqueeze_101
|- axes : [0]
====
Operator[Concat]: 1
Concat_102
|- axis : 0
====
Operator[Reshape]: 1
Reshape_103
====
Operator[Gemm]: 1
Gemm_104
|- alpha : 0
|- beta : 0
|- transB : 1
====
```

### 2-2-2. Data bandwidth requirement 
#### Code
```python=
import onnx
from onnx import shape_inference
from os import path
import sys
from tabulate import tabulate
from onnx import onnx_ml_pb2 as xpb2
import torch
from torchvision import models, datasets, transforms as T

mobilenet_v2 = models.mobilenet_v2(pretrained=True)


image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
torch_out = mobilenet_v2(x)

# Export the model
torch.onnx.export(mobilenet_v2,              # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "mobilenet_v2_test.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names


onnx_model = onnx.load("./mobilenet_v2_test.onnx", load_external_data=False)
onnx.checker.check_model(onnx_model)

inferred_model = shape_inference.infer_shapes(onnx_model)
print('shape inference complete ...')

def _parse_element(elem: xpb2.ValueInfoProto):
    name = getattr(elem, 'name', "None")
    data_type = "NA"
    shape_str = "NA"
    etype = getattr(elem, 'type', False)
    if etype:
        ttype = getattr(etype, 'tensor_type', False)
        if ttype:
            data_type = getattr(ttype, 'elem_type', 0)
            shape = getattr(elem.type.tensor_type, "shape", False)
            if shape:
                shape_str = "["
                dims = getattr(shape, 'dim', [])
                for dim in dims:
                    vals = getattr(dim, 'dim_value', "?")
                    shape_str += (str(vals) + ",")
                shape_str = shape_str.rstrip(",")
                shape_str += "]"
    return name, data_type, shape_str

def get_valueproto_or_tensorproto_by_name(name: str, graph: xpb2.GraphProto):
    for i, node in enumerate(inferred_model.graph.node):
            if node.name == "":
                inferred_model.graph.node[i].name = str(i)
    input_nlist = [k.name for k in graph.input]
    initializer_nlist = [k.name for k in graph.initializer]
    value_info_nlist = [k.name for k in graph.value_info]
    output_nlist = [k.name for k in graph.output]

    # get tensor data
    if name in input_nlist:
        idx = input_nlist.index(name)
        return graph.input[idx], int(1)
    elif name in value_info_nlist:
        idx = value_info_nlist.index(name)
        return graph.value_info[idx], int(2)
    elif name in initializer_nlist:
        idx = initializer_nlist.index(name)
        return graph.initializer[idx], int(3)
    elif name in output_nlist:
        idx = output_nlist.index(name)
        return graph.output[idx], int(4)
    else:
        print("[ERROR MASSAGE] Can't find the tensor: ", name)
        print('input_nlist:\n', input_nlist)
        print('===================')
        print('value_info_nlist:\n', value_info_nlist)
        print('===================')
        print('initializer_nlist:\n', initializer_nlist)
        print('===================')
        print('output_nlist:\n', output_nlist)
        print('===================')
        return False, 0

def cal_tensor_mem_size(elem_type: str, shape: [int]):
    """ given the element type of the tensor and its shape, and return its memory size.

    Utility.

    Args:
        ttype: the type of the element of the given tensor. format: 'int', ...
        shape: the shape of the given tensor. format: [] of int

    Returns:
        mem_size: int
    """
    # init
    mem_size = int(1)
    # traverse the list to get the number of the elements
    # print(shape)
    for num in shape:
        mem_size *= num
    # multiple the size of variable with the number of the elements
    # "FLOAT": 1,
    # "UINT8": 2,
    # "INT8": 3,
    # "UINT16": 4,
    # "INT16": 5,
    # "INT32": 6,
    # "INT64": 7,
    # # "STRING" : 8,
    # "BOOL": 9,
    # "FLOAT16": 10,
    # "DOUBLE": 11,
    # "UINT32": 12,
    # "UINT64": 13,
    # "COMPLEX64": 14,
    # "COMPLEX128": 15
    if elem_type == 1:
        mem_size *= 4
    elif elem_type == 2:
        mem_size *= 1
    elif elem_type == 3:
        mem_size *= 1
    elif elem_type == 4:
        mem_size *= 2
    elif elem_type == 5:
        mem_size *= 2
    elif elem_type == 6:
        mem_size *= 4
    elif elem_type == 7:
        mem_size *= 8
    elif elem_type == 9:
        mem_size *= 1
    elif elem_type == 10:
        mem_size *= 2
    elif elem_type == 11:
        mem_size *= 8
    elif elem_type == 12:
        mem_size *= 4
    elif elem_type == 13:
        mem_size *= 8
    elif elem_type == 14:
        mem_size *= 8
    elif elem_type == 15:
        mem_size *= 16
    else:
        print("Undefined data type")

    return mem_size



def get_bandwidth(graph: xpb2.GraphProto):
    try:
        mem_BW_list = []
        total_mem_BW = 0
        unknown_tensor_list = []
        # traverse all the nodes
        for nodeProto in graph.node:
            if nodeProto.op_type == 'Constant':
                continue
            # init variables
            read_mem_BW_each_layer = 0
            write_mem_BW_each_layer = 0
            total_each_layer = 0
            # traverse all input tensor
            for input_name in nodeProto.input:
                # get the TensorProto/ValueInfoProto by searching its name
                proto, type_Num = get_valueproto_or_tensorproto_by_name(
                    input_name, graph)
                # parse the ValueInfoProto/TensorProto
                if proto:
                    if type_Num == 3:
                        dtype = getattr(proto, 'data_type', False)
                        # get the shape of the tensor
                        shape = getattr(proto, 'dims', [])
                    elif type_Num == 1 or type_Num == 2:
                        name, dtype, shape_str = _parse_element(proto)
                        shape_str = shape_str.strip('[]')
                        shape_str = shape_str.split(',')
                        shape = []
                        for dim in shape_str:
                            try:
                                shape.append(int(dim))
                            except:
                                shape.append(0)
                    else:
                        print(
                            '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                            input_name, ' is from a wrong list !')
                else:
                    print(
                        '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                        input_name, ' is no found !')
                    unknown_tensor_list.append(
                        (nodeProto.name, input_name, nodeProto.op_type))
                # calculate the tensor size in btye
                
                read_mem_BW_each_layer += cal_tensor_mem_size(dtype, shape)

            # traverse all output tensor
            for output_name in nodeProto.output:
                # get the TensorProto/ValueInfoProto by searching its name
                proto, type_Num = get_valueproto_or_tensorproto_by_name(
                    output_name, graph)
                # parse the ValueInfoProto
                if proto:
                    if type_Num == 2 or type_Num == 4:
                        # name, dtype, shape = utils._parse_ValueInfoProto(proto)
                        name, dtype, shape_str = _parse_element(proto)
                        shape_str = shape_str.strip('[]')
                        shape_str = shape_str.split(',')
                        shape = []
                        for dim in shape_str:
                            try:
                                shape.append(int(dim))
                            except:
                                shape.append(0)
                            
                    else:
                        print(
                            '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                            output_name, ' is from a wrong list !')
                else:
                    print(
                        '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                        input_name, ' is no found !')
                    unknown_tensor_list.append(
                        (nodeProto.name, output_name, nodeProto.op_type))
                # calculate the tensor size in btye
                write_mem_BW_each_layer += cal_tensor_mem_size(dtype, shape)
            # cal total bw
            total_each_layer = read_mem_BW_each_layer + write_mem_BW_each_layer

            # store into tuple
            temp_tuple = (nodeProto.name, read_mem_BW_each_layer,
                        write_mem_BW_each_layer, total_each_layer)
            #append it
            mem_BW_list.append(temp_tuple)
            # accmulate the value
            total_mem_BW += total_each_layer

        # display the mem_bw of eahc layer
        columns = ['layer', 'read_bw', 'write_bw', 'total_bw']
        # resort the list
        mem_BW_list = sorted(mem_BW_list,
                             key=lambda Layer: Layer[1],
                             reverse=True)
        print(tabulate(mem_BW_list, headers=columns))
        print(
            '====================================================================================\n'
        )
        # display it
        print(
            "The memory bandwidth for processor to execute a whole model without on-chip-buffer is: \n",
            total_mem_BW, '(bytes)\n',
            float(total_mem_BW) / float(1000000), '(MB)\n')
        # display the unknown tensor
        columns = ['op_name', 'unfound_tensor', 'op_type']
        print(tabulate(unknown_tensor_list, headers=columns))
        print(
            '====================================================================================\n'
        )
    except Exception as e:
        print("[ERROR MASSAGE] Unable to display: " + str(e))
        return False

    return True

#從這裡開始
print("start")
get_bandwidth(inferred_model.graph)
```

#### Execution Result
```
shape inference complete ...
start
layer                                              read_bw    write_bw    total_bw
-----------------------------------------------  ---------  ----------  ----------
/classifier/classifier.1/Gemm                      5129120        4000     5133120
/features/features.2/conv/conv.1/conv.1.0/Conv     4820736     1204224     6024960
/features/features.2/conv/conv.0/conv.0.2/Clip     4816896     4816896     9633792
/features/features.3/conv/conv.2/Conv              1820256      301056     2121312
/features/features.3/conv/conv.1/conv.1.0/Conv     1812096     1806336     3618432
/features/features.4/conv/conv.1/conv.1.0/Conv     1812096      451584     2263680
/features/features.3/conv/conv.0/conv.0.2/Clip     1806336     1806336     3612672
/features/features.3/conv/conv.1/conv.1.2/Clip     1806336     1806336     3612672
/features/features.4/conv/conv.0/conv.0.2/Clip     1806336     1806336     3612672
/features/features.18/features.18.0/Conv           1706240      250880     1957120
/features/features.1/conv/conv.1/Conv              1607744      802816     2410560
/features/features.1/conv/conv.0/conv.0.0/Conv     1606912     1605632     3212544
/features/features.0/features.0.2/Clip             1605632     1605632     3211264
/features/features.1/conv/conv.0/conv.0.2/Clip     1605632     1605632     3211264
/features/features.17/conv/conv.2/Conv             1418240       62720     1480960
/features/features.2/conv/conv.2/Conv              1213536      301056     1514592
/features/features.2/conv/conv.1/conv.1.2/Clip     1204224     1204224     2408448
/features/features.2/conv/conv.0/conv.0.0/Conv      809344     4816896     5626240
/features/features.15/conv/conv.2/Conv              803200       31360      834560
/features/features.16/conv/conv.2/Conv              803200       31360      834560
/features/features.12/conv/conv.2/Conv              673152       75264      748416
/features/features.13/conv/conv.2/Conv              673152       75264      748416
/features/features.15/conv/conv.0/conv.0.0/Conv     649600      188160      837760
/features/features.16/conv/conv.0/conv.0.0/Conv     649600      188160      837760
/features/features.17/conv/conv.0/conv.0.0/Conv     649600      188160      837760
/features/features.5/conv/conv.2/Conv               626816      100352      727168
/features/features.6/conv/conv.2/Conv               626816      100352      727168
/features/features.5/conv/conv.1/conv.1.0/Conv      609792      602112     1211904
/features/features.6/conv/conv.1/conv.1.0/Conv      609792      602112     1211904
/features/features.7/conv/conv.1/conv.1.0/Conv      609792      150528      760320
/features/features.0/features.0.0/Conv              605696     1605632     2211328
/features/features.3/Add                            602112      301056      903168
/features/features.5/conv/conv.0/conv.0.2/Clip      602112      602112     1204224
/features/features.5/conv/conv.1/conv.1.2/Clip      602112      602112     1204224
/features/features.6/conv/conv.0/conv.0.2/Clip      602112      602112     1204224
/features/features.6/conv/conv.1/conv.1.2/Clip      602112      602112     1204224
/features/features.7/conv/conv.0/conv.0.2/Clip      602112      602112     1204224
/features/features.14/conv/conv.2/Conv              482176       31360      513536
/features/features.12/conv/conv.1/conv.1.0/Conv     474624      451584      926208
/features/features.13/conv/conv.1/conv.1.0/Conv     474624      451584      926208
/features/features.14/conv/conv.1/conv.1.0/Conv     474624      112896      587520
/features/features.4/conv/conv.2/Conv               470144      100352      570496
/features/features.4/conv/conv.1/conv.1.2/Clip      451584      451584      903168
/features/features.12/conv/conv.0/conv.0.2/Clip     451584      451584      903168
/features/features.12/conv/conv.1/conv.1.2/Clip     451584      451584      903168
/features/features.13/conv/conv.0/conv.0.2/Clip     451584      451584      903168
/features/features.13/conv/conv.1/conv.1.2/Clip     451584      451584      903168
/features/features.14/conv/conv.0/conv.0.2/Clip     451584      451584      903168
/features/features.11/conv/conv.2/Conv              448896       75264      524160
/features/features.8/conv/conv.2/Conv               399616       50176      449792
/features/features.9/conv/conv.2/Conv               399616       50176      449792
/features/features.10/conv/conv.2/Conv              399616       50176      449792
/features/features.8/conv/conv.1/conv.1.0/Conv      316416      301056      617472
/features/features.9/conv/conv.1/conv.1.0/Conv      316416      301056      617472
/features/features.10/conv/conv.1/conv.1.0/Conv     316416      301056      617472
/features/features.11/conv/conv.1/conv.1.0/Conv     316416      301056      617472
/features/features.3/conv/conv.0/conv.0.0/Conv      315456     1806336     2121792
/features/features.4/conv/conv.0/conv.0.0/Conv      315456     1806336     2121792
/features/features.8/conv/conv.0/conv.0.2/Clip      301056      301056      602112
/features/features.8/conv/conv.1/conv.1.2/Clip      301056      301056      602112
/features/features.9/conv/conv.0/conv.0.2/Clip      301056      301056      602112
/features/features.9/conv/conv.1/conv.1.2/Clip      301056      301056      602112
/features/features.10/conv/conv.0/conv.0.2/Clip     301056      301056      602112
/features/features.10/conv/conv.1/conv.1.2/Clip     301056      301056      602112
/features/features.11/conv/conv.0/conv.0.2/Clip     301056      301056      602112
/features/features.11/conv/conv.1/conv.1.2/Clip     301056      301056      602112
/features/features.12/conv/conv.0/conv.0.0/Conv     298752      451584      750336
/features/features.13/conv/conv.0/conv.0.0/Conv     298752      451584      750336
/features/features.14/conv/conv.0/conv.0.0/Conv     298752      451584      750336
/features/features.18/features.18.2/Clip            250880      250880      501760
/GlobalAveragePool                                  250880        5120      256000
/features/features.15/conv/conv.1/conv.1.0/Conv     226560      188160      414720
/features/features.16/conv/conv.1/conv.1.0/Conv     226560      188160      414720
/features/features.17/conv/conv.1/conv.1.0/Conv     226560      188160      414720
/features/features.5/Add                            200704      100352      301056
/features/features.6/Add                            200704      100352      301056
/features/features.7/conv/conv.2/Conv               199936       50176      250112
/features/features.15/conv/conv.0/conv.0.2/Clip     188160      188160      376320
/features/features.15/conv/conv.1/conv.1.2/Clip     188160      188160      376320
/features/features.16/conv/conv.0/conv.0.2/Clip     188160      188160      376320
/features/features.16/conv/conv.1/conv.1.2/Clip     188160      188160      376320
/features/features.17/conv/conv.0/conv.0.2/Clip     188160      188160      376320
/features/features.17/conv/conv.1/conv.1.2/Clip     188160      188160      376320
/features/features.7/conv/conv.1/conv.1.2/Clip      150528      150528      301056
/features/features.12/Add                           150528       75264      225792
/features/features.13/Add                           150528       75264      225792
/features/features.8/conv/conv.0/conv.0.0/Conv      150016      301056      451072
/features/features.9/conv/conv.0/conv.0.0/Conv      150016      301056      451072
/features/features.10/conv/conv.0/conv.0.0/Conv     150016      301056      451072
/features/features.11/conv/conv.0/conv.0.0/Conv     150016      301056      451072
/features/features.5/conv/conv.0/conv.0.0/Conv      125696      602112      727808
/features/features.6/conv/conv.0/conv.0.0/Conv      125696      602112      727808
/features/features.7/conv/conv.0/conv.0.0/Conv      125696      602112      727808
/features/features.14/conv/conv.1/conv.1.2/Clip     112896      112896      225792
/features/features.8/Add                            100352       50176      150528
/features/features.9/Add                            100352       50176      150528
/features/features.10/Add                           100352       50176      150528
/features/features.15/Add                            62720       31360       94080
/features/features.16/Add                            62720       31360       94080
/Flatten                                              5120        5120       10240
====================================================================================

The memory bandwidth for processor to execute a whole model without on-chip-buffer is: 
 119445696 (bytes)
 119.445696 (MB)

op_name    unfound_tensor    op_type
---------  ----------------  ---------
====================================================================================
```

### 2-2-3. activation memory storage requirement
#### Code
```python=
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
```

#### Execution Result
```
Activation memory storage requirement: 107117792 byte (102.16MB)
```

## HW 2-3 Build tool scripts to manipulate an ONNX model graph

### 2-3-1. create a subgraph (1) that consist of a single Linear layer of size MxKxN

#### Code
```python=
# reference: https://onnx.ai/onnx/intro/python.html (onnx linear regression ex)
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# inputs
A = make_tensor_value_info('A', TensorProto.FLOAT, [128, 128])
B = make_tensor_value_info('B', TensorProto.FLOAT, [128, 128])

# outputs, the shape is left undefined
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [128, 128])

# nodes
# It creates a node defined by the operator type MatMul,
node1 = make_node('MatMul', ['A', 'B'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.
graph = make_graph([node1],  # nodes
                    'lr',  # a name
                    [A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)
check_model(onnx_model)

# Display the model
print(onnx_model)

# The serialization
with open("Subgraph1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

#### Visualize the subgraph (1)
![](https://course.playlab.tw/md/uploads/5b921526-a69c-408b-9090-0ee8a9283d8e.png)



### 2-3-2. create a subgraph (2) as shown in the above diagram for the subgraph (1)  

#### Code
```python=
### 2-3-2 訂正: 參數版
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# input
A = make_tensor_value_info('A', TensorProto.FLOAT, [128, 128])
B = make_tensor_value_info('B', TensorProto.FLOAT, [128, 128])
# output
C = make_tensor_value_info('C', TensorProto.FLOAT, [128, 128])

# segment_size
segment_size = 64
num_seg = int(A.type.tensor_type.shape.dim[0].dim_value/segment_size)

# A splitted submatrix
input_var = ['A', 'B']
output_layer_dict = {}
output_layer_dict['A'] = []
output_layer_dict['B'] = []

for input in input_var:
    for x_dim in range(num_seg):
        for y_dim in range(num_seg):
            output_layer_dict[input].append(f'{input}_{x_dim}{y_dim}')

# split node list
split_node1_1 = make_node("Split", inputs=["A"], outputs=["A_0", "A_1"], axis=0)
split_node1_2 = make_node("Split", inputs=["A_0", "A_1"], outputs=output_layer_dict["A"], axis=1)
split_node2_1 = make_node("Split", inputs=["B"], outputs=["B_0", "B_1"], axis=0)
split_node2_2 = make_node("Split", inputs=["B_0", "B_1"], outputs=output_layer_dict["B"], axis=1) 

# matmal node list
h_A = 2
v_A = 2
h_B = 2
v_B = 2

output_layer_dict['A'][:h_B]*h_B + output_layer_dict['A'][h_B:]*h_B

# construct A
A_list = []
for i in range(0, h_B*v_B, h_B):
    A_list += output_layer_dict['A'][i:h_B*i+h_B]*h_B

B_list = []
for i in range(0, h_B):
    for j in range(0, h_B*h_B, h_B):
        B_list += [output_layer_dict['B'][i+j]]

B_list *= 2

matmul_list = []
matmul_node = []
for a, b in zip(A_list, B_list):
    matmul_list.append(f'{a}*{b}')
    matmul_node.append(make_node('MatMul', [a, b], [f'{a}*{b}'], name="2DMM 64*64*64")) # -> C00)


# operatpr - sum
sum_node = []
matmul_idx = 0
for i in range(v_A):
    for j in range(h_B):   
        sum_node.append(make_node('Sum', matmul_list[matmul_idx*h_A:matmul_idx*h_A+h_A], [f"C_{i}{j}"], name = "Sum"))
        matmul_idx += 1

# opeator = concat
node_concat = make_node('Concat', ['C_00', 'C_01', 'C_10', 'C_11'], ['C'], axis = 0)
concat_node = []
for i in range(v_A):
    concat_node.append(make_node('Concat', [f'C_{i}{x}' for x in range(v_B)], [f'C_{i}'], axis = 1))
concat_node.append(make_node('Concat', [f'C_{x}' for x in range(v_A)], ['C'], axis = 0))

graph = make_graph([split_node1_1, split_node1_2, split_node2_1, split_node2_2] + matmul_node + sum_node + concat_node,  # nodes
                    'subgraph-2',  # a name
                    [A, B],  # inputs
                    [C])  # outputs

onnx_model = make_model(graph)
check_model(onnx_model)

# # The serialization
with open("Subgraph2.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

#### Visualize the subgraph (2)
![](https://course.playlab.tw/md/uploads/56e935a2-358b-47a7-927e-8e691cc4bbf4.png)



### 2-3-3. replace the `Linear` layers in the AlexNet with the equivalent subgraphs (2)
#### Code
```python=
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# input_segment
def CustomizeGemm(input_name, weight_name, bias_name, output_name, inSeg, outSeg, prefix):
    inputSeg = inSeg
    outputSeg = outSeg
    # output_segment
    splitted_input = [f"{prefix}/Input_{x}" for x in range(inputSeg)]
    splitted_weight_horizon = [f"{prefix}/W_{x}" for x in range(outputSeg)]
    Splitted_list = {}
    for item in splitted_weight_horizon:
        Splitted_list[item] = []
        for i in range(inputSeg):
            Splitted_list[item].append(f'{prefix}/{item}_{i}')
    # Construct the hidden
    Split_in = make_node("Split", [input_name], splitted_input, axis = 1, num_outputs = inputSeg, name = f'{prefix}/Split/input')
    Split_weight_h = make_node("Split", [weight_name], splitted_weight_horizon, axis = 0, num_outputs = outputSeg, name = f'{prefix}/Split/weight_h')
    Split_node = [Split_in, Split_weight_h]
    for i in range(len(splitted_weight_horizon)):
        weight_blk = splitted_weight_horizon[i]
        Split_node.append(make_node("Split", [weight_blk], Splitted_list[splitted_weight_horizon[i]], num_outputs = inputSeg, axis = 1, name = f'{prefix}/Split/{splitted_weight_horizon[i]}'))
    Gemm_node = []
    Sum_node = []
    for i in range(outputSeg):
        for j in range(inputSeg):
            Gemm_node.append(make_node("Gemm", [splitted_input[j], Splitted_list[splitted_weight_horizon[i]][j]], [f'{prefix}/matmal_{i}_{j}'], alpha = 1.0, beta = 1.0, transA = False, transB = True, name=f"{prefix}/2DMM 64*64*64/matmal_{i}_{j}"))
        Sum_node.append(make_node("Sum", [f'{prefix}/matmal_{i}_{t}' for t in range(inputSeg)], [f'{prefix}/C_{i}'], name = f"{prefix}/Sum/{i}"))
    Concat_node = make_node('Concat', [f'{prefix}/C_{i}' for i in range(outputSeg)], [f'{prefix}/before_bias'], axis = 1, name = f"{prefix}/Concat/C")
    Bias = make_node('Sum', [f'{prefix}/before_bias', bias_name], [output_name], name = f"{prefix}/Sum/bias")
    # combine all node
    insert_node_list = Split_node + Gemm_node + Sum_node + [Concat_node] + [Bias]
    return insert_node_list

insert_node_list = CustomizeGemm('/Flatten_output_0', 'learned_10', 'learned_11', '/classifier/classifier.1/Gemm_output_0', 144, 64, 'CG1')
insert_node_list2 = CustomizeGemm('/classifier/classifier.2/Relu_output_0', 'learned_12', 'learned_13', '/classifier/classifier.4/Gemm_output_0', 64, 64, 'CG2')
insert_node_list3 = CustomizeGemm('/classifier/classifier.5/Relu_output_0', 'learned_14', 'learned_15', 'output1', 64, 50, 'CG3')
# node insertion
onnx_model = onnx.load('./alexnet.onnx')
idx = 15
idx2 = 17
idx3 = 19
graph = onnx_model.graph
original_nodes = onnx_model.graph.node
new_nodes = original_nodes[:idx] + insert_node_list + original_nodes[idx+1:idx2] + insert_node_list2 + original_nodes[idx2+1:idx3] + insert_node_list3
onnx_model.graph.ClearField("node")
onnx_model.graph.node.extend(new_nodes)

onnx_model = make_model(graph)
check_model(onnx_model)

with open("hw2-3-3.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

#### Visualize the transformed model graph
:::info
在這邊為了方便呈現，做了參數的調整：將原本規定的 64 調整 1024 (取 9216, 4096 的最大公因數)，因此可以看到我們在第一層 linear 得到 (9216/1024 = 9) * (4096/1024 = 4) 個 Gemm operators，而第二、第三層 linear 也是比照辦理。但在實際 2-3-3 code 中（如上方 code section），可以看到我們是傳入 144(9216/64)、64(4096/64) 進入 function，以達到功課中 64*64 2DMM 的要求。
:::
![](https://course.playlab.tw/md/uploads/213989d7-7b2a-449d-8c19-beed6856f6b1.png)



### 2-3-4. Correctness Verification
#### Code
```python=
# 使用 onnx 進行 inference 
# ./alexnet 為預存好的 alexnet model
# ./hw2-3-3.onnx 為 modified model
import onnx
import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession("./alexnet.onnx")
print('Starting original alexnet inference...')
input_ = np.random.randn(1, 3, 224, 224).astype(np.float32)
ans_origin = session.run(None,{"actual_input_1": input_})

print('Starting customized alexnet inference...')
session = onnxruntime.InferenceSession("./hw2-3-3.onnx")
ans_test = session.run(None,{"actual_input_1": input_})[0]

# 以 Mean square error 衡量誤差，確保小於 10^-5
print('MSE between two models:', np.square(np.subtract(ans_origin[0], ans_test[0])).mean())
```

#### Execution Result
```
Starting original alexnet inference...
Starting customized alexnet inference...
MSE between two models: 8.943285e-14
```


## HW 2-4 Using Pytorch C++ API to do model analysis on the transformed model graph
### 2-4-0 Solution to model transformation
在 2-4 中因為無法成功做到 onnx -> torch script 的轉換，因此採取的方法是刻一個相同功能的 class，並轉為 .pt 檔（已驗證過正確性，MSE = -2.1606683731079102e-07）
```python=
import math
import torch
import torch.nn as nn
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init


import math
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init

class SegmentedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # 被分割成多少份（ex. 128 -> 2 份）
        self.segment_size = 64
        self.in_segment = int(in_features/64)
        self.out_segment = int(out_features/64)
        # self.padding = 0
        if out_features%self.segment_size != 0:   
            self.out_segment = int(out_features/64)+1
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        
        # store it into ParameterList
        self.P_list = {}
        # init
        for i in range(self.out_segment):
            self.P_list[i] = []
        # split the weight
        for i in range(self.out_segment):
            w_horizon = torch.split(self.weight, self.segment_size, dim = 0)[i]
            w_units = torch.split(w_horizon, self.segment_size, dim = 1)
            for j in range(self.in_segment):
                self.P_list[i].append(w_units[j].T)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input = torch.split(input, self.in_segment)
        return output
            
    def forward(self, input: Tensor) -> Tensor:
        # Reshape input to (batch_size, num_segments, segment_size)
        input = input.view(-1, self.in_segment, self.segment_size)
        # Perform segmented matrix multiplication
        outputs = []
        weight = self.weight.T
        # print(self.weight.shape)
        weight = [torch.split(x, self.segment_size, dim = 1) for x in torch.split(weight, self.segment_size, dim=0)]
        # print(self.out_segment, self.in_segment)
        for j in range(self.out_segment):
            seg_size = weight[0][j].size(1)
            # know the segment size
            output_segment = torch.zeros(1, seg_size)
            for i in range(self.in_segment):
                output_segment = torch.add(output_segment, F.linear(input[0][i].reshape([1, 64]), weight[i][j].T, bias=None))
            outputs.append(output_segment)
        output = torch.cat(outputs, dim=1).reshape(self.out_features)
        output = output.reshape([1, self.out_features])
        # print(output.size())
        # Add bias
        output += self.bias
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class AlexNet_SegmentedLinear(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            SegmentedLinear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            SegmentedLinear(4096, 4096),
            nn.ReLU(inplace=True),
            SegmentedLinear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AlexNet_SegmentedLinear()
model.eval()
print(model)
# Create some sample input in the shape this model expects
dummy_input = torch.randn(1, 3, 224, 224)

my_scripted_model = torch.jit.script(model)
torch.jit.save(my_scripted_model, "SegmentAlexnet.pt")
```
### Commend Example
```
~/projects/hw02/hw2-4/analyzer/build$ ./hw2-4-1 ../../SegmentAlexnet.pt 
```

### 2-4-1. Calculate memory requirements for storing the model weights.

#### Code
```C++=
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
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    size_t total_params = 0;
    for(const auto& p : module.parameters()){
      //std::cout << p.sizes() << std::endl;
      size_t layer_size = 1;
      for( auto iter = p.sizes().begin(); iter != p.sizes().end(); iter++){
        layer_size *= *iter;
      }
      total_params += layer_size;
    }
    std::cout << "total size " << total_params*4 << bytes std::endl;
}
```

#### Execution Result
```
total size: 244403360 bytes
```
與 pytorch 的 alexnet parameter_size*4 以後相等

### 2-4-2. Calculate memory requirements for storing the activations

#### Code
```C=
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
      std::cout << prefix <<  "Submodule Name: " << s.name << std::endl;
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
        std::cout << " output size = " << output.sizes() << std::endl;
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

  std::cout << "ok\n";
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  // at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << RecursiveFunc(module);
  std::cout << "Input Size: " << std::endl;
  std::cout << inputs[0].toTensor().sizes() << std::endl;
  // ChildModule(module, " ", inputs);
  long long activations = 1;
  ChildModulePrint(module, " ", inputs, &activations);
  std::cout << "Activations = " << activations*4 << " bytes" << std::endl;
}
```

#### Execution Result
```
Input Size: 
[1, 3, 224, 224]
Activations = 4392864 bytes
```


### 2-4-3. Calculate computation requirements

#### Code
```C++=
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
```

#### Execution Result
```
MACs = 714188480
```

#### Ans in lab2-3-3
```
Layer: features.0, Type: Conv2d, Input Shape: (3, 224, 224), Output Shape: (64, 55, 55), MACs: 70276800
Layer: features.2, Type: MaxPool2d, Input Shape: (64, 55, 55), Output Shape: (64, 27, 27), MACs: N/A
Layer: features.3, Type: Conv2d, Input Shape: (64, 27, 27), Output Shape: (192, 27, 27), MACs: 223948800
Layer: features.5, Type: MaxPool2d, Input Shape: (192, 27, 27), Output Shape: (192, 13, 13), MACs: N/A
Layer: features.6, Type: Conv2d, Input Shape: (192, 13, 13), Output Shape: (384, 13, 13), MACs: 112140288
Layer: features.8, Type: Conv2d, Input Shape: (384, 13, 13), Output Shape: (256, 13, 13), MACs: 149520384
Layer: features.10, Type: Conv2d, Input Shape: (256, 13, 13), Output Shape: (256, 13, 13), MACs: 99680256
Layer: features.12, Type: MaxPool2d, Input Shape: (256, 13, 13), Output Shape: (256, 6, 6), MACs: N/A
Layer: classifier.1, Type: Linear, Input Shape: (256, 6, 6), Output Shape: (4096,), MACs: 37748736
Layer: classifier.4, Type: Linear, Input Shape: (4096,), Output Shape: (4096,), MACs: 16777216
Layer: classifier.6, Type: Linear, Input Shape: (4096,), Output Shape: (1000,), MACs: 4096000
Total MACs: 714188480
```


### 2-4-4. Compare your results to the result in HW2-1 and HW2-2

#### Discussion
1. **比較 output answer (weight memory)**: 在 weight memory 計算結果相同，猜測可能是因為即使我們的目標是執行 64*64 2DMM 的 operator，但其實 model 使用的 weight 數量是不變的，因此在 weight memory 上不會有影響
2. **比較 output answer (MACs)**: MACs 計算上使用 C++ 計算之結果也與直接使用 lab 提供的計算 mac 的 function 結果相同。推論原因：因為我們僅考慮 Conv 和 Linear，其中 Conv 沒有被抽換所以不影響，而 Linear 層的計算方式為 in_feature * out_feature，因此如果是拆成小的 matrix 做 linear，其最終相加結果仍會相同。
3. **比較 output answer (activation)**：其中讓我自己在寫的時候感到最困惑的是 activation memory，使用 C++ API計算出的結果為 4392864 bytes，但使用 pytorch tool 計算的結果為 4437728 byte (4.23MB)，就我自己計算出來的結果是相差了 44864 bytes，在分析時推論可能是有少算了幾個 layer，但逐一檢查每個 layer 的總和還是沒有發現到底是少計算了哪一層
4. 比較整體執行效率：C++在 build 時會花較久時間，而 pytorch tool 基本上可以很快給出結果。



