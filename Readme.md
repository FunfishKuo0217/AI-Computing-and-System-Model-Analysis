# AIAS 2024 Lab 2 HW Submission

此為 2024 修習 人工智慧計算與架構的 hw02，作業目標為針對現有的大型 model 進行分析，理解 model 架構、運算量、儲存量，並使用各種工具如 pytorch, libtorch(c++) 對 model 進行分析。


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

#### Execution Result（節錄部分結果）
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
│    │    └─BasicConv2d: 3-4             [1, 128, 28, 28]          110,
...
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
#### Execution Result(節錄部分結果)
```
Layer: conv1.conv, Type: Conv2d, Input Shape: (3, 224, 224), Output Shape: (64, 112, 112), MACs: 118013952
Layer: maxpool1, Type: MaxPool2d, Input Shape: (64, 112, 112), Output Shape: (64, 55, 55), MACs: N/A
Layer: conv2.conv, Type: Conv2d, Input Shape: (64, 55, 55), Output Shape: (64, 55, 55), MACs: 12390400
Layer: conv3.conv, Type: Conv2d, Input Shape: (64, 55, 55), Output Shape: (192, 55, 55), MACs: 334540800
Layer: maxpool2, Type: MaxPool2d, Input Shape: (192, 55, 55), Output Shape: (192, 27, 27), MACs: N/A
...
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

#### Execution Result(節錄部分結果)
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
...
```

## HW 2-2 Add more statistics to analyze the an ONNX model Using sclblonnx

### 2-2-1. model characteristics
#### Code
請參考 hw2-2-1.py

#### Execution Result(節錄部分結果)
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
...
====
```

### 2-2-2. Data bandwidth requirement 
#### Code
請參考 hw2-2-2.py

#### Execution Result(節錄部分結果)
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
...

The memory bandwidth for processor to execute a whole model without on-chip-buffer is: 
 119445696 (bytes)
 119.445696 (MB)
=========================================================================
```

### 2-2-3. activation memory storage requirement
#### Code
請參考 hw2-2-3.py

#### Execution Result
```
Activation memory storage requirement: 107117792 byte (102.16MB)
```

## HW 2-3 Build tool scripts to manipulate an ONNX model graph

### 2-3-1. create a subgraph (1) that consist of a single Linear layer of size MxKxN

#### Code
請參考 hw2-3-1.py

#### Visualize the subgraph (1)
![](https://course.playlab.tw/md/uploads/5b921526-a69c-408b-9090-0ee8a9283d8e.png)



### 2-3-2. create a subgraph (2) as shown in the above diagram for the subgraph (1)  

#### Code
請參考 hw2-3-2.py

#### Visualize the subgraph (2)
![](https://course.playlab.tw/md/uploads/56e935a2-358b-47a7-927e-8e691cc4bbf4.png)



### 2-3-3. replace the `Linear` layers in the AlexNet with the equivalent subgraphs (2)
#### Code
請參考 hw2-3-3.py

#### Visualize the transformed model graph
:::info
在這邊為了方便呈現，做了參數的調整：將原本規定的 64 調整 1024 (取 9216, 4096 的最大公因數)，因此可以看到我們在第一層 linear 得到 (9216/1024 = 9) * (4096/1024 = 4) 個 Gemm operators，而第二、第三層 linear 也是比照辦理。但在實際 2-3-3 code 中（如上方 code section），可以看到我們是傳入 144(9216/64)、64(4096/64) 進入 function，以達到功課中 64*64 2DMM 的要求。
:::
![](https://course.playlab.tw/md/uploads/213989d7-7b2a-449d-8c19-beed6856f6b1.png)



### 2-3-4. Correctness Verification
#### Code
請參考 hw2-3-4.py
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
請參考 hw2-4-1.py (in analyzer)

#### Execution Result
```
total size: 244403360 bytes
```
與 pytorch 的 alexnet parameter_size*4 以後相等

### 2-4-2. Calculate memory requirements for storing the activations

#### Code
請參考 hw2-4-2.py (in analyzer)
```

#### Execution Result
```
Input Size: 
[1, 3, 224, 224]
Activations = 4392864 bytes```



### 2-4-3. Calculate computation requirements
#### Code
請參考 hw2-4-3.py (in analyzer)

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



