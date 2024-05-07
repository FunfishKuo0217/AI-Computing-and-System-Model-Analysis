# from torchvision import models
from torchsummary import summary
import torchinfo
import torchvision.models as models

# 加載 GoogLeNet 模型
model = models.googlenet(pretrained=True)
input_shape = (3, 224, 224)
print(torchinfo.summary(model, input_shape, batch_dim = 0, col_names=("output_size", "num_params"), verbose=0))
