# 导入相关包
import torch
from torchvision import models

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True)
model = model.eval().to(device)
print(model)

x = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():   #推理
    torch.onnx.export(
        model,                       # 要转换的模型
        x,                           # 模型的任意一组输入
        'resnet18_imagenet.onnx',    # 导出的 ONNX 文件名
        opset_version=11,            # ONNX 算子集版本
        input_names=['input'],       # 输入 Tensor 的名称（自己起名字）
        output_names=['output']      # 输出 Tensor 的名称（自己起名字）
    ) 
