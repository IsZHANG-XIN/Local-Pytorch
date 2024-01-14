import onnxruntime
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch
import pandas as pd

# 创建 ONNX Runtime 会话
ort_session = onnxruntime.InferenceSession('resnet18_imagenet.onnx')

img_path = 'ImageNet上训练的resnet18模型/1.png'  # 输入图片路径

# 用 Pillow 载入图像
img_pil = Image.open(img_path).convert('RGB')  # 转换为RGB格式

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

input_img = test_transform(img_pil)  # 将图像送入预处理函数中

# 添加额外的维度，并转为numpy array
input_tensor = input_img.unsqueeze(0).numpy()

# 打印处理后的图像
print(input_tensor)

# ONNX Runtime 输入
ort_inputs = {'input': input_tensor}

# 运行 ONNX Runtime
pred_logits = ort_session.run(['output'], ort_inputs)[0]
print(pred_logits)

# 将 pred_logits_tensor 转换为 PyTorch 张量
pred_logits_tensor = torch.from_numpy(pred_logits)

# 对 logit 分数做 softmax 运算，得到置信度概率
pred_softmax = F.softmax(pred_logits_tensor, dim=1)

# 取置信度最高的前 n 个结果
n = 3
top_n = torch.topk(pred_softmax, n)
# 预测类别
pred_ids = top_n.indices.numpy()[0]
# 预测置信度
confs = top_n.values.numpy()[0]

# 载入类别 ID 和 类别名称 对应关系
df = pd.read_csv('ImageNet上训练的resnet18模型\imagenet_class_index.csv')   
idx_to_labels = {}
for idx, row in df.iterrows():
    #idx_to_labels[row['ID']] = row['class']   # 英文
    idx_to_labels[row['ID']] = row['Chinese'] # 中文

# 分别用英文和中文打印预测结果
# 分别用英文和中文打印预测结果
for i in range(n):
    try:
        class_name = idx_to_labels[int(pred_ids[i])]
    except KeyError:
        class_name = f"Class {pred_ids[i]} not found in the dictionary"
    confidence = confs[i] * 100             # 获取置信度
    text = '{:<20} {:>.3f}'.format(class_name, confidence)
    print(text)
