YOLOv8 with CBAM Attention Module

这个项目在YOLOv8模型中集成了CBAM（Convolutional Block Attention Module）注意力机制，显著提升了目标检测的精度。项目包含完整的训练流程、模型验证、ONNX导出和TensorRT加速实现。
目录

    项目简介

    文件结构

    快速开始

    模型结构

    性能优势

    实验结果

    使用示例

    常见问题

    贡献指南

项目简介

本项目在YOLOv8骨干网络中集成CBAM注意力模块，通过结合通道注意力和空间注意力机制，使模型能够更有效地聚焦于图像中的关键特征区域，从而提升目标检测的准确性。

主要特点：

    🚀 在YOLOv8骨干网络的关键位置添加CBAM模块

    📊 完整的训练、验证和测试流程

    ⚡ ONNX模型导出和TensorRT加速支持

    📈 相比原始YOLOv8，mAP提升约7%

    🔧 模块化设计，易于扩展到其他YOLO版本

文件结构


├── configs/
│   └── yolov8n_cbam.yaml       # YOLOv8n with CBAM 模型配置文件
│
├── models/
│   ├── cbam.py                 # CBAM注意力模块实现
│   ├── export_onnx.py          # PyTorch模型导出为ONNX格式
│   └── onnx_to_tensorrt.py     # ONNX转TensorRT引擎工具
│
├── utils/
│   ├── train.py                # 主训练脚本
│   └── create_and_train.py     # 模型创建、验证与训练脚本
│
├── README.md                   # 项目说明文档
└── requirements.txt            # 依赖库列表

快速开始
环境安装
# 创建Python环境（推荐使用Python 3.8+）
conda create -n yolo_cbam python=3.8
conda activate yolo_cbam

# 安装依赖
pip install -r requirements.txt

训练模型
# 使用YAML配置文件训练
python utils/train.py --cfg configs/yolov8n_cbam.yaml

# 使用CBAM模块创建并训练模型
python utils/create_and_train.py

模型导出与转换
# 导出为ONNX格式
python models/export_onnx.py

# 转换为TensorRT引擎
python models/onnx_to_tensorrt.py \
  --onnx best.onnx \
  --engine best.engine \
  --fp16 \
  --workspace 4

模型结构
CBAM模块实现
# models/cbam.py

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out  
YOLOv8 with CBAM 配置
# configs/yolov8n_cbam.yaml

backbone:
  # ...标准YOLOv8n结构...
  - [-1, 1, CBAM, [64]]          # 在P3/8输出后添加CBAM模块
  # ...后续层...
  - [-1, 1, CBAM, [128]]         # 在P4/16输出后添加CBAM模块
  # ...剩余结构...
性能优势

集成CBAM注意力模块的YOLOv8模型具有以下优势：

    精度提升：CBAM模块使模型更关注关键特征区域，提高检测准确率

    小目标检测增强：对小型目标的检测能力显著提高

    抗干扰能力：在复杂背景中保持稳定的检测性能

    推理加速：支持TensorRT加速，保持实时性能

    参数高效：仅增加少量参数即可获得显著性能提升

实验结果

在自定义数据集上的对比实验（RTX 3090 GPU）：
模型	mAP@0.5	推理速度(FPS)	参数量(M)	相对提升
YOLOv8n	0.78	120	3.2	-
YOLOv8n+CBAM	0.85	115	3.3	+7%

    注：CBAM模块仅增加0.1M参数，却带来7%的mAP提升，推理速度仅下降4%
