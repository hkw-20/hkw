from ultralytics import YOLO

# 加载训练好的YOLOv8模型
model = YOLO('/root/ultralytics-8.3.27/best.pt')  # 替换为你的模型路径

# 导出模型为ONNX格式
model.export(
    format='onnx',          # 导出格式
    dynamic=False,          # 固定批处理维度 (True允许动态batch)
    simplify=True,          # 简化ONNX模型
    opset=12,               # ONNX算子集版本
    imgsz=[640, 640]        # 输入图像尺寸 [高度, 宽度]
)

print("导出成功！ONNX模型已保存为 'best.onnx'")