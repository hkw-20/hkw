import os
import sys
import torch
from ultralytics import YOLO

# 获取当前目录并添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 1. 首先导入CBAM模块
from cbam import CBAM

# 2. 创建模型实例
def create_model():
    # 确保YOLO能够找到CBAM类
    # 将CBAM添加到全局作用域
    globals()["CBAM"] = CBAM
    
    # 创建模型实例
    model = YOLO('yolov8n_cbam.yaml')
    return model

def main():
    print("="*50)
    print("创建YOLO模型")
    print("="*50)
    
    # 创建模型
    model = create_model()
    
    # 打印模型结构验证
    print("\n" + "="*50)
    print("验证模型结构")
    print("="*50)
    model.info(verbose=True)
    
    # 测试前向传播
    print("\n" + "="*50)
    print("测试前向传播")
    print("="*50)
    test_input = torch.randn(1, 3, 640, 640)
    try:
        output = model.model(test_input)
        print("✅ 前向传播成功!")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 开始训练
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)
    results = model.train(
        data='hkw.yaml',
        epochs=40,
        imgsz=640,
        workers=8,
        batch=4,
        name='yolov8_cbam',
        patience=50,
        device='0' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == '__main__':
    main()