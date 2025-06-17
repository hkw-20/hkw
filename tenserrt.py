import tensorrt as trt
import os
import argparse

def convert_onnx_to_engine(onnx_path, engine_path, max_workspace_size=4 << 30, fp16_mode=False):
    """
    将 ONNX 模型转换为 TensorRT Engine
    
    参数:
    onnx_path (str): ONNX 模型路径
    engine_path (str): 输出的 TensorRT Engine 路径
    max_workspace_size (int): 最大工作空间大小，默认为 4GB
    fp16_mode (bool): 是否使用 FP16 精度，默认为 False
    """
    # 创建 TensorRT 记录器
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建 TensorRT 构建器
    builder = trt.Builder(logger)
    
    # 创建网络定义
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, logger)
    
    # 读取 ONNX 文件
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("解析 ONNX 模型失败")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # 配置构建器
    config = builder.create_builder_config()
    
    # 设置工作空间大小 - 兼容 TensorRT 8+
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    # 设置精度模式
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("使用 FP16 精度")
    
    # 构建引擎
    print("正在构建 TensorRT Engine... 这可能需要一些时间")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("构建引擎失败")
        return False
    
    # 保存引擎到文件
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"成功将 ONNX 模型转换为 TensorRT Engine 并保存到 {engine_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 ONNX 模型转换为 TensorRT Engine')
    parser.add_argument('--onnx', type=str, default='/root/ultralytics-8.3.27/best.onnx', help='ONNX 模型路径')
    parser.add_argument('--engine', type=str, default='best.engine', help='输出的 TensorRT Engine 路径')
    parser.add_argument('--fp16', action='store_true', help='使用 FP16 精度')
    parser.add_argument('--workspace', type=int, default=4, help='最大工作空间大小 (GB)')
    
    args = parser.parse_args()
    
    # 检查 ONNX 文件是否存在
    if not os.path.exists(args.onnx):
        print(f"错误: ONNX 文件 {args.onnx} 不存在")
    else:
        # 转换工作空间大小为字节
        max_workspace = args.workspace << 30  # GB 转字节
        convert_onnx_to_engine(args.onnx, args.engine, max_workspace, args.fp16)    