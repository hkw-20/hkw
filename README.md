# 这是一个改进yolov8引入cbam的代码
## 代码的使用
### 环境的安装
pip install requirements.txt
这可以安装所有的依赖库可能有些库下不了请联系15057840059@163.com我会帮你解决
### 训练模型
 首先训练没有引入cbam模块的模型
 导入自己的数据集修改自己的yaml路径让其能够训练自己的数据集建议将本文件里面的代码全部放入Ultralytics 8.3.27文件夹中方便使用
 #### 1
 conda actiavte your_env  #切换conda环境
 #### 2
 cd /home/robot/your_space/yolov8_cbam
 #### 3
 python train_v8.py
 ### 训练cbam
 训练引入cbam模块的代码
 #### 1
 conda actiavte your_env 
 #### 2
 cd /home/robot/your_space/yolov8_cbam
 #### 3
python train_cbam.py
### 模型的导出
#### 1
conda actiavte your_env 
#### 2
python3 onnx.py #导出onnx模型
#### 3
python3 tenserrt.py #导出engine


