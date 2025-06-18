# 这是一个改进yolov8引入cbam的代码
## 代码的使用
### 环境的安装
pip install requirements.txt
这可以安装所有的依赖库可能有些库下不了请联系15057840059@163.com我会帮你解决
### 训练模型
 首先训练没有引入cbam模块的模型
 导入自己的数据集修改自己的yaml路径让其能够训练自己的数据集建议将本文件里面的代码全部放入Ultralytics 8.3.27文件夹中方便使用
 #### 
 conda actiavte your_env  #切换conda环境
 #### 
 cd /home/robot/your_space/yolov8_cbam
 #### 
 python train_v8.py
 ### 训练cbam
 训练引入cbam模块的代码
 #### 
 conda actiavte your_env 
 #### 
 cd /home/robot/your_space/yolov8_cbam
 #### 
python train_cbam.py
### 模型的导出
#### 
conda actiavte your_env 
#### 
python3 onnx.py #导出onnx模型
#### 
python3 tenserrt.py #导出engine
### 预期训练结果

#### 
模型	         mAP@0.5  	mAP@0.5:0.95    	参数量(M)   	GFLOPs	  推理时间(ms)    	FPS
#### 
YOLOv8n       	0.856	      0.678	          3.1	         8.7	       6.5	         154
####
YOLOv8n-CBAM  	0.872	       0.695	         3.3	         9.1	       7.0	         143
### 性能说明
#### 
mAP@0.5提升1.6%，mAP@0.5:0.95提升1.7%
####
参数量仅增加6.5%，推理时间增加7.7%
####
在复杂场景下（遮挡、小目标）提升效果更明显
