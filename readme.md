Implementation of YOLOv8 Improvement with CBAM Integration
Code Usage
Environment Installation
bash
pip install -r requirements.txt  
This installs all required dependencies. If any libraries fail to install, contact 15057840059@163.com for assistance.

Model Training
First train the baseline model (without CBAM module)
Import your custom dataset and modify the YAML configuration path.
Recommendation: Place all code files in the Ultralytics 8.3.27 directory for optimal compatibility.

bash
conda activate your_env  # Switch conda environment  
cd /home/robot/your_space/yolov8_cbam  
python train_v8.py  
Training CBAM-Enhanced Model
bash
conda activate your_env  
cd /home/robot/your_space/yolov8_cbam  
python train_cbam.py  
Model Export
bash
conda activate your_env  
python3 onnx.py       # Export ONNX model  
python3 tensorrt.py   # Export TensorRT engine  
Expected Training Results
Model	mAP@0.5	mAP@0.5:0.95	Params (M)	GFLOPs	Inference (ms)	FPS
YOLOv8n	0.856	0.678	3.1	8.7	6.5	154
YOLOv8n-CBAM	0.872	0.695	3.3	9.1	7.0	143
Performance Summary
+1.6% improvement in mAP@0.5

+1.7% improvement in mAP@0.5:0.95

Parameter count increased by only 6.5%

Inference time increased by 7.7%

More significant improvements observed in complex scenarios (occlusion, small objects)
