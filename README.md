# Improved YOLOv8 Code with CBAM Integration
## Code Usage
### Environment Installation
pip install requirements.txt
If any libraries fail to install, contact 15057840059@163.com for assistance.

### Model Training
First train the baseline model without CBAM module:  
Import your dataset and modify the YAML configuration path to train on your custom dataset.  
Recommend placing all code in the `Ultralytics 8.3.27` folder for compatibility.

conda activate your_env  # Switch conda environment
cd /home/robot/your_space/yolov8_cbam
python train_v8.py

### Training with CBAM
Train the improved model with CBAM module:
conda activate your_env 
cd /home/robot/your_space/yolov8_cbam
python train_cbam.py

### Model Export
conda activate your_env 
python3 onnx.py  # Export ONNX model
python3 tenserrt.py  # Export TensorRT engine

### Expected Training Results

Model           | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs | Inference (ms) | FPS
--------------- | ------- | ------------ | ---------- | ------ | -------------- | ---
YOLOv8n         | 0.856   | 0.678        | 3.1        | 8.7    | 6.5            | 154
YOLOv8n-CBAM    | 0.872   | 0.695        | 3.3        | 9.1    | 7.0            | 143

### Performance Summary
mAP@0.5 improved by 1.6%, mAP@0.5:0.95 improved by 1.7%
Parameters increased by only 6.5%, inference time increased by 7.7%
More significant improvements in complex scenarios (occlusion, small objects)
