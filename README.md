```markdown
#  Plant Disease Classification Using MobileNetV2

This repository contains a deep learning model built with MobileNetV2 to classify
plant diseases from leaf images using transfer learning.
It includes a trained `.h5` model, an interactive Jupyter Notebook for testing and evaluation, and a Python script for making predictions on new images.


##  Features

- ✅ MobileNetV2 pretrained on ImageNet
- ✅ Fine-tuned on 15-class plant disease dataset
- ✅ Lightweight `.h5` model for fast inference
- ✅ Prediction script for custom images
- ✅ Visualizations and metrics in notebook

##  Model Details

- Architecture: MobileNetV2 (include_top=False)
- Layers Added:
  - GlobalAveragePooling2D
  - Dense (512, ReLU)
  - Dropout (0.3)
  - Dense (15, Softmax)
- Input Size: 224 × 224 × 3
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Accuracy: ~89% on validation data

##  Classes Supported

- Tomato (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy)
- Pepper bell (Bacterial spot, Healthy)
- Potato (Early blight, Late blight, Healthy)

Total: 15 classes



### 1. Install Dependencies

```bash
pip install tensorflow matplotlib numpy pillow
````

