# Week-1
Developing an AI-based image classification model that can identify whether waste items are recyclable or non-recyclable. This helps in automating waste segregation, supporting sustainable waste management and recycling efforts.
Dataset Used: TrashNet Dataset (Kaggle)
# Dataset
This dataset includes images of six waste categories:
Cardboard
Glass
Metal
Paper
Plastic
Trash
The dataset was uploaded and extracted inside Google Colab for use in model training.

# Model
Built a Convolutional Neural Network (CNN) using TensorFlow/Keras.
Input image size: 128 × 128 RGB.
Layers included convolution, pooling, flattening, and dense layers.
The final output layer uses softmax activation for six classes.
At this stage, the model architecture has been successfully built and verified.
Training and evaluation will be completed in the next phase.

# Tools & Technologies
Google Colab
Python
TensorFlow / Keras
NumPy, Matplotlib

# Model File
The trained model (waste_classifier.h5) was not uploaded because of GitHub’s 25 MB file size limit.
It will be stored locally and can be recreated later once training is completed.

# Project Theme

Sustainability × AI/ML — Promoting responsible waste segregation using artificial intelligence.

 # Week 2 Updates
- Enhanced CNN architecture with additional convolution blocks.
- Added Batch Normalization and Dropout layers.
- Implemented Data Augmentation to reduce overfitting.
- Added Early Stopping, Learning-Rate Scheduling, and Model Checkpointing.
- Generated confusion matrix and classification report.
- Observed validation accuracy around 17 % (model still under training).
- Plan to improve further in Week 3 using Transfer Learning.

# Waste Classification Using Deep Learning (ResNet50 – Fine-Tuned)
1. Project Overview
The goal of this project is to build an accurate image classification model for different categories of waste, supporting automated and sustainable waste management systems.
Initial attempts using basic CNNs and MobileNet architectures resulted in low accuracy. Therefore, the project progressed toward a more advanced approach using ResNet50 with transfer learning and fine-tuning, which significantly improved the results.
The final model achieves approximately 97% accuracy across 12 waste categories, after applying multiple improvements in data handling, model architecture, and training methods.

# 2. Dataset Used
Dataset Name: Garbage Classification Dataset (Kaggle)
The dataset contains images belonging to the following 12 classes:
battery
biological
brown-glass
green-glass
white-glass
cardboard
clothes
metal
paper
plastic
shoes
trash

The dataset includes more than 15,000 images.
A proper stratified split was created with:
80% training data
20% validation data
This corrected split is one of the major reasons for improved accuracy.

# 3. Key Improvements Implemented
3.1 Transfer Learning with ResNet50
Used a pretrained ResNet50 model trained on ImageNet.
Removed the original classification layers.
Added a custom classification head consisting of:
Global Average Pooling layer
Dense layer with 512 units and ReLU activation
Dropout (0.5)
Output layer with 12 units and softmax activation

3.2 Fine-Tuning
Initially trained only the added top layers.
Later unfroze deeper layers of ResNet50 for fine-tuning.
Used a very low learning rate (1e-5) to avoid damaging pretrained weights.


3.3 Data Augmentation
To reduce overfitting and improve generalization:

Rotation
Width and height shifting
Zoom
Horizontal flipping
Shear transformations


3.4 Correct Train–Validation Split
Previous low accuracy (~15–20%) was caused by an incorrect or imbalanced dataset split.
After reconstructing the dataset into class-balanced training and validation folders, the model performance improved dramatically.

3.5 Training Callbacks
EarlyStopping to avoid overfitting
ModelCheckpoint to save the best-performing model
ReduceLROnPlateau to adjust learning rate automatically

# 4. Final Results After All Improvements
MetricScoreOverall Accuracy97%Macro F1 Score0.96Weighted F1 Score0.97
Observations
Almost all classes achieved high precision and recall.
The model performs significantly better after fine-tuning.
The corrected dataset split had a major impact on overall performance.

# 5. Prediction Example
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = ['battery','biological','brown-glass','cardboard','clothes',
               'green-glass','metal','paper','plastic','shoes','trash','white-glass']

model = tf.keras.models.load_model("garbage_resnet50_finetuned.h5")

img = image.load_img("test_image.jpg", target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0) / 255.0

pred = model.predict(img)
print("Predicted class:", class_names[np.argmax(pred)])


# 6. Project Structure
project/
│
├── garbage_resnet50_finetuned.h5        # Final trained model
├── Waste_Classification_Final.ipynb     # Complete code notebook
├── README.md
├── prediction_example.py                 # Optional prediction script
└── requirements.txt                      # Python dependencies


# 7. Conclusion
This project demonstrates how advanced deep learning techniques such as transfer learning, fine-tuning, proper dataset preparation, and augmentation can drastically improve classification performance.
The model progressed from approximately 15% accuracy to 97% accuracy, proving the importance of both high-quality data handling and the use of robust pretrained architectures like ResNet50.

