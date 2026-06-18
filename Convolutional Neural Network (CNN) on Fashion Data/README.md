
# Fashion-MNIST Image Classification using Convolutional Neural Networks (CNN) 👕👟👜

This repository contains a **Convolutional Neural Network (CNN)** developed using **TensorFlow/Keras** for classifying fashion products from the Fashion-MNIST dataset. The model learns visual patterns from grayscale clothing images and accurately predicts one of ten fashion categories.

---

## 🔥 Project Overview

Fashion-MNIST is a benchmark computer vision dataset consisting of 70,000 grayscale images belonging to 10 fashion categories. This project demonstrates the complete deep learning workflow including:

* Data loading and preprocessing
* Image normalization and reshaping
* CNN model design and training
* Performance evaluation
* Confusion matrix generation
* Classification report analysis
* Prediction visualization

The trained CNN achieved approximately **90% test accuracy**, demonstrating strong performance on unseen fashion images.

---

## 📂 Dataset Information

**Dataset:** Fashion-MNIST

* Total Images: 70,000
* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Image Type: Grayscale
* Number of Classes: 10

### Fashion Categories

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

## 🏗️ CNN Architecture

The model consists of multiple convolutional and dense layers designed for feature extraction and classification.

### Architecture Summary

Input Layer (28 × 28 × 1)

⬇️

Conv2D (32 Filters, 3×3, ReLU)

⬇️

MaxPooling2D (2×2)

⬇️

Conv2D (64 Filters, 3×3, ReLU)

⬇️

MaxPooling2D (2×2)

⬇️

Dropout (25%)

⬇️

Flatten Layer

⬇️

Dense Layer (128 Neurons, ReLU)

⬇️

Dropout (50%)

⬇️

Output Layer (10 Neurons, Softmax)

---

## ⚙️ Model Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Framework         | TensorFlow / Keras              |
| Optimizer         | Adam                            |
| Loss Function     | Sparse Categorical Crossentropy |
| Metric            | Accuracy                        |
| Epochs            | 10                              |
| Batch Size        | 64                              |
| Validation Split  | 20%                             |
| Output Activation | Softmax                         |

---

## ✅ Features

* Deep CNN architecture for image classification
* Automatic feature extraction using convolutional layers
* MaxPooling for dimensionality reduction
* Dropout regularization to reduce overfitting
* Softmax-based multi-class classification
* Confusion matrix visualization
* Classification report generation
* Prediction analysis on unseen test images
* TensorFlow/Keras implementation
* Beginner-friendly and research-ready

---

## 📊 Model Performance

### Training Results

* Training Accuracy: ~92–94%
* Validation Accuracy: ~89–91%

### Test Results

* Test Accuracy: ~90%
* Test Loss: Low and stable

The model demonstrates strong generalization performance and accurately classifies most fashion categories.

---

## 📈 Evaluation Metrics

The model was evaluated using:

* Test Accuracy
* Test Loss
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

These metrics provide a comprehensive understanding of model performance across all fashion categories.

---

## 🖼️ Visualization

The project includes multiple visualizations:

### Sample Dataset Images

Visualization of Fashion-MNIST samples with corresponding labels.

### Training Curves

* Training Loss vs Epochs
* Validation Loss vs Epochs
* Training Accuracy vs Epochs
* Validation Accuracy vs Epochs

### Confusion Matrix

Heatmap representation of classification performance across all classes.

### Predictions

Display of test images with:

* Predicted Label
* True Label

---

## 🚀 Applications

This project can be used for:

1. Fashion product recognition
2. Clothing classification systems
3. Retail AI applications
4. Inventory automation
5. E-commerce recommendation systems
6. Computer vision learning
7. Deep learning experimentation
8. CNN architecture studies
9. Academic research projects
10. AI portfolio development

---

## 💼 Use Cases

Perfect for:

1. Machine Learning Students
2. Deep Learning Beginners
3. Computer Vision Projects
4. Academic Assignments
5. Research Experiments
6. TensorFlow/Keras Learning
7. CNN Architecture Practice
8. AI Portfolio Projects
9. Interview Preparation
10. Classification Model Benchmarking

---

## 📚 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook / Google Colab

---

## ⚡ Why This Project

* Demonstrates end-to-end image classification workflow
* Uses industry-standard TensorFlow/Keras framework
* Implements modern CNN architecture techniques
* Includes complete evaluation and visualization pipeline
* Excellent project for learning computer vision fundamentals
* Strong baseline for advanced image classification research

---

## 🔗 References

* TensorFlow Documentation
* Keras Documentation
* Fashion-MNIST Dataset
* Deep Learning with Python
* Computer Vision Best Practices

---

## 🙌 Author

**Zohaib Sattar**

📧 Email: [zabizubi86@gmail.com](mailto:zabizubi86@gmail.com)

🔗 LinkedIn: https://www.linkedin.com/in/zohaib-sattar

🔗 GitHub: https://github.com/

---

## ⭐ Support the Project

If you found this project useful, please consider giving it a ⭐ on GitHub.

Your support encourages more open-source AI and Machine Learning projects.
