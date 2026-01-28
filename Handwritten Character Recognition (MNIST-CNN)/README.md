# Handwritten Character Recognition using CNN

This project demonstrates **handwritten character recognition** using **Convolutional Neural Networks (CNNs)**. It is designed to classify handwritten digits (0-9) and can be extended to letters and full words using sequence modeling.

The project was completed as part of my **CodeAlpha Machine Learning Internship**.

---

## 🔹 Objective

The main goal of this project is to build a **deep learning model** that can accurately identify handwritten digits or characters from images. Applications include:

- Digit recognition for forms and exams
- Optical Character Recognition (OCR)
- Automated data entry
- Handwriting analysis

---

## 🔹 Dataset

This project uses popular datasets for handwritten characters:

| Dataset | Type | Number of Classes | Notes |
|---------|------|-----------------|-------|
| MNIST   | Digits | 10 | Handwritten digits 0-9 |
| EMNIST  | Characters | 26 (letters) | Handwritten uppercase/lowercase letters |

> MNIST is used for demonstration purposes, but EMNIST can be used for extended character recognition.

---

## 🔹 Approach

The workflow for this project is as follows:

1. **Data Preprocessing**
   - Normalize image pixel values (0-255 → 0-1)
   - Reshape images for CNN input `(28,28,1)`
   - One-hot encode labels for multi-class classification

2. **Model Architecture**
   - **Convolutional Layers (Conv2D):** Extract features from images
   - **Pooling Layers (MaxPooling2D):** Reduce dimensionality
   - **Dropout Layers:** Prevent overfitting
   - **Dense Layers:** Perform final classification
   - **Activation Functions:** ReLU for hidden layers, Softmax for output layer

3. **Training**
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Epochs: 10-20 (depending on dataset size)
   - Batch Size: 64

4. **Evaluation**
   - Test accuracy
   - Sample predictions visualization

---

## 🔹 Tech Stack

- **Programming Language:** Python 3.x
- **Libraries & Frameworks:**
  - `TensorFlow` / `Keras` for deep learning
  - `NumPy` for numerical operations
  - `Matplotlib` for data visualization

---

## 🔹 Code Example

```python
# Import Libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Visualize Predictions
preds = model.predict(X_test[:10])
for i, pred in enumerate(preds):
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {pred.argmax()}, True: {y_test[i].argmax()}")
    plt.show()


```
🔹 Results

Test Accuracy: ~99% on MNIST digits

Sample Predictions: The model correctly identifies handwritten digits in the test set.

The model can be extended to EMNIST for full alphabet recognition.

Handwritten-Character-Recognition/
│
├── data/                 # MNIST/EMNIST datasets
├── notebooks/            # Jupyter notebooks for experiments
├── scripts/              # Python scripts for training and evaluation
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

🔹 Future Work

Extend to full word recognition using CRNN (Convolutional Recurrent Neural Network)

Deploy as a web app for real-time handwriting recognition

Experiment with data augmentation to improve robustness

🔹 References

MNIST Dataset

EMNIST Dataset

Keras CNN Documentation


---

🔹 Author

Zohaib Sattar – Machine Learning Enthusiast

This README is **professional, clear, and fully explanatory** for anyone viewing your repo. It shows:  
- Objective, approach, and dataset  
- Code example with results  
- Folder structure and future extensions  

---

If you want, I can also **create a matching README for the Emotion Recognition task** in the **same style**, so your GitHub repo looks consistent and polished.  

Do you want me to do that too?

## 🙌 Author

**Zohaib Sattar**  
📧 Email: [zabizubi86@gmail.com](mailto:zabizubi86@gmail.com)
🔗 LinkedIn: [Zohaib Sattar](https://www.linkedin.com/in/zohaib-sattar)

---

## ⭐️ Support the Project

If you find this project helpful, please ⭐️ star the repo and share it with your network. It motivates further open-source contributions!
