# 🎙️ Speech Emotion Recognition using LSTM (RAVDESS Dataset)

## 📌 Project Overview
This project focuses on **Speech Emotion Recognition (SER)** using deep learning techniques.  
The system analyzes human speech audio and predicts the underlying emotion such as **happy, sad, angry, fear**, etc.

We use the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and apply **MFCC feature extraction** combined with an **LSTM neural network** to capture temporal patterns in speech signals.

---

## 🎯 Objectives
- Extract meaningful audio features from raw speech signals  
- Learn temporal dependencies using LSTM  
- Classify multiple human emotions from speech  
- Predict emotions from unseen audio samples  

---

## 🧠 Emotions Recognized
The model is trained to recognize the following emotions:

- Neutral  
- Calm  
- Happy  
- Sad  
- Angry  
- Fearful  
- Disgust  
- Surprised  

---

## 📂 Dataset
**RAVDESS – Emotional Speech Audio Dataset**

- Source: Kaggle  
- Kaggle Dataset ID: `uwrfkaggler/ravdess-emotional-speech-audio`  
- Audio Format: `.wav`  
- Sampling Rate: 48kHz  

Emotion labels are extracted directly from the filename structure provided by the dataset.

---

## 🛠️ Technologies Used
- Python  
- Librosa (Audio Processing)  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- KaggleHub  

---

## 🔍 Feature Extraction
- **MFCC (Mel-Frequency Cepstral Coefficients)**
- 40 MFCC features extracted from each audio sample
- Variable-length sequences padded for LSTM input

---

## 🧩 Model Architecture

Input (MFCC Sequences)
↓
LSTM (128 units, return_sequences=True)
↓
Dropout (0.3)
↓
LSTM (64 units)
↓
Dropout (0.3)
↓
Dense (64, ReLU)
↓
Dense (Softmax Output)


- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  

---

## 📊 Training Details
- Train/Test Split: 80% / 20%  
- Validation Split: 20%  
- Batch Size: 32  
- Epochs: 40  
- Early Stopping used to prevent overfitting  

---

## ✅ Results
- Model successfully learns emotional patterns in speech  
- Achieves meaningful accuracy on test data  
- Predicts emotions for unseen audio samples  

---

## 🔮 Sample Output
```text
True: happy  --> Predicted: happy
True: angry --> Predicted: angry
True: sad   --> Predicted: sad

```
How to Run the Project

Open a Kaggle Notebook

Paste the complete code into a single cell

Run the notebook

Dataset will download automatically using KaggleHub

Model will train and evaluate automatically

📌 Future Enhancements

CNN + LSTM hybrid architecture

Confusion matrix visualization

Model saving and deployment

Real-time emotion recognition

Multimodal emotion recognition (audio + face)



## 🙌 Author

**Zohaib Sattar**  
📧 Email: [zabizubi86@gmail.com](mailto:zabizubi86@gmail.com)
🔗 LinkedIn: [Zohaib Sattar](https://www.linkedin.com/in/zohaib-sattar)

---

## ⭐️ Support the Project

If you find this project helpful, please ⭐️ star the repo and share it with your network. It motivates further open-source contributions!


