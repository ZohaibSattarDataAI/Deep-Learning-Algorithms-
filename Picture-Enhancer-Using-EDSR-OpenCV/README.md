# Picture Enhancer using EDSR and OpenCV

A **high-quality image enhancement project using deep learning**, designed to upscale and improve image resolution with **EDSR (Enhanced Deep Super-Resolution) models** via OpenCV. This repository is ideal for learners, researchers, and developers exploring AI-based image enhancement and super-resolution techniques.

---

## 🔥 Overview

Super-resolution deep learning models have revolutionized image processing by enabling **high-fidelity upscaling** from low-resolution inputs. This project leverages **EDSR**, a state-of-the-art CNN-based super-resolution model, to produce **4× enhanced images** while preserving fine details.

Key highlights:

- Deep learning-based image upscaling
- Pre-trained EDSR model for fast deployment
- OpenCV DNN integration for simplicity and performance
- Optional preview mode for rapid testing

---

## 📂 Contents

### 1. Project Structure


Picture-Enhancer/
├── models/ # Pre-trained EDSR models (.pb files)
│ └── EDSR_x4.pb
├── images/ # Input images
│ └── Data Science.jpg
├── output/ # Enhanced images
├── main.py # Python script to run enhancement
├── requirements.txt
└── README.md



### 2. Functionality

- **Upscale images** 4× using EDSR
- **Optional fast preview** for reduced-size testing
- **Save enhanced images** in `output/` folder
- **Cross-platform** Python implementation

---

## 🧠 Design Principles

- **Modular and clean code**: Easy to read, reuse, and extend  
- **Theory + Practice**: Implements a real-world super-resolution model  
- **Reproducibility**: Pre-trained model included, deterministic output  
- **Extensibility**: Add new models or image-processing pipelines easily

---

## 🛠️ Tech Stack

- **Programming Language**: Python 3.x  
- **Libraries**: OpenCV (contrib), NumPy  
- **Deep Learning Model**: EDSR (pre-trained `.pb`)  
- **Optional GPU Acceleration**: CUDA-supported OpenCV or PyTorch for faster processing  

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/Picture-Enhancer.git
cd Picture-Enhancer
pip install -r requirements.txt
python main.py
```

📈 Performance Notes

CPU-based upscaling can be slow (e.g., 512×640 → 2048×2560 may take ~30 minutes).

GPU acceleration is highly recommended for faster results.

Previews can help test images before committing to full upscale.

📂 Use Cases

This repository is suitable for:

Learning super-resolution and deep learning pipelines

Image enhancement in photography and design

AI-based preprocessing for computer vision projects

High-quality upscaling for archival or printing

Experimentation with neural network models for super-resolution

Rapid prototyping of AI-based image enhancement tools

Academic projects or research in image processing

## 🙌 Author

**Zohaib Sattar**  
📧 Email: [zabizubi86@gmail.com](mailto:zabizubi86@gmail.com)  
🔗 LinkedIn: [Zohaib Sattar](https://www.linkedin.com/in/zohaib-sattar)  

---

## ⭐️ Support the Project

If you find this project helpful, please ⭐️ star the repo and share it with your network. It motivates further open-source contributions!  
