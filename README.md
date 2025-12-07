# 🛣️ AI Road Lane Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Advanced Computer Vision for Autonomous Driving & Road Safety**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model) • [Author](#-author)

---

## 📖 Overview

AI Road Lane Detection is a real-time lane marking detection system powered by **YOLOv11** and **Streamlit**. It can detect and classify 14 different types of road lane markings from images, videos, or live webcam feed.

## ✨ Features

- 🖼️ **Image Detection** - Upload images and detect lane markings instantly
- 🎥 **Video Processing** - Process videos with frame-by-frame detection and analysis
- 📹 **Live Webcam** - Real-time lane detection from your webcam
- 📊 **Analytics Dashboard** - Comprehensive statistics and detection breakdown
- 🎨 **Beautiful UI** - Modern glassmorphism design with video background
- ⬇️ **Download Results** - Export processed images and videos
- ⚙️ **Configurable Settings** - Adjustable confidence threshold and visualization options

## 🏷️ Detected Classes

The model can detect **14 different lane marking types**:

| ID | Class | ID | Class |
|:--:|:------|:--:|:------|
| 0 | BUS LANE | 7 | SLOW |
| 1 | Yellow Markings | 8 | Left Arrow |
| 2 | Line 1 | 9 | Forward Arrow |
| 3 | Line 2 | 10 | Forward Arrow -Left |
| 4 | Line 3 | 11 | Forward Arrow -Right |
| 5 | Crossing | 12 | Right Arrow |
| 6 | Romb | 13 | Bicycle |

## 🎬 Demo

### Video Processing
- Upload any road video
- Watch real-time detection as each frame is processed
- View comprehensive analysis summary with per-class breakdown
- Download the processed video

### Image Detection
- Side-by-side comparison of original and detected image
- Detection statistics with class breakdown
- Download processed image

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SyedSarimAbbas/Road-Lane-Sign-Prediction.git
   cd Road-Lane-Sign-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

## 📦 Requirements

```txt
streamlit>=1.28.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

## 💻 Usage

### Image Upload
1. Navigate to the **📷 Image Upload** tab
2. Upload an image (JPG, JPEG, PNG, BMP)
3. View detection results with bounding boxes and overlays
4. Check detection statistics
5. Download the processed image

### Video Upload
1. Navigate to the **🎥 Video Upload** tab
2. Upload a video (MP4, AVI, MOV, MKV)
3. Click **🚀 Start Lane Detection**
4. Watch real-time processing preview
5. View analysis summary after completion
6. Download the processed video

### Live Webcam
1. Navigate to the **📹 Live Webcam** tab
2. Click **🎬 Start Webcam**
3. View real-time lane detection
4. Click **⏹️ Stop Webcam** to end

### Settings (Sidebar)
- **Confidence Threshold** - Adjust detection sensitivity (0.0 - 1.0)
- **Line Thickness** - Change bounding box thickness (1-5)
- **Show Labels** - Toggle class labels on/off

## 🤖 Model

This project uses a custom-trained **YOLOv11** model specifically designed for road lane marking detection.

- **Framework**: Ultralytics YOLOv11
- **Input**: RGB images/video frames
- **Output**: Bounding boxes with class labels and confidence scores
- **Model File**: `Road_Lane_Detector.pt`

## 📁 Project Structure

```
Road-Lane-Sign-Prediction/
├── main.py                         # Main Streamlit application
├── Road_Lane_Detector.pt           # YOLOv11 trained model
├── Road_Lane_Mark_Detection.ipynb  # Training notebook
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🎨 UI Features

- **Video Background** - Continuous HD road driving video
- **Glassmorphism Design** - Modern blur effects and transparency
- **Gradient Text** - Beautiful color gradients
- **Dark Theme** - Easy on the eyes
- **Responsive Layout** - Works on different screen sizes
- **Animated Elements** - Smooth transitions and hover effects

## 🛠️ Technologies Used

- **YOLOv11** - State-of-the-art object detection
- **Streamlit** - Web application framework
- **OpenCV** - Image and video processing
- **NumPy** - Numerical computations
- **Pillow** - Image handling

## 📈 Performance

- Real-time detection on modern hardware
- GPU acceleration supported (T4)
- Optimized video processing pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Syed Sarim Abbas**

Built for Autonomous Driving & Road Safety

---

**⭐ Star this repository if you found it helpful!**
