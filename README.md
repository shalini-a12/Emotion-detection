# 🎭 Emotion Detection using Deep Learning

This project detects **human emotions in real-time** using a **Convolutional Neural Network (CNN)** model built with **TensorFlow/Keras**, combined with **OpenCV** for face detection and **Streamlit** for an interactive web interface.  

---

## 📂 Project Files
- `realtimedetection.py` → Main Streamlit app (runs the webcam + prediction)  
- `emotiondetector.json` → Model architecture  
- `emotiondetector.h5` → Trained model weights  
- `requirements.txt` → List of required Python packages  
- `.gitignore` → Ignores virtual environment and cache files  

---

## 🚀 Features
- Real-time face detection with **OpenCV**  
- Emotion classification into **7 categories**:  
  - Angry 😡  
  - Disgust 🤢  
  - Fear 😨  
  - Happy 😀  
  - Neutral 😐  
  - Sad 😢  
  - Surprise 😲  
- Runs as a **Streamlit web app**  
- Simple interface – works directly in your browser  

---

## 🛠️ Installation
Clone this repository and install the dependencies:

```bash
git clone https://github.com/shalini-a12/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
