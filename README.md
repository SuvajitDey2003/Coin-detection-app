# 💰 Coin Detection & Change Calculator App

A web-based application built with **Streamlit** and **YOLOv11** that detects Indian coins (₹1, ₹2, ₹5, ₹10) in real-time using your webcam and calculates the **total change amount** automatically.

---

## 🚀 Features

- 🧠 **YOLOv11-powered coin detection**
- 📷 Capture images directly from your webcam
- 💰 Automatically **count coins** and **calculate total change**
- 🌐 Deployable on **Render** or **Streamlit Cloud**
- ⚡ Lightweight & fast inference

---

## 🏗️ Tech Stack

- **Python 3.10+**
- **Streamlit**
- **OpenCV )**
- **Ultralytics YOLOv11**
- **NumPy**

---


## ⚙️ Installation (Local Setup)

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Coin_detection_App.git
   cd Coin_detection_App

2. **Create a virtual environment**
   ```bash
    python -m venv coin-env
    source coin-env/bin/activate      # Mac/Linux
    coin-env\Scripts\activate         # Windows

3. **Install dependencies**
   ```bash
    pip install -r requirements.txt

5. **Run the app locally**
   ```bash
   streamlit run coinapp.py

Then open http://localhost:8501 in your browser.

---

## 🧾 Sample requirements.txt

- **numpy>=1.26.0**
- **opencv_contrib_python==4.10.0.84**
- **opencv_python==4.12.0.88**
- **streamlit==1.50.0**
- **ultralytics==8.3.214**

---

## 📸 How It Works

1. Load the trained YOLOv11 model (my_model.pt).

2. Capture an image from webcam or upload a file.

3. YOLO detects coins and returns bounding boxes.

4. Draws rectangles with class labels and confidence scores.

5. Counts total coins and displays the rupee value.

---

## ☁️ Deployment Guide

✅ Deploy on Streamlit Cloud

1. Push your project (with coinapp.py, best.pt, and requirements.txt) to GitHub.

2. Go to Streamlit Cloud

3. Click “New App” → Select your GitHub repo.

4. Set the main file path to:
   ```bash
   Your_github_repo/coinapp.py

5. Deploy

---







