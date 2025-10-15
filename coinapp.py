import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="💰 Indian Coin Detector", layout="wide")
st.title("💰 Indian Coin Detection & Change Calculator")
st.markdown("Detect Indian coins and calculate the **total Rs. value** using YOLOv11!")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

model_path = st.text_input("Enter YOLO model path:", "my_model.pt")

if model_path and model_path.endswith(".pt"):
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚠️ Please enter a valid YOLOv11 `.pt` model file path.")

# Coin value mapping
coin_values = {
    '1_Rupee_Coin': 1,
    '2_Rupee_Coin': 2,
    '5_Rupee_Coin': 5,
    '10_Rupee_Coin': 10
}

# Color mapping for each coin type (BGR)
class_colors = {
    '1_Rupee_Coin': (255, 128, 0),
    '2_Rupee_Coin': (0, 165, 255),
    '5_Rupee_Coin': (147, 20, 255),
    '10_Rupee_Coin': (255, 0, 255)
}

# ---------------------- SIDEBAR SETTINGS ----------------------
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)
source_option = st.sidebar.radio("Select Input Source:", ["Upload Image", "Upload Video", "Use Webcam"])

# ---------------------- IMAGE UPLOAD ----------------------
if source_option == "Upload Image":
    uploaded_img = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None and model_path.endswith(".pt"):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_img.read())
            img_path = temp_file.name

        frame = cv2.imread(img_path)
        results = model(frame)
        detections = results[0].boxes

        coin_counts = {k: 0 for k in coin_values}

        for det in detections:
            conf = det.conf.item()
            if conf >= confidence_threshold:
                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(det.cls.item())
                classname = model.names[classidx]

                color = class_colors.get(classname, (0, 255, 0))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                cv2.putText(frame, f"{classname} ({conf*100:.1f}%)", 
                            (xmin, max(ymin - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if classname in coin_values:
                    coin_counts[classname] += 1

        total_value = sum(coin_counts[c] * v for c, v in coin_values.items())

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Total: Rs.{total_value}", use_container_width=True)
        st.write("### 💰 Coin Summary:")
        st.json(coin_counts)
        st.success(f"**Total Value: Rs.{total_value}**")

# ---------------------- VIDEO UPLOAD ----------------------
elif source_option == "Upload Video":
    uploaded_vid = st.file_uploader("🎥 Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid is not None and model_path.endswith(".pt"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results[0].boxes
            coin_counts = {k: 0 for k in coin_values}

            for det in detections:
                conf = det.conf.item()
                if conf >= confidence_threshold:
                    xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    classidx = int(det.cls.item())
                    classname = model.names[classidx]

                    color = class_colors.get(classname, (0, 255, 0))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                    cv2.putText(frame, f"{classname} ({conf*100:.1f}%)", 
                                (xmin, max(ymin - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if classname in coin_values:
                        coin_counts[classname] += 1

            total_value = sum(coin_counts[c] * v for c, v in coin_values.items())
            cv2.putText(frame, f'Total: Rs.{total_value}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {fps:.1f}', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()
        st.success("✅ Video processing complete!")

# ---------------------- WEBCAM STREAM ----------------------
elif source_option == "Use Webcam":
    st.info("🎥 Select a camera and press Start. You can switch cameras live!")

    # --- Detect available cameras ---
    def detect_cameras(max_cameras=5):
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    if 'available_cams' not in st.session_state:
        st.session_state.available_cams = detect_cameras()
    if 'selected_cam' not in st.session_state:
        st.session_state.selected_cam = st.session_state.available_cams[0] if st.session_state.available_cams else None
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    # Rescan cameras
    if st.button("🔄 Rescan Cameras"):
        st.session_state.available_cams = detect_cameras()
        st.success(f"Detected cameras: {st.session_state.available_cams}")
        if st.session_state.selected_cam not in st.session_state.available_cams:
            st.session_state.selected_cam = st.session_state.available_cams[0] if st.session_state.available_cams else None

    available_cams = st.session_state.available_cams

    if not available_cams:
        st.warning("⚠️ No camera detected. Please connect a webcam or mobile camera.")
    else:
        selected_cam = st.selectbox("Select Camera", available_cams, index=available_cams.index(st.session_state.selected_cam))
        st.session_state.selected_cam = selected_cam

        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])

        if st.session_state.cap is None or st.session_state.cap_idx != selected_cam:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
            st.session_state.cap = cv2.VideoCapture(selected_cam)
            st.session_state.cap_idx = selected_cam
            st.session_state.prev_time = 0

        cap = st.session_state.cap
        prev_time = st.session_state.prev_time

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access selected webcam.")
                break

            results = model(frame)
            detections = results[0].boxes
            coin_counts = {k: 0 for k in coin_values}

            for det in detections:
                conf = det.conf.item()
                if conf >= confidence_threshold:
                    xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    classidx = int(det.cls.item())
                    classname = model.names[classidx]

                    color = class_colors.get(classname, (0, 255, 0))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                    cv2.putText(frame, f"{classname} ({conf*100:.1f}%)",
                                (xmin, max(ymin - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if classname in coin_values:
                        coin_counts[classname] += 1

            total_value = sum(coin_counts[c] * v for c, v in coin_values.items())
            cv2.putText(frame, f'Total: Rs.{total_value}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            st.session_state.prev_time = prev_time
            cv2.putText(frame, f'FPS: {fps:.1f}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.success("🛑 Webcam stopped.")

