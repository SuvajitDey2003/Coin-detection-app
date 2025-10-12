# app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ----------------------------------------
# Streamlit Config
# ----------------------------------------
st.set_page_config(page_title="Indian Coin Detector", page_icon="💰", layout="wide")
st.title("💰 Indian Coin Detection & Value Calculator")
st.write("This Streamlit app reproduces your cv2_imshow output exactly — no color conversion done.")

# ----------------------------------------
# Load Model
# ----------------------------------------
@st.cache_resource
def load_model():
    return YOLO("my_model.pt")

model = load_model()

# ----------------------------------------
# Your original functions (unchanged)
# ----------------------------------------
def write_label_bounding_box(img, class_id, x1, y1, x2, y2, score, result):
    score_str = 'Score: {:.2f}'.format(score)
    class_name = result.names[int(class_id)].replace("₹", "")
    text = class_name + ' ' + score_str

    if class_id == 0:
        color = (255, 128, 0)
    elif class_id == 1:
        color = (0, 165, 255)
    elif class_id == 2:
        color = (147, 20, 255)
    elif class_id == 3:
        color = (255, 0, 255)
    else:
        color = (0, 0, 0)  # Default color

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 20)
    cv2.putText(img, text, (int(x1), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 20, cv2.LINE_AA)

    return img


def prediction(img, model):
    results = model(img)
    result = results[0]
    threshold = 65

    output = {
        '1_Rupee_Coin': 0,
        '2_Rupee_Coin': 0,
        '5_Rupee_Coin': 0,
        '10_Rupee_Coin': 0
    }

    for i in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = i
        if score >= threshold / 100:
            pred_class = result.names[class_id]
            output[pred_class] += 1
            img = write_label_bounding_box(img, class_id, x1, y1, x2, y2, score, result)

    total = (output['1_Rupee_Coin']) + (2 * output['2_Rupee_Coin']) + (5 * output['5_Rupee_Coin']) + (10 * output['10_Rupee_Coin'])

    text = f"Total = {total}"
    color = (0, 255, 0)
    cv2.putText(img, text, (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 30, cv2.LINE_AA)

    return img, total, output


# ----------------------------------------
# Streamlit Workflow
# ----------------------------------------
uploaded_file = st.file_uploader("📸 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as np.array (OpenCV style)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Display uploaded image (raw, no conversion)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    with st.spinner("Running prediction..."):
        output_img, total, counts = prediction(img.copy(), model)

    # Resize for display like cv2_imshow
    scale_percent = 10
    width = int(output_img.shape[1] * scale_percent / 100)
    height = int(output_img.shape[0] * scale_percent / 100)
    resized = cv2.resize(output_img, (width, height), interpolation=cv2.INTER_AREA)

    # Show output directly (no RGB conversion)
    st.image(resized, caption=f"Detected Coins — Total = ₹{total}", use_column_width=True)

    # Show counts
    st.subheader("🪙 Coin Count Summary")
    st.write(f"1 Rupee Coins: **{counts['1_Rupee_Coin']}**")
    st.write(f"2 Rupee Coins: **{counts['2_Rupee_Coin']}**")
    st.write(f"5 Rupee Coins: **{counts['5_Rupee_Coin']}**")
    st.write(f"10 Rupee Coins: **{counts['10_Rupee_Coin']}**")
    st.success(f"💰 **Total Value: ₹{total}**")
else:
    st.info("👆 Upload an image to start detection.")
