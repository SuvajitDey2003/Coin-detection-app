import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- Load model ---
@st.cache_resource
def load_model():
    model = YOLO("my_model.pt")  # change to your trained model path
    return model

model = load_model()

# --- Helper functions ---
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
        color = (0, 0, 0)

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 20)
    cv2.putText(img, text, (int(x1), int(y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 4, color, 20, cv2.LINE_AA)
    return img


def prediction(img, model):
    results = model(img)
    result = results[0]
    threshold = 65
    output = {'1_Rupee_Coin': 0, '2_Rupee_Coin': 0, '5_Rupee_Coin': 0, '10_Rupee_Coin': 0}

    for i in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = i
        if score >= threshold / 100:
            pred_class = result.names[class_id]
            output[pred_class] += 1
            img = write_label_bounding_box(img, class_id, x1, y1, x2, y2, score, result)

    total = (output['1_Rupee_Coin']) + (2 * output['2_Rupee_Coin']) + \
            (5 * output['5_Rupee_Coin']) + (10 * output['10_Rupee_Coin'])

    text = f"Total = Rupees {total}"
    color = (0, 255, 0)
    cv2.putText(img, text, (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 30, cv2.LINE_AA)

    return img, total

# --- Streamlit UI ---
st.title("🪙 Indian Coin Detection & Total Calculator")

uploaded_file = st.file_uploader("Upload an image of coins", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    if st.button("🔍 Detect Coins"):
        with st.spinner("Detecting coins..."):
            result_img, total = prediction(img.copy(), model)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                 caption=f"Detected Coins — Total Value: Rupees {total}",
                 use_column_width=True)
        st.success(f"✅ Total Value Detected: Rupees {total}")
