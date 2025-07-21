import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title="ANPR with PaddleOCR", page_icon="🚘")

# Load PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # angle classifier helps with tilted plates

# Function to enhance plate image
def enhance_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return enhanced

# Function to extract text using PaddleOCR
def extract_text_from_plate(plate_img):
    enhanced = enhance_image(plate_img)
    result = ocr.ocr(enhanced, cls=True)
    if result and result[0]:
        return " ".join([line[1][0] for line in result[0]])
    return "No Text Found"

# Function to run ANPR
def anpr_from_image(image):
    model = YOLO("best1.pt")  # Replace with your YOLO model
    results = model.predict(source=image, conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = map(int, det[:6])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        plate = image[y_min:y_max, x_min:x_max]
        text = extract_text_from_plate(plate)
        cv2.putText(image, text, (x_min, y_max + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        st.write(f"Detected Plate Text: `{text}`")
    return image

# Streamlit UI
st.title("Automatic Number Plate Recognition (ANPR) - PaddleOCR")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    result_img = anpr_from_image(image)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Detected Plates", use_column_width=True)
else:
    st.info("Please upload an image to begin.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Debjit Das**")
