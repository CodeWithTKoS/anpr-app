import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import streamlit as st

# Configure the Streamlit page
st.set_page_config(page_title="ANPR", page_icon="🚘")

# Initialize PaddleOCR
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')  # You can set lang='en' or add multiple languages

# Function to enhance the license plate image
def enhance_image(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return enhanced

# Function to extract characters using PaddleOCR
def extract_characters(plate_image):
    enhanced_image = enhance_image(plate_image)
    result = ocr_model.ocr(enhanced_image, cls=True)
    extracted_text = " ".join([line[1][0] for line in result[0]]) if result else ""
    return extracted_text.strip()

# Function to perform ANPR using uploaded image
def anpr_from_image(image):
    model = YOLO("best1.pt")  # Load your YOLO model
    results = model.predict(source=image, conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()

    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls = map(int, detection[:6])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, "Plate", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        plate_image = image[y_min:y_max, x_min:x_max]
        plate_text = extract_characters(plate_image)

        cv2.putText(image, plate_text, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        st.write(f"Detected License Plate Text: {plate_text}")

    return image

# Streamlit App UI
st.title("Automatic Number Plate Recognition (ANPR)")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    result_image = anpr_from_image(image)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    st.image(result_image_rgb, caption="Processed Image", use_column_width=True)
else:
    st.write("Upload an image to start the ANPR process.")

with st.sidebar:
    st.write("---")
    st.write("AI App created by @ Debjit Das")
