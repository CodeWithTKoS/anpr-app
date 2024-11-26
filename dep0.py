import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="ANPR", page_icon="🚘")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the root directory (adjust if needed)
root_dir = os.path.join(current_dir, "best.pt")  # Assuming root is one level up

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to extract characters using EasyOCR
def extract_characters(plate_image):
    # Convert the license plate image to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Enhance the image (e.g., thresholding or noise removal)
    _, enhanced_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use EasyOCR for text recognition
    results = reader.readtext(enhanced_image)
    extracted_text = " ".join([text for (bbox, text, confidence) in results])
    return extracted_text.strip()

# Function to process uploaded image
def process_uploaded_image(uploaded_image):
    # Convert the uploaded image to an OpenCV format
    image = np.array(uploaded_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Load YOLO model
    model = YOLO(root_dir)

    # Detect license plate using YOLO
    results = model.predict(source=image, conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data

    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls = map(int, detection)

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, "Plate", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Crop the license plate region
        plate_image = image[y_min:y_max, x_min:x_max]

        # Enhance the cropped image for OCR
        plate_text = extract_characters(plate_image)

        # Display the recognized text on the image
        cv2.putText(image, plate_text, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Convert frame to RGB for Streamlit
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, channels="RGB", caption="Processed Image", use_column_width=True)

    return plate_text

# Streamlit app
st.title("Automatic Number Plate Recognition (ANPR)")

st.sidebar.header("Settings")
model_path = root_dir

uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    plate_text = process_uploaded_image(uploaded_image)
    st.write(f"Detected License Plate Text: {plate_text}")

with st.sidebar:
    st.write("---")
    st.write("AI App created by @ Puja Ghosal")
