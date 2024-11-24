import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import streamlit as st
st.set_page_config(page_title="ANPR", page_icon="🚘")
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Set the root directory (adjust if needed)
root_dir = os.path.join(current_dir, "best.pt")  # Assuming root is one level up

print(f"Model loaded from: {root_dir}")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to extract characters using EasyOCR
def extract_characters(plate_image):
    # Convert the license plate image to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Use EasyOCR for text recognition
    results = reader.readtext(gray)
    extracted_text = " ".join([text for (bbox, text, confidence) in results])
    return extracted_text.strip()

# Function to perform ANPR using uploaded image
def anpr_from_image(image):
    # Load YOLO model
    model = YOLO("best.pt")  # Adjust path as needed

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

        # Extract characters from the license plate
        plate_text = extract_characters(plate_image)
        
        # Display the recognized text on the frame
        cv2.putText(image, plate_text, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        st.write(f"Detected License Plate Text: {plate_text}")

    return image

# Streamlit app
st.title("Automatic Number Plate Recognition (ANPR)")

# Image upload functionality
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Convert uploaded image to OpenCV format
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Process the uploaded image for ANPR
    result_image = anpr_from_image(image)
    
    # Display the processed image with detected license plate and text
    st.image(result_image, caption="Processed Image", use_column_width=True)
else:
    st.write("Upload an image to start the ANPR process.")

# Sidebar with additional options
with st.sidebar:
    st.write("---")
    st.write("AI App created by @ Puja Ghosal")
