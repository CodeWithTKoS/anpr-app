import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import streamlit as st

st.set_page_config(page_title="ANPR", page_icon="🚘")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR with English language

# Function to extract characters using PaddleOCR
def extract_characters(plate_image):
    # Convert the license plate image to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR on the grayscale license plate image
    results = ocr.ocr(gray, cls=True)
    extracted_text = " ".join([line[1][0] for line in results[0]])
    return extracted_text.strip()

# Function to perform ANPR using uploaded image
def anpr_from_image(image):
    # Load YOLO model
    model = YOLO("best.pt")  # Provide the path to your YOLO model file

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
    
    # Convert the result image to RGB for displaying in Streamlit
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Display the processed image with detected license plate and text
    st.image(result_image_rgb, caption="Processed Image", use_column_width=True)
else:
    st.write("Upload an image to start the ANPR process.")

# Sidebar with additional options
with st.sidebar:
    st.write("---")
    st.write("AI App created by @ Puja Ghosal")
