import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import streamlit as st
import re

# Configure the Streamlit page
st.set_page_config(page_title="ANPR", page_icon="🚘")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to perform OCR on a cropped license plate image with preprocessing
def get_ocr(im, coors):
    x_min, y_min, x_max, y_max = map(int, coors)
    cropped_plate = im[y_min:y_max, x_min:x_max]

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing to improve OCR accuracy
    # Thresholding: Convert to binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Sharpen the image to enhance text clarity
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    # Resize image to make text larger
    resized = cv2.resize(sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Use EasyOCR for text recognition
    results = reader.readtext(resized)

    # Extract and format the recognized text
    ocr_text = ""
    for result in results:
        text, conf = result[1], result[2]
        if conf > 0.2:  # Confidence threshold for filtering results
            ocr_text = text
            break
    
    # Apply regex validation to correct OCR errors
    ocr_text = validate_license_plate(ocr_text)

    return ocr_text.strip()

# Function to validate the license plate format (using regex)
def validate_license_plate(text):
    # Generalized regex pattern for license plates (handles various formats like ABC 1234, 1234 XYZ, etc.)
    pattern = r"([A-Za-z]{1,3})\s?(\d{1,4})\s?([A-Za-z]{1,3})?"
    match = re.match(pattern, text.strip())
    
    if match:
        # Format the license plate in a consistent way (e.g., "ABC 1234 XYZ")
        formatted_plate = " ".join([match.group(1), match.group(2), match.group(3) if match.group(3) else ""])
        return formatted_plate.strip()
    else:
        return text  # Return original text if it doesn't match the pattern

# Function to perform ANPR using YOLO and OCR
def anpr_from_image(image):
    # Load YOLO model
    model = YOLO("best.pt")  # Replace with your YOLO model file path
    
    # Detect license plates using YOLO
    results = model.predict(source=image, conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data

    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls = detection[:6]
        coords = [x_min, y_min, x_max, y_max]
        
        # Draw a bounding box around the detected license plate
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        # Perform OCR on the cropped license plate region
        plate_text = get_ocr(image, coords)
        
        # Display recognized text on the image
        cv2.putText(image, plate_text, (int(x_min), int(y_max) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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
