import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import streamlit as st

# Configure the Streamlit page
st.set_page_config(page_title="ANPR", page_icon="🚘")

# Configure pytesseract to point to the Tesseract executable (if needed)
# Uncomment the line below and adjust the path if you're using Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to perform OCR on a cropped license plate image using PyTesseract
def get_ocr(im, coors):
    x_min, y_min, x_max, y_max = map(int, coors)
    cropped_plate = im[y_min:y_max, x_min:x_max]

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract for text recognition
    ocr_text = pytesseract.image_to_string(gray, config='--psm 8')  # --psm 8 for sparse text
    return ocr_text.strip()

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
        
        # Crop the detected license plate region
        cropped_plate = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Perform OCR on the cropped license plate region
        plate_text = get_ocr(image, coords)
        
        # Display recognized text on the image
        cv2.putText(image, plate_text, (int(x_min), int(y_max) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Optionally, display the cropped license plate on Streamlit (for debugging purposes)
        st.image(cropped_plate, caption="Cropped License Plate", use_column_width=True)
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
