import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import streamlit as st
st.set_page_config(page_title="ANPR",page_icon="🚘")
import os
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Set the root directory (adjust if needed)
root_dir = os.path.join(current_dir, "best.pt") # Assuming root is one level up

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

# Function to perform ANPR using webcam feed
def anpr_webcam(model_path):
    # Load YOLO model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
        return

    st.warning("Press 'q' in the console to quit the webcam feed.")
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Detect license plate using YOLO
        results = model.predict(source=frame, conf=0.5)
        detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data

        for detection in detections:
            x_min, y_min, x_max, y_max, conf, cls = map(int, detection)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Plate", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Crop the license plate region
            plate_image = frame[y_min:y_max, x_min:x_max]

            # Extract characters from the license plate
            plate_text = extract_characters(plate_image)
            
            # Display the recognized text on the frame
            cv2.putText(frame, plate_text, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            st.write(f"Detected License Plate Text: {plate_text}")

        # Convert frame to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

        # Break loop on 'q' key press in the console
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# File path input
st.sidebar.header("Settings")
model_path = root_dir

# Streamlit app
st.title("Automatic Number Plate Recognition (ANPR)")

if st.sidebar.button("Start Webcam"):
    if model_path:
        anpr_webcam(model_path)
    else:
        st.error("Please provide a valid YOLO model path.")

if st.sidebar.button("Stop Webcam"):
    st.stop()

with st.sidebar:
    st.write("---")
    st.write("AI App created by @ Puja Ghosal")
    