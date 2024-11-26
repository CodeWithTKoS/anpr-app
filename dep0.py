import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import streamlit as st

st.set_page_config(page_title="ANPR", page_icon="🚘")

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

# Function to perform ANPR on video frames
def anpr_from_video_frame(frame):
    # Load YOLO model
    model = YOLO("best.pt")  # Provide the path to your YOLO model file

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

    return frame

# Streamlit app
st.title("Automatic Number Plate Recognition (ANPR)")

# Video upload functionality
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary location
    video_path = f"/tmp/{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Stream video frames for ANPR
    stframe = st.empty()  # Placeholder for video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform ANPR on the current frame
        result_frame = anpr_from_video_frame(frame)
        
        # Convert the frame to RGB for displaying in Streamlit
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame
        stframe.image(result_frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
else:
    st.write("Upload a video to start the ANPR process.")

# Sidebar with additional options
with st.sidebar:
    st.write("---")
    st.write("AI App created by @ Puja Ghosal")
