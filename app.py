import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os

# Import YOLO
from ultralytics import YOLO

# Set page config
st.set_page_config(
    page_title="Textile Damage Detection",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Textile Damage Detection")
    st.write("Upload an image of textile material to detect any damages or defects.")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Model inference
        with st.spinner("Analyzing the image..."):
            try:
                # Load the YOLO model
                model_path = os.path.join("models", "yolov8n.pt")
                model = YOLO(model_path)
                
                # Run inference
                results = model(image_np, conf=confidence_threshold)
                
                # Process results
                output_image = image_np.copy()
                detections = []
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, class_id in zip(boxes, confs, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[int(class_id)]
                        
                        # Store detection info
                        detections.append({
                            "bbox": [x1, y1, x2-x1, y2-y1],
                            "confidence": float(conf),
                            "class": class_name
                        })
                        
                        # Draw bounding box and label
                        color = (0, 255, 0)  # Green for all detections
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(output_image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # If no detections, show a message
                if not detections:
                    st.warning("No defects detected. Try adjusting the confidence threshold.")
                
                # Store detections for display
                st.session_state.detections = detections
                st.session_state.output_image = output_image
                    
            except Exception as e:
                st.error(f"Error during model inference: {str(e)}")
                st.warning("Falling back to mock detections for demonstration.")
                
                # Fallback to mock detections
                height, width = image_np.shape[:2]
                mock_detections = [
                    {"bbox": [width//4, height//4, width//2, height//2],
                     "confidence": 0.87,
                     "class": "tear"},
                    {"bbox": [width//2, height//3, width//3, height//3],
                     "confidence": 0.92,
                     "class": "stain"}
                ]
                
                # Store mock detections for display
                st.session_state.detections = mock_detections
                st.session_state.output_image = image_np.copy()
                
                # Draw mock detections
                for det in mock_detections:
                    if det["confidence"] >= confidence_threshold:
                        x, y, w, h = det["bbox"]
                        color = (0, 0, 255)
                        cv2.rectangle(st.session_state.output_image, (x, y), (x+w, y+h), color, 2)
                        label = f"{det['class']} {det['confidence']:.2f}"
                        cv2.putText(st.session_state.output_image, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display result
            with col2:
                st.subheader("Detection Results")
                if 'output_image' in st.session_state:
                    st.image(st.session_state.output_image, use_column_width=True)
                
                # Show detection summary
                st.subheader("Detection Summary")
                if 'detections' in st.session_state and st.session_state.detections:
                    for i, det in enumerate(st.session_state.detections, 1):
                        if det["confidence"] >= confidence_threshold:
                            st.write(f"{i}. {det['class'].capitalize()}: {det['confidence']:.2f} confidence")
                else:
                    st.warning("No detections to display")
    
    # Add some information about the app
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This is a prototype application for detecting damages in textile materials. "
        "Upload an image of textile material to identify potential defects such as tears, stains, or holes."
    )

if __name__ == "__main__":
    main()
