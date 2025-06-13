import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import shutil
import glob

# Configuration
SAMPLE_IMAGES_DIR = "sample_images"
ANNOTATIONS_DIR = "annotations"

def load_sample_images():
    """Load sample images from the sample_images directory"""
    image_paths = glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.jpg")) + \
                 glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.jpeg")) + \
                 glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.png"))
    return image_paths[:10]  # Return first 10 images

def setup_sample_data():
    """Copy sample images and annotations from the original dataset"""
    # Create directories if they don't exist
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    # Source directories
    source_img_dir = os.path.join("..", "textile-damage-detection", "valid", "images")
    source_label_dir = os.path.join("..", "textile-damage-detection", "valid", "labels")
    
    # Get first 10 images
    image_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")))[:10]
    
    # Clear existing data
    for file in os.listdir(SAMPLE_IMAGES_DIR):
        os.remove(os.path.join(SAMPLE_IMAGES_DIR, file))
    for file in os.listdir(ANNOTATIONS_DIR):
        os.remove(os.path.join(ANNOTATIONS_DIR, file))
    
    # Copy images and annotations
    for i, img_path in enumerate(image_files, 1):
        # Copy image
        img_dst = os.path.join(SAMPLE_IMAGES_DIR, f"sample_{i}.jpg")
        shutil.copy2(img_path, img_dst)
        
        # Copy annotation if exists
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_src = os.path.join(source_label_dir, f"{base_name}.txt")
        if os.path.exists(label_src):
            label_dst = os.path.join(ANNOTATIONS_DIR, f"sample_{i}.txt")
            shutil.copy2(label_src, label_dst)

def draw_annotations(image, annotation_path):
    """Draw annotations on the image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    if not os.path.exists(annotation_path):
        return img
    
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                    _, x_center, y_center, box_w, box_h = map(float, parts[:5])
                    
                    # Convert normalized to pixel coordinates
                    x_center *= w
                    y_center *= h
                    box_w *= w
                    box_h *= h
                    
                    # Calculate coordinates
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    # Draw rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except Exception as e:
        print(f"Error drawing annotations: {e}")
    
    return img

def main():
    st.set_page_config(
        page_title="Textile Defect Viewer",
        page_icon="🔍",
        layout="wide"
    )
    
    # Setup sample data
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    # Copy sample data if directories are empty
    if not os.listdir(SAMPLE_IMAGES_DIR) or not os.listdir(ANNOTATIONS_DIR):
        setup_sample_data()
    
    # Initialize session state
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None
    
    st.title("🔍 Textile Defect Viewer")
    
    # Get sample images
    sample_images = sorted(glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.jpg")))
    
    if st.session_state.selected_sample is not None:
        # Back button
        if st.button("← Back to samples"):
            st.session_state.selected_sample = None
            st.experimental_rerun()
        
        # Show selected image with annotations
        img = cv2.imread(st.session_state.selected_sample)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get corresponding annotation
        base_name = os.path.splitext(os.path.basename(st.session_state.selected_sample))[0]
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{base_name}.txt")
        
        # Draw annotations
        annotated_img = draw_annotations(img, annotation_path)
        
        # Display images side by side
        return
    
    # Show sample grid
    st.write("Click on any sample to view annotations")
    
    # Display 2 columns of samples
    cols = st.columns(2)
    for i, img_path in enumerate(sample_images):
        with cols[i % 2]:
            img = Image.open(img_path)
            # Make image clickable
            if st.button(f"Sample {i+1}", key=f"btn_{i}"):
                st.session_state.selected_sample = img_path
                st.experimental_rerun()
            st.image(img, use_column_width=True, caption=f"Sample {i+1}")

if __name__ == "__main__":
    main()
                
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
