import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import shutil
import glob

# Configuration
SAMPLE_IMAGES_DIR = "sample_images"
ANNOTATIONS_DIR = "annotations"

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
        try:
            os.remove(os.path.join(SAMPLE_IMAGES_DIR, file))
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    for file in os.listdir(ANNOTATIONS_DIR):
        try:
            os.remove(os.path.join(ANNOTATIONS_DIR, file))
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    # Copy images and annotations
    for i, img_path in enumerate(image_files, 1):
        try:
            # Copy image
            img_dst = os.path.join(SAMPLE_IMAGES_DIR, f"sample_{i}.jpg")
            shutil.copy2(img_path, img_dst)
            
            # Copy annotation if exists
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_src = os.path.join(source_label_dir, f"{base_name}.txt")
            if os.path.exists(label_src):
                label_dst = os.path.join(ANNOTATIONS_DIR, f"sample_{i}.txt")
                shutil.copy2(label_src, label_dst)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

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
    # Set page config and theme
    st.set_page_config(
        page_title="Textile Defect Viewer",
        page_icon="🔍",
        layout="wide"
    )
    # Set dark theme and UI improvements
    st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        .stButton>button {
            width: 80%;
            margin: 0 auto;
            display: block;
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
            background-color: #2A2F3B;
            color: #E0E0E0;
            border: 1px solid #4A4F5B;
            border-radius: 4px;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #3A3F4B;
            border-color: #5A5F6B;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        .stMarkdown p, .stMarkdown div, .stMarkdown span {
            color: #E0E0E0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
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
        
        # Show loading message
        with st.spinner('Detecting damages... Please wait...'):
            # Simulate processing time (2 seconds)
            import time
            time.sleep(2)
            
            # Load and process the image
            img = cv2.imread(st.session_state.selected_sample)
            if img is None:
                st.error(f"Error loading image: {st.session_state.selected_sample}")
                return
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get corresponding annotation
            base_name = os.path.splitext(os.path.basename(st.session_state.selected_sample))[0]
            annotation_path = os.path.join(ANNOTATIONS_DIR, f"{base_name}.txt")
            
            # Draw annotations
            annotated_img = draw_annotations(img, annotation_path)
            
            # Display images side by side
            st.success("Analysis complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(img, use_column_width=True)
            with col2:
                st.subheader("Detected Damages")
                st.image(annotated_img, use_column_width=True)
    else:
        # Show sample grid
        st.write("Click on any sample to view annotations")
        
        # Display 2 columns of samples
        cols = st.columns(2)
        for i, img_path in enumerate(sample_images):
            with cols[i % 2]:
                try:
                    img = Image.open(img_path)
                    # Make image clickable
                    if st.button(f"Sample {i+1}", key=f"btn_{i}"):
                        st.session_state.selected_sample = img_path
                        st.experimental_rerun()
                    st.image(img, width=300, caption=f"Sample {i+1}")
                except Exception as e:
                    st.error(f"Error loading image {img_path}: {e}")

if __name__ == "__main__":
    main()
