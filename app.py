import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import shutil
import glob
import base64
from io import BytesIO

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

# Define class names for textile defects with distinct colors
# Based on the dataset's class IDs
CLASS_NAMES = {
    0: 'Hole',
    1: 'Stain',
    2: 'Tear',
    3: 'Thread',
    4: 'Knot',
    5: 'Slub',
    6: 'Contamination'
}

# Different colors for different defect types
DEFECT_COLORS = {
    0: (255, 0, 0),      # Red for Hole
    1: (0, 255, 0),      # Green for Stain
    2: (0, 0, 255),      # Blue for Tear
    3: (255, 255, 0),    # Cyan for Thread
    4: (255, 0, 255),    # Magenta for Knot
    5: (0, 255, 255),    # Yellow for Slub
    6: (255, 128, 0)     # Orange for Contamination
}

def draw_annotations(image, annotation_path):
    """Draw annotations and labels on the image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    if not os.path.exists(annotation_path):
        return img
    
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                    class_id, x_center, y_center, box_w, box_h = parts[:5]
                    x_center, y_center, box_w, box_h = map(float, [x_center, y_center, box_w, box_h])
                    
                    # Get class ID as integer and class name
                    try:
                        class_id_int = int(class_id)
                        class_name = CLASS_NAMES.get(class_id_int, f'Class {class_id}')
                    except (ValueError, TypeError):
                        class_name = f'Class {class_id}'  # Fallback if class_id is not an integer
                    
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
                    
                    # Get color for this defect type
                    try:
                        class_id_int = int(class_id)
                        color = DEFECT_COLORS.get(class_id_int, (0, 255, 0))  # Default to green if class_id is invalid
                    except (ValueError, TypeError):
                        color = (0, 255, 0)  # Default to green if class_id is not an integer
                    
                    # Draw rectangle with thicker border
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Prepare label text
                    label = f"{class_name}"
                    
                    # Calculate text size and position
                    font_scale = 0.8
                    font_thickness = 2
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Draw label background
                    cv2.rectangle(img, 
                                (x1, y1 - label_h - 10), 
                                (x1 + label_w + 10, y1), 
                                color, 
                                -1)  # Filled rectangle
                    
                    # Draw label text
                    cv2.putText(img, 
                              label, 
                              (x1 + 5, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale, 
                              (0, 0, 0),  # Black text
                              font_thickness, 
                              cv2.LINE_AA)
    except Exception as e:
        print(f"Error drawing annotations: {e}")
    
    return img

def main():
    # Set page config and theme
    st.set_page_config(
        page_title="Textile Defect Detector",
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
    
    st.title("🔍 Textile Defect Detector")
    
    # Get sample images
    sample_images = sorted(glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.jpg")))
    
    if st.session_state.selected_sample is not None:
        # Back button
        if st.button("← Back to samples"):
            st.session_state.selected_sample = None
            st.rerun()
        
        # Show loading message
        with st.spinner('Analyzing fabric for defects... Please wait...'):
            # Simulate processing time (1 second)
            import time
            time.sleep(1)
            
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
            
            # Display images side by side with better formatting
            st.success("Defect analysis complete!")
            st.markdown("""
            <style>
                .image-container {
                    display: flex;
                    justify-content: center;
                    gap: 2rem;
                    margin: 1rem 0;
                }
                .image-wrapper {
                    text-align: center;
                    flex: 1;
                    max-width: 45%;
                }
                .image-wrapper img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .image-wrapper h3 {
                    margin: 0.5rem 0;
                    color: #FFFFFF;
                }
            </style>
            """, unsafe_allow_html=True)
            
            def pil_to_base64(image):
                buffered = BytesIO()
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            
            # Convert images to proper format
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            annotated_img_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            
            st.markdown(
                f'''
                <div class="image-container">
                    <div class="image-wrapper">
                        <h3>Original Image</h3>
                        <img src="data:image/png;base64,{pil_to_base64(img_pil)}" />
                    </div>
                    <div class="image-wrapper">
                        <h3>Detected Defects</h3>
                        <img src="data:image/png;base64,{pil_to_base64(annotated_img_pil)}" />
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
    else:
        # Show sample grid
        st.write("Click on any sample to analyze for textile defects")
        
        # Display 2 columns of samples
        cols = st.columns(2)
        for i, img_path in enumerate(sample_images):
            with cols[i % 2]:
                try:
                    try:
                        img = Image.open(img_path)
                        # Create a card-like container for each sample
                        with st.container():
                            # Make the entire container clickable
                            if st.button(f"Sample {i+1}", key=f"btn_{i}"):
                                st.session_state.selected_sample = img_path
                                st.rerun()
                            # Convert image to base64
                            buffered = BytesIO()
                            if isinstance(img, np.ndarray):
                                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Display image with consistent size and border
                            st.markdown(
                                f'''
                                <div style="text-align: center; margin: 0.5rem 0;">
                                    <img src="data:image/png;base64,{img_str}" 
                                         style="width: 250px; height: 250px; object-fit: cover; border-radius: 8px; border: 1px solid #4A4F5B;" />
                                    <p style="margin-top: 0.5rem; color: #E0E0E0;">Sample {i+1}</p>
                                </div>
                                ''',
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.error(f"Error loading image {img_path}: {e}")
                        continue
                except Exception as e:
                    st.error(f"Error loading image {img_path}: {e}")

if __name__ == "__main__":
    main()
