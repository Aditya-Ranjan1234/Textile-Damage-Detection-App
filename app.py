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

# Define class names based on the dataset's numeric IDs
# Since the dataset uses numeric class IDs, we'll use them directly
CLASS_NAMES = {
    0: 'Defect Type 0',
    1: 'Defect Type 1',
    2: 'Defect Type 2',
    3: 'Defect Type 3',
    4: 'Defect Type 4',
    5: 'Defect Type 5',
    6: 'Defect Type 6'
}

# Different colors for different defect types
DEFECT_COLORS = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Blue
    3: (255, 255, 0),    # Cyan
    4: (255, 0, 255),    # Magenta
    5: (0, 255, 255),    # Yellow
    6: (255, 128, 0)     # Orange
}

def draw_annotations(image, annotation_path):
    """Draw bounding boxes on the image without labels"""
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
                    
                    # Draw rectangle with thicker border (no label)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
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
        
        # Create a responsive grid for the samples
        st.markdown("""
        <style>
            .responsive-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 1rem;
                padding: 0.5rem;
                max-width: 800px;
                margin: 0 auto;
            }
            .sample-card {
                background: #1E1E1E;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid #4A4F5B;
                transition: transform 0.2s, box-shadow 0.2s;
                display: flex;
                flex-direction: column;
            }
            .sample-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            .sample-img-container {
                width: 100%;
                height: 150px;
                position: relative;
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #2A2F3B;
            }
            .sample-img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .sample-title {
                padding: 0.75rem;
                text-align: center;
                color: #E0E0E0;
                font-weight: 500;
            }
            .sample-btn {
                width: 100%;
                padding: 0.5rem;
                background: #2A2F3B;
                color: #E0E0E0;
                border: none;
                cursor: pointer;
                transition: background 0.2s;
            }
            .sample-btn:hover {
                background: #3A3F4B;
            }
        </style>
        """, unsafe_allow_html=True)

        # Display samples in a responsive grid
        st.markdown('<div class="responsive-grid">', unsafe_allow_html=True)
        
        for i, img_path in enumerate(sample_images):
            try:
                img = Image.open(img_path)
                # Convert image to base64
                buffered = BytesIO()
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Create a card for each sample
                st.markdown(
                    f'''
                    <div class="sample-card">
                        <div class="sample-img-container">
                            <img class="sample-img" src="data:image/png;base64,{img_str}" alt="Sample {i+1}" />
                        </div>
                        <div class="sample-title">Sample {i+1}</div>
                        <button class="sample-btn" onclick="window.location.href='?sample={i}'; return false;">View Analysis</button>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                
                # Store the mapping of sample index to path
                if 'sample_paths' not in st.session_state:
                    st.session_state.sample_paths = {}
                st.session_state.sample_paths[str(i)] = img_path
                
            except Exception as e:
                st.error(f"Error loading image {img_path}: {e}")
                continue
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle sample selection from URL parameter
        if 'sample' in st.query_params:
            sample_idx = st.query_params['sample']
            if 'sample_paths' in st.session_state and sample_idx in st.session_state.sample_paths:
                st.session_state.selected_sample = st.session_state.sample_paths[sample_idx]
                st.rerun()

if __name__ == "__main__":
    main()
