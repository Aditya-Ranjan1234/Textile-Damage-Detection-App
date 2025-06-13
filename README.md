# Textile Damage Detection App

A Streamlit-based web application for detecting damages and defects in textile materials using computer vision.

## Features

- Upload and analyze textile images for defects
- Adjustable confidence threshold for detections
- Visual display of detected damages
- Real-time results with detailed detection information
- User-friendly interface

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd textile-damage-detection-app
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate  # On Windows
   source venv/bin/activate    # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

3. Upload an image of textile material using the file uploader

4. Adjust the confidence threshold using the slider in the sidebar if needed

5. View the detection results and analysis

## Project Structure

```
textile-damage-detection-app/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Future Improvements

- Integrate with a trained deep learning model for accurate defect detection
- Add support for batch processing of multiple images
- Implement export functionality for detection reports
- Add more detailed analysis and statistics
- Support for video input and real-time detection

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
