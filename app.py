import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Amogh's YOLOv8 Detection Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Try to import YOLO, fallback to demo mode if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("⚠️ YOLOv8 not installed. Install with: pip install ultralytics")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stat-box {
        text-align: center;
        padding: 1rem;
        background: #f0f2f6;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .detection-result {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ecdc4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLOv8 model."""
    try:
        model = YOLO("best.pt")  # Your trained model in the repo
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def process_image(image, model):
    """Process image with YOLO model and return results."""
    try:
        image_np = np.array(image)
        results = model(image_np, conf=0.3)
        result = results[0]
        annotated_image = result.plot()

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)

        return annotated_image, detections, True

    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
        return None, [], False

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 YOLOv8 Object Detection Demo</h1>
        <p>Real-time AI model trained during my internship at PruTech Solutions</p>
        <p><strong>Achieved 80% accuracy on skateboarder and pedestrian detection</strong></p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📋 Project Details")
        st.markdown("""
        **Developer:** Amogh Dagar  
        **Company:** PruTech Solutions  
        **Duration:** May 2024 - Dec 2024  
        **Tech Stack:** Python, YOLOv8, OpenCV, PyTorch  
        **Achievement:** 80% accuracy improvement  
        """)

        st.markdown("### 🎯 Model Capabilities")
        st.markdown("""
        - **Skateboarder Detection**
        - **Pedestrian Recognition** 
        - **Real-time Processing**
        - **High Confidence Scoring**
        """)

        st.markdown("### 🔗 Links")
        st.markdown("""
        - [Portfolio](https://your-portfolio.vercel.app)
        - [GitHub](https://github.com/amogh-dagar)
        - [LinkedIn](https://linkedin.com/in/amogh-dagar-613807215)
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Image for Detection")
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing people or skateboarders for best results"
        )

        st.markdown("### 🖼️ Or Try Sample Images")
        sample_choice = st.selectbox(
            "Choose a sample image:",
            ["None", "Street Scene", "Skateboarding", "Pedestrians"]
        )

        if sample_choice != "None":
            st.info(f"📷 Selected: {sample_choice} - Click 'Run Detection' to process")

    with col2:
        st.markdown("### 🔍 Detection Results")

        if not YOLO_AVAILABLE:
            st.error("YOLOv8 not available. This is a demo interface.")
            st.info("In production, this connects to my trained model achieving 80% accuracy.")
            return

        model = load_model()
        if model is None:
            st.error("Failed to load YOLO model")
            return

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Original Image", use_column_width=True)

            if st.button("🔍 Run Detection", type="primary"):
                with st.spinner("🤖 AI model processing..."):
                    annotated_image, detections, success = process_image(image, model)

                    if success and annotated_image is not None:
                        st.image(annotated_image, caption="🎯 Detection Results", use_column_width=True)

                        if detections:
                            st.markdown("### 📊 Detection Summary")
                            total_detections = len(detections)
                            avg_confidence = np.mean([d['confidence'] for d in detections])
                            unique_classes = len(set([d['class'] for d in detections]))

                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("Objects Found", total_detections)
                            with col_stat2:
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            with col_stat3:
                                st.metric("Object Types", unique_classes)

                            st.markdown("### 📋 Detailed Results")
                            for i, detection in enumerate(detections, 1):
                                confidence_color = "🟢" if detection['confidence'] > 0.7 else "🟡" if detection['confidence'] > 0.5 else "🔴"
                                st.markdown(f"""
                                <div class="detection-result">
                                    <strong>Detection {i}:</strong> {detection['class'].title()} 
                                    {confidence_color} <strong>{detection['confidence']:.1%}</strong> confidence
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("🔍 No objects detected. Try an image with people or skateboarders.")
                    else:
                        st.error("❌ Detection failed. Please try another image.")

        elif sample_choice != "None":
            st.info(f"📷 Sample image selected: {sample_choice}")
            st.markdown("Upload an image or use the sample to see detection in action!")

        else:
            st.info("📤 Upload an image to start object detection")
            st.markdown("""
            ### 🎯 What this demo shows:
            - **Real-time object detection** using my trained YOLOv8 model
            - **80% accuracy** achieved during PruTech internship
            - **Specialized detection** for skateboarders and pedestrians
            - **Confidence scoring** for each detected object
            - **Production-ready** AI model deployment
            """)

    st.markdown("---")
    st.markdown("""
    ### 🛠️ Technical Implementation
    **Model Architecture:** YOLOv8 (You Only Look Once)  
    **Training Dataset:** Custom curated dataset with augmentation  
    **Framework:** PyTorch, Ultralytics  
    **Optimization:** Hyperparameter tuning, dataset curation  
    **Deployment:** Real-time video inference pipeline with OpenCV  
    
    *This demo represents actual work completed during my ML internship at PruTech Solutions,
    where I achieved 80% accuracy improvement through dataset curation and model optimization.*
    """)

if __name__ == "__main__":
    main()
