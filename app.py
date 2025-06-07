import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Amogh's YOLOv8 Detection Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
from PIL import Image
import os

# Try to import YOLO with better error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    st.success("✅ YOLOv8 successfully loaded!")
except ImportError as e:
    YOLO_AVAILABLE = False
    st.error(f"⚠️ YOLOv8 import failed: {str(e)}")
except Exception as e:
    YOLO_AVAILABLE = False
    st.error(f"⚠️ Unexpected error loading YOLOv8: {str(e)}")

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
    .detection-result {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ecdc4;
    }
    .demo-mode-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffeeba;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLOv8 model with fallback options."""
    if not YOLO_AVAILABLE:
        return None
        
    try:
        # Try to load your custom trained model first
        if os.path.exists("best.pt"):
            st.info("🎯 Loading your custom trained model...")
            model = YOLO("best.pt")
            st.success("✅ Custom model loaded successfully!")
            return model
        
        # Fallback to a pretrained model for demo purposes
        st.warning("⚠️ Custom model 'best.pt' not found. Using pretrained YOLOv8n for demo.")
        model = YOLO("yolov8n.pt")  # This will auto-download
        st.info("📥 Downloaded and loaded YOLOv8n pretrained model for demonstration")
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load any YOLO model: {str(e)}")
        return None

def process_image(image, model):
    """Process image with YOLO model and return results."""
    try:
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Run YOLO detection
        results = model(image_np, conf=0.25)  # Lower confidence for demo
        
        # Get the first result
        result = results[0]
        
        # Plot results on image
        annotated_image = result.plot()
        
        # Extract detection details
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

def show_demo_mode():
    """Show demo mode when YOLO is not available."""
    st.markdown("""
    <div class="demo-mode-warning">
        <h4>🎭 Demo Mode Active</h4>
        <p>YOLOv8 is not currently available in this environment. This showcases the interface design.</p>
        <p><strong>In production:</strong> This connects to my actual trained model achieving 80% accuracy on skateboarder and pedestrian detection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show a placeholder result
    st.markdown("### 🎯 Sample Detection Results")
    st.markdown("""
    <div class="detection-result">
        <strong>Detection 1:</strong> Person 🟢 <strong>94.2%</strong> confidence
    </div>
    <div class="detection-result">
        <strong>Detection 2:</strong> Skateboard 🟢 <strong>87.5%</strong> confidence
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 YOLOv8 Object Detection Demo</h1>
        <p>Real-time AI model trained during my internship at PruTech Solutions</p>
        <p><strong>Achieved 80% accuracy on skateboarder and pedestrian detection</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with project info
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
        
        # Debug info in sidebar
        st.markdown("### 🔧 Debug Info")
        st.write(f"YOLO Available: {YOLO_AVAILABLE}")
        if os.path.exists("best.pt"):
            st.write("✅ Custom model found")
        else:
            st.write("❌ Custom model not found")
    
    # Main content columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image for Detection")
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing people or skateboarders for best results"
        )
        
        # Sample images section
        st.markdown("### 🖼️ Or Try Sample Images")
        sample_choice = st.selectbox(
            "Choose a sample image:",
            ["None", "Street Scene", "Skateboarding", "Pedestrians"]
        )
        
        if sample_choice != "None":
            st.info(f"📷 Selected: {sample_choice}")
    
    with col2:
        st.markdown("### 🔍 Detection Results")
        
        # Check if YOLO is available
        if not YOLO_AVAILABLE:
            show_demo_mode()
            return
        
        # Load model
        model = load_model()
        if model is None:
            show_demo_mode()
            return
        
        # Process uploaded image
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Original Image", use_column_width=True)
            
            # Run detection button
            if st.button("🔍 Run Detection", type="primary"):
                with st.spinner("🤖 AI model processing..."):
                    annotated_image, detections, success = process_image(image, model)
                    
                    if success and annotated_image is not None:
                        # Display annotated image
                        st.image(
                            annotated_image, 
                            caption="🎯 Detection Results", 
                            use_column_width=True
                        )
                        
                        # Display detection statistics
                        if detections:
                            st.markdown("### 📊 Detection Summary")
                            
                            # Stats
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
                            
                            # Detailed results
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
                            st.warning("🔍 No objects detected. Try an image with people or objects.")
                    else:
                        st.error("❌ Detection failed. Please try another image.")
        
        else:
            st.info("📤 Upload an image to start object detection")
            
            # Show what the demo does
            st.markdown("""
            ### 🎯 What this demo shows:
            - **Real-time object detection** using YOLOv8 architecture
            - **80% accuracy** achieved during PruTech internship  
            - **Specialized detection** for skateboarders and pedestrians
            - **Confidence scoring** for each detected object
            - **Production-ready** AI model deployment
            """)
    
    # Footer with technical details
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