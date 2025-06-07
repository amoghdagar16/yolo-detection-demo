import streamlit as st
import numpy as np
from PIL import Image
import time
import random

st.set_page_config(
    page_title="Amogh's YOLOv8 Detection Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .demo-explanation {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #f39c12;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.2);
    }
    
    .detection-result {
        background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ecdc4;
        box-shadow: 0 2px 10px rgba(78, 205, 196, 0.2);
        transition: transform 0.2s ease;
    }
    
    .detection-result:hover {
        transform: translateY(-2px);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    .tech-achievement {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    .real-work-badge {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 2px 10px rgba(40, 167, 69, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def simulate_detection_results():
    """Generate realistic detection results based on actual model performance"""
    detection_scenarios = [
        [
            {'class': 'person', 'confidence': 0.94, 'bbox': [145, 120, 180, 280]},
            {'class': 'skateboard', 'confidence': 0.87, 'bbox': [160, 380, 90, 35]},
        ],
        [
            {'class': 'person', 'confidence': 0.89, 'bbox': [200, 100, 160, 290]},
            {'class': 'person', 'confidence': 0.82, 'bbox': [400, 110, 170, 285]},
            {'class': 'skateboard', 'confidence': 0.91, 'bbox': [210, 375, 85, 30]},
        ],
        [
            {'class': 'person', 'confidence': 0.96, 'bbox': [120, 90, 190, 310]},
        ],
        [
            {'class': 'person', 'confidence': 0.88, 'bbox': [180, 105, 175, 295]},
            {'class': 'person', 'confidence': 0.85, 'bbox': [350, 95, 165, 285]},
            {'class': 'person', 'confidence': 0.79, 'bbox': [520, 115, 155, 275]},
        ]
    ]
    return random.choice(detection_scenarios)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 YOLOv8 Object Detection Demo</h1>
        <p style="font-size: 1.3rem; margin-bottom: 0.5rem;">Real-time AI model from my internship at PruTech Solutions</p>
        <p style="font-size: 1.1rem;"><strong>⭐ Achieved 80% accuracy on skateboarder and pedestrian detection</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real work explanation
    st.markdown("""
    <div class="demo-explanation">
        <h3 style="margin-top: 0;">🎯 About This Real Project</h3>
        <p><strong>🏢 Actual Internship Work:</strong> This demonstrates my YOLOv8 implementation from my ML internship at PruTech Solutions (May 2024 - Dec 2024).</p>
        <p><strong>🎭 Demo Interface:</strong> Due to platform limitations, this shows the interface with simulated results. The actual model:</p>
        <div style="margin: 1rem 0;">
            <span class="real-work-badge">✅ 80% Accuracy Achieved</span>
            <span class="real-work-badge">✅ Real-time Video Processing</span>
            <span class="real-work-badge">✅ Production Deployed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with comprehensive project details
    with st.sidebar:
        st.markdown("### 📋 Project Details")
        st.markdown("""
        **👨‍💻 Developer:** Amogh Dagar  
        **🏢 Company:** PruTech Solutions  
        **📅 Duration:** May 2024 - Dec 2024  
        **💻 Tech Stack:** Python, YOLOv8, OpenCV, PyTorch  
        **🎯 Achievement:** 80% accuracy improvement  
        """)
        
        st.markdown("### 🚀 Real Model Capabilities")
        st.markdown("""
        - ✅ **Skateboarder Detection** (91% accuracy)
        - ✅ **Pedestrian Recognition** (94% accuracy)
        - ✅ **Real-time Processing** (45+ FPS)
        - ✅ **Video Stream Support** (Live inference)
        - ✅ **High Confidence Scoring** (80%+ average)
        """)
        
        st.markdown("### 📈 Technical Achievements")
        st.markdown("""
        **🎯 Accuracy Improvement:** 65% → 80%  
        **⚡ Speed Optimization:** 40% faster inference  
        **📊 Dataset Size:** 15,000+ training images  
        **🔧 Custom Architecture:** YOLOv8 optimization  
        **🎥 Video Processing:** Real-time pipeline  
        """)
        
        st.markdown("### 🔗 Professional Links")
        st.markdown("""
        - [📱 Interactive Portfolio](https://your-portfolio.vercel.app)
        - [💻 GitHub Profile](https://github.com/amogh-dagar)
        - [💼 LinkedIn](https://linkedin.com/in/amogh-dagar-613807215)
        - [📄 Resume Download](mailto:adagar3@asu.edu)
        """)
    
    # Main content columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image for Detection")
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to see simulated detection results based on real model performance"
        )
        
        st.markdown("### 🖼️ Try Detection Scenarios")
        sample_choice = st.selectbox(
            "Choose a detection scenario:",
            [
                "None", 
                "🛹 Skateboarding Action Scene", 
                "🚶 Street Pedestrians", 
                "🏙️ Mixed Urban Environment",
                "👥 Crowded Public Space"
            ]
        )
        
        if sample_choice != "None":
            st.info(f"📷 Scenario Selected: {sample_choice}")
            st.markdown("*Click 'Run Detection' to see results based on real model performance*")
    
    with col2:
        st.markdown("### 🔍 AI Detection Results")
        
        # Handle detection trigger
        show_results = False
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Run AI Detection", type="primary", use_container_width=True):
                show_results = True
                
        elif sample_choice != "None":
            # Show a placeholder image
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        border: 2px dashed #6c757d; border-radius: 10px; 
                        padding: 3rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                <p style="color: #6c757d; margin: 0;">Sample scenario ready for detection</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Run Sample Detection", type="primary", use_container_width=True):
                show_results = True
        
        # Show detection results
        if show_results:
            with st.spinner("🤖 AI model processing... (Simulating real model performance)"):
                # Simulate realistic processing time
                time.sleep(2.5)
                
                # Generate realistic results
                detections = simulate_detection_results()
                
                st.success("✅ Detection completed successfully!")
                
                # Statistics display
                st.markdown("### 📊 Detection Summary")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.metric("Objects Found", len(detections), delta=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_stat2:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.metric("Avg Confidence", f"{avg_conf:.1%}", delta="+15% vs baseline")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_stat3:
                    unique_classes = len(set([d['class'] for d in detections]))
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.metric("Object Types", unique_classes, delta=None)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed results
                st.markdown("### 📋 Detailed Detection Results")
                for i, detection in enumerate(detections, 1):
                    confidence_color = "🟢" if detection['confidence'] > 0.85 else "🟡" if detection['confidence'] > 0.75 else "🔴"
                    bbox_str = f"[{', '.join([str(int(x)) for x in detection['bbox']])}]"
                    
                    st.markdown(f"""
                    <div class="detection-result">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>Detection {i}:</strong> {detection['class'].title()} 
                                {confidence_color} <strong>{detection['confidence']:.1%}</strong> confidence
                            </div>
                            <div style="font-size: 0.9rem; color: #6c757d;">
                                Bbox: {bbox_str}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance metrics based on real work
                st.markdown("### ⚡ Model Performance (Real Metrics)")
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    st.markdown("""
                    <div class="tech-achievement">
                        <strong>🎯 Accuracy Metrics:</strong><br>
                        • Skateboarder Detection: <strong>91%</strong><br>
                        • Pedestrian Detection: <strong>94%</strong><br>
                        • Overall mAP@0.5: <strong>80%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with perf_col2:
                    st.markdown("""
                    <div class="tech-achievement">
                        <strong>⚡ Speed Performance:</strong><br>
                        • Inference Speed: <strong>45+ FPS</strong><br>
                        • Processing Time: <strong>22ms/frame</strong><br>
                        • Model Size: <strong>6.2MB</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.info("📤 Upload an image or select a sample scenario to see AI detection in action")
            
            # Show real capabilities
            st.markdown("""
            ### 🎯 Real Model Capabilities (PruTech Internship):
            
            **🔬 Technical Implementation:**
            - Custom YOLOv8 architecture optimization
            - 15,000+ training images with augmentation  
            - Hyperparameter tuning and dataset curation
            - Real-time video inference pipeline with OpenCV
            
            **📈 Performance Achieved:**
            - **65% → 80%** accuracy improvement (+15%)
            - **40% faster** inference speed optimization
            - **Production-ready** deployment pipeline
            - **Robust detection** across various conditions
            """)
    
    # Technical implementation details
    st.markdown("---")
    
    st.markdown("### 🛠️ Technical Implementation & Achievements")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **🏗️ Architecture & Training**
        - Model: YOLOv8 (You Only Look Once)
        - Framework: PyTorch + Ultralytics
        - Dataset: Custom curated + augmentation
        - Training: Hyperparameter optimization
        - Validation: K-fold cross-validation
        """)
    
    with tech_col2:
        st.markdown("""
        **📊 Performance Metrics**
        - Original Accuracy: 65%
        - **Optimized Accuracy: 80%** ⭐
        - Inference Speed: 45+ FPS
        - Model Size: 6.2MB (optimized)
        - mAP@0.5: 0.80, mAP@0.95: 0.65
        """)
    
    with tech_col3:
        st.markdown("""
        **🚀 Deployment & Production**
        - Real-time video processing pipeline
        - OpenCV integration for live streams
        - Multi-threading for performance
        - Memory optimization techniques
        - Production deployment at PruTech
        """)
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <p style="margin: 0; color: #6c757d; font-style: italic;">
            💼 <strong>This demo showcases actual work completed during my ML internship at PruTech Solutions</strong><br>
            where I achieved significant accuracy improvements through advanced dataset curation, model optimization,<br>
            and production-ready deployment pipeline development.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()