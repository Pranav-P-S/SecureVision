"""
SecureVision - Intelligent Fire & Smoke Detection System

A real-time fire and smoke detection application using deep learning.
Built with Streamlit, TensorFlow, and OpenCV.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from pathlib import Path

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="SecureVision - Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
from config import get_model_path, DETECTION_CONFIG, UI_CONFIG, ASSETS_DIR
from utils import load_model, prepare_image, predict, draw_prediction, get_detection_status


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
def load_css():
    """Load custom CSS for modern UI."""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-right: 1px solid #334155;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #f8fafc !important;
        }
        
        /* Alert boxes */
        .alert-danger {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            animation: pulse 1s infinite;
            margin: 10px 0;
        }
        
        .alert-warning {
            background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            animation: pulse 1.5s infinite;
            margin: 10px 0;
        }
        
        .alert-safe {
            background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.02); }
        }
        
        /* Cards */
        .info-card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
        }
        
        /* Feature cards */
        .feature-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            border: 1px solid #475569;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 12px;
        }
        
        /* Stats */
        .stat-card {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
        }
        
        /* Confidence meter */
        .confidence-meter {
            height: 24px;
            border-radius: 12px;
            background: #334155;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        
        /* Tips section */
        .safety-tip {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #64748b;
            border-top: 1px solid #334155;
            margin-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# AUDIO ALERT SYSTEM
# ============================================================================
def generate_alert_sound():
    """Generate a simple alert sound using base64 encoded audio."""
    # Simple beep sound encoded as base64 WAV
    # This is a 440Hz tone for 0.5 seconds
    alert_js = """
    <script>
    function playAlertSound() {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 880;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    }
    playAlertSound();
    </script>
    """
    return alert_js


def play_alert():
    """Play audio alert when danger is detected."""
    alert_html = generate_alert_sound()
    st.components.v1.html(alert_html, height=0)


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def get_model():
    """Load and cache the detection model."""
    try:
        model_path = get_model_path()
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# ============================================================================
# PAGE: HOME
# ============================================================================
def page_home():
    """Render the home page with fire safety information."""
    
    # Hero section
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 10px;">
            🔥 SecureVision
        </h1>
        <p style="font-size: 1.3rem; color: #94a3b8; max-width: 600px; margin: 0 auto;">
            Intelligent Fire & Smoke Detection powered by Deep Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Real-time Detection</h3>
            <p style="color: #94a3b8;">Advanced computer vision detects fire and smoke instantly from your webcam feed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔔</div>
            <h3>Instant Alerts</h3>
            <p style="color: #94a3b8;">Audio and visual warnings notify you immediately when danger is detected.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>AI Powered</h3>
            <p style="color: #94a3b8;">Built on ResNet50 architecture trained on thousands of fire and smoke images.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fire Safety Tips Section
    st.markdown("## 🛡️ Fire Safety Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚪 Trapped in Room", 
        "💨 Caught in Smoke", 
        "🏢 High-Rise Safety",
        "🧯 Fire Extinguisher"
    ])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
            <h3>If You're Trapped in a Room</h3>
            <div class="safety-tip">
                <strong>1.</strong> Seal all cracks with wet cloth to prevent smoke entry
            </div>
            <div class="safety-tip">
                <strong>2.</strong> Close windows — only break them as absolute last resort
            </div>
            <div class="safety-tip">
                <strong>3.</strong> Call emergency services (101 for Fire in India)
            </div>
            <div class="safety-tip">
                <strong>4.</strong> Stay on the line until help arrives
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="info-card">
            <h3>If Caught in Smoke</h3>
            <div class="safety-tip">
                <strong>1.</strong> Drop to the ground and crawl — smoke rises, cleaner air is below
            </div>
            <div class="safety-tip">
                <strong>2.</strong> Keep your head at 30-35° angle from the floor
            </div>
            <div class="safety-tip">
                <strong>3.</strong> Breathe shallowly through your shirt as a filter
            </div>
            <div class="safety-tip">
                <strong>4.</strong> Move quickly away from the smoke-filled area
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="info-card">
            <h3>High-Rise Apartment Safety</h3>
            <div class="safety-tip">
                <strong>1.</strong> Use a bright colored blanket to signal from window
            </div>
            <div class="safety-tip">
                <strong>2.</strong> At night, use a flashlight or phone light
            </div>
            <div class="safety-tip">
                <strong>3.</strong> Never use elevators during a fire
            </div>
            <div class="safety-tip">
                <strong>4.</strong> Know your building's evacuation routes in advance
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="info-card">
            <h3>PASS Technique for Fire Extinguishers</h3>
            <div class="safety-tip">
                <strong>P</strong> — PULL the safety pin from handle
            </div>
            <div class="safety-tip">
                <strong>A</strong> — AIM the nozzle at the BASE of the fire
            </div>
            <div class="safety-tip">
                <strong>S</strong> — SQUEEZE the trigger handle
            </div>
            <div class="safety-tip">
                <strong>S</strong> — SWEEP from side to side
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Emergency numbers
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h2>🚒 101</h2>
            <p>Fire Department (India)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);">
            <h2>🚔 100</h2>
            <p>Police Emergency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);">
            <h2>🚑 102</h2>
            <p>Medical Emergency</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: LIVE DETECTION
# ============================================================================
def page_live_detection():
    """Render the live webcam detection page."""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>📹 Live Detection</h1>
        <p style="color: #94a3b8;">Real-time fire and smoke detection from your webcam</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = get_model()
    if model is None:
        st.error("⚠️ Model could not be loaded. Please check if the model file exists.")
        return
    
    # Detection controls
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=DETECTION_CONFIG["confidence_threshold"],
            step=0.05,
            help="Minimum confidence to trigger detection"
        )
        
        enable_audio = st.checkbox("🔊 Enable Audio Alerts", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Status")
        status_placeholder = st.empty()
        confidence_placeholder = st.empty()
    
    with col1:
        # Camera input using Streamlit's native camera
        st.markdown("### 📸 Camera Feed")
        
        # Note about WebRTC
        st.info("💡 **Tip:** For best results, ensure good lighting and position the camera to capture the area you want to monitor.")
        
        camera_image = st.camera_input("Capture frame for detection", label_visibility="collapsed")
        
        if camera_image is not None:
            # Process the captured image
            image = Image.open(camera_image)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Prepare and predict
            config = DETECTION_CONFIG.copy()
            config["confidence_threshold"] = confidence_threshold
            
            prepared = prepare_image(img_bgr, config["img_size"])
            class_id, confidence, label = predict(model, prepared, config)
            
            # Draw prediction on image
            annotated = draw_prediction(img_bgr, label, confidence, class_id, config)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Display result
            st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
            
            # Update status
            status = get_detection_status(class_id, confidence, config)
            
            if status["is_danger"]:
                if class_id == 1:
                    status_placeholder.markdown(
                        '<div class="alert-danger">🚨 FIRE DETECTED!</div>',
                        unsafe_allow_html=True
                    )
                else:
                    status_placeholder.markdown(
                        '<div class="alert-warning">⚠️ SMOKE DETECTED!</div>',
                        unsafe_allow_html=True
                    )
                
                # Play audio alert
                if enable_audio:
                    play_alert()
            else:
                status_placeholder.markdown(
                    '<div class="alert-safe">✅ Environment Safe</div>',
                    unsafe_allow_html=True
                )
            
            # Confidence meter
            color = status["color"]
            confidence_placeholder.markdown(f"""
            <div style="margin-top: 10px;">
                <p style="margin-bottom: 5px;">Confidence: {confidence:.1%}</p>
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {confidence*100}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# PAGE: IMAGE UPLOAD
# ============================================================================
def page_image_upload():
    """Render the image upload detection page."""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🖼️ Image Analysis</h1>
        <p style="color: #94a3b8;">Upload an image to check for fire or smoke</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = get_model()
    if model is None:
        st.error("⚠️ Model could not be loaded. Please check if the model file exists.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image to analyze for fire or smoke"
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
        
        # Prepare and predict
        prepared = prepare_image(img_bgr, DETECTION_CONFIG["img_size"])
        class_id, confidence, label = predict(model, prepared, DETECTION_CONFIG)
        
        # Draw prediction
        annotated = draw_prediction(img_bgr, label, confidence, class_id, DETECTION_CONFIG)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.markdown("### Analysis Result")
            st.image(annotated_rgb, use_container_width=True)
        
        # Status display
        status = get_detection_status(class_id, confidence, DETECTION_CONFIG)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if status["is_danger"]:
            if class_id == 1:
                st.markdown(
                    '<div class="alert-danger">🚨 FIRE DETECTED! Confidence: {:.1%}</div>'.format(confidence),
                    unsafe_allow_html=True
                )
                play_alert()
            else:
                st.markdown(
                    '<div class="alert-warning">⚠️ SMOKE DETECTED! Confidence: {:.1%}</div>'.format(confidence),
                    unsafe_allow_html=True
                )
                play_alert()
        else:
            st.markdown(
                '<div class="alert-safe">✅ No Fire or Smoke Detected - Environment appears safe ({:.1%} confidence)</div>'.format(confidence),
                unsafe_allow_html=True
            )
    else:
        # Show sample
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <p style="font-size: 3rem; margin-bottom: 10px;">📤</p>
            <p style="color: #94a3b8;">Drag and drop an image here or click to browse</p>
            <p style="color: #64748b; font-size: 0.9rem;">Supports: JPG, JPEG, PNG, WebP</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: ABOUT
# ============================================================================
def page_about():
    """Render the about page."""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>ℹ️ About SecureVision</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Project Overview</h3>
            <p style="color: #94a3b8;">
                SecureVision is an AI-powered fire and smoke detection system designed 
                to protect homes and buildings. Using state-of-the-art deep learning, 
                it provides real-time monitoring and instant alerts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Model Architecture</h3>
            <p style="color: #94a3b8;">
                Built on <strong>ResNet50</strong>, a powerful convolutional neural network 
                pre-trained on ImageNet. The model is fine-tuned on fire and smoke 
                datasets for accurate detection in various conditions.
            </p>
            <ul style="color: #94a3b8;">
                <li>3 classes: Safe, Fire, Smoke</li>
                <li>224x224 input resolution</li>
                <li>Real-time inference capable</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🛠️ Technology Stack</h3>
            <ul style="color: #94a3b8;">
                <li><strong>TensorFlow/Keras</strong> - Deep Learning</li>
                <li><strong>OpenCV</strong> - Image Processing</li>
                <li><strong>Streamlit</strong> - Web Interface</li>
                <li><strong>NumPy</strong> - Numerical Computing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📚 Resources</h3>
            <ul style="color: #94a3b8;">
                <li><a href="https://emergency.vt.edu/ready/guides/building-fire/building-fire-during.html" target="_blank">Fire Safety Guidelines</a></li>
                <li><a href="https://nidm.gov.in/PDF/pubs/Fires_in_India_2020.pdf" target="_blank">Fire Statistics - India</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Made with ❤️ for safety | SecureVision © 2024</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""
    
    # Load CSS
    load_css()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2>🔥 SecureVision</h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">Fire & Smoke Detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📹 Live Detection", "🖼️ Image Upload", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 10px; text-align: center;">
            <p style="color: #64748b; font-size: 0.8rem;">
                🚨 Emergency: 101<br>
                (Fire Department - India)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Render selected page
    if page == "🏠 Home":
        page_home()
    elif page == "📹 Live Detection":
        page_live_detection()
    elif page == "🖼️ Image Upload":
        page_image_upload()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
