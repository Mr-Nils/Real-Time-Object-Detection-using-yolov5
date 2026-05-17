import cv2
import torch
import streamlit as st
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")
st.title("📹 YOLOv5 Real-time Object Detection Engine")
st.text("Click 'Start Webcam' to open your local camera stream in its native aspect ratio.")

# ----------------------------------------------------
# 2. Hardware Acceleration Auto-Detection
# ----------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    st.sidebar.success("✅ GPU Acceleration (CUDA) Active")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    st.sidebar.success("✅ GPU Acceleration (MPS) Active")
else:
    device = torch.device("cpu")
    st.sidebar.info("ℹ️ Running on CPU")

# ----------------------------------------------------
# 3. Local Model Engine Loading & Class Inspection
# ----------------------------------------------------
WEIGHTS_PATH = "yolov5/best.pt" 

@st.cache_resource
def load_yolov5_model(weights_path):
    model = torch.hub.load(
        'yolov5',          
        'custom',          
        path=weights_path, 
        source='local',    
        force_reload=True  
    )
    model.to(device)
    return model

try:
    model = load_yolov5_model(WEIGHTS_PATH)
    st.success(f"Successfully loaded model weights from: {WEIGHTS_PATH}")
    
    # Print custom object classes to terminal for validation
    print("\n" + "="*50)
    print("SUCCESS: Your model is trained to detect the following classes:")
    print(model.names)
    print("="*50 + "\n")

except Exception as e:
    st.error(f"Error loading model weights. Verify that '{WEIGHTS_PATH}' exists.")
    st.info("Error details: " + str(e))
    st.stop()

# ----------------------------------------------------
# 4. Object Detection Core Parameters
# ----------------------------------------------------
st.sidebar.header("🛠️ Detection Tuning")

conf_thres = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.40, 
    step=0.05,
    help="Minimum confidence score required to display a detected object."
)
model.conf = conf_thres

iou_thres = st.sidebar.slider(
    "Overlap (IoU) Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="Higher values allow more overlapping boxes around adjacent objects."
)
model.iou = iou_thres

# ----------------------------------------------------
# 5. Streamlit Control State & Hardware Interlocking
# ----------------------------------------------------
col1, col2 = st.columns([1, 4])

with col1:
    start_button = st.button("🚀 Start Webcam", use_container_width=True)
    stop_button = st.button("🛑 Stop Webcam", use_container_width=True)

FRAME_WINDOW = st.image([])

if "run_webcam" not in st.session_state:
    st.session_state["run_webcam"] = False

if stop_button:
    st.session_state["run_webcam"] = False
    FRAME_WINDOW.empty()
    
    if "camera_device" in st.session_state and st.session_state["camera_device"] is not None:
        st.session_state["camera_device"].release()
        st.session_state["camera_device"] = None
    st.rerun()

if start_button:
    st.session_state["run_webcam"] = True

# ----------------------------------------------------
# 6. Core Video Capture Loop (Native Aspect Ratio)
# ----------------------------------------------------
if st.session_state["run_webcam"]:
    if "camera_device" not in st.session_state or st.session_state["camera_device"] is None:
        st.session_state["camera_device"] = cv2.VideoCapture(0)
    
    camera = st.session_state["camera_device"]
    
    if not camera.isOpened():
        st.error("Cannot access webcam hardware stream.")
        st.session_state["run_webcam"] = False
    
    while st.session_state["run_webcam"]:
        ret, frame = camera.read()
        if not ret:
            st.warning("Webcam hardware data feed interrupted.")
            break
            
        # Mirrors the frame horizontally for natural movement tracking
        frame = cv2.flip(frame, 1)
            
        # REMOVED cv2.resize -> The frame now flows down using its native dimensions
            
        # Convert frame array structure from OpenCV standard (BGR) to PyTorch standard (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Compute inferences directly on the raw-dimension matrix
        results = model(img_rgb)
        
        # Render boundary coordinates and label layers back over the matrix
        annotated_img = results.render()[0]
        
        # Push the processed matrix frame to the Streamlit viewport container
        FRAME_WINDOW.image(annotated_img, channels="RGB")
        
    camera.release()
    st.session_state["camera_device"] = None