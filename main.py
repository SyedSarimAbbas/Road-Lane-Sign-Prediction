import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from time import sleep

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Road Lane Detection",
    layout="wide",
    page_icon="🛣️",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# Video Background
# -----------------------------------
# Using a free road driving video from Pexels
VIDEO_URL = "https://videos.pexels.com/video-files/2795173/2795173-uhd_2560_1440_25fps.mp4"

st.markdown(f"""
<style>
    /* Video Background Container */
    .video-bg {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -2;
        object-fit: cover;
    }}
    
    /* Dark overlay for readability */
    .video-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 12, 41, 0.20);
        z-index: -1;
    }}
</style>

<!-- Video Background -->
<video autoplay muted loop playsinline class="video-bg">
    <source src="{VIDEO_URL}" type="video/mp4">
</video>

<!-- Dark Overlay -->
<div class="video-overlay"></div>
""", unsafe_allow_html=True)

# -----------------------------------
# Custom CSS Styling
# -----------------------------------
st.markdown("""
<style>
    /* Main app styling - transparent to show video */
    .stApp {
        background: transparent !important;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #a8b2d1;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #667eea !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a8b2d1 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8892b0 !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.6);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: rgba(17, 153, 142, 0.2);
        border-left: 4px solid #11998e;
        color: #38ef7d !important;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #a8b2d1;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #a8b2d1;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Webcam button */
    .webcam-btn {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .processing-text {
        animation: pulse 1.5s infinite;
        color: #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load YOLO Model
# -----------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "Road_Lane_Detector.pt")
    return YOLO(model_path)

model = load_model()

# -----------------------------------
# Lane Detection Class Names
# -----------------------------------
CLASS_NAMES = {
    0: 'BUS LANE',
    1: 'Yellow Markings',
    2: 'Line 1',
    3: 'Line 2',
    4: 'Line 3',
    5: 'Crossing',
    6: 'Romb',
    7: 'SLOW',
    8: 'Left Arrow',
    9: 'Forward Arrow',
    10: 'Forward Arrow -Left',
    11: 'Forward Arrow -Right',
    12: 'Right Arrow',
    13: 'Bicycle'
}

# Lane colors for visualization
LANE_COLORS = {
    0: (255, 87, 87),     # Red - BUS LANE
    1: (255, 193, 7),     # Yellow - Yellow Markings
    2: (102, 126, 234),   # Purple-blue - Line 1
    3: (138, 43, 226),    # Blue-violet - Line 2
    4: (75, 0, 130),      # Indigo - Line 3
    5: (0, 255, 127),     # Spring Green - Crossing
    6: (255, 165, 0),     # Orange - Romb
    7: (255, 20, 147),    # Deep Pink - SLOW
    8: (0, 191, 255),     # Deep Sky Blue - Left Arrow
    9: (50, 205, 50),     # Lime Green - Forward Arrow
    10: (64, 224, 208),   # Turquoise - Forward Arrow -Left
    11: (255, 105, 180),  # Hot Pink - Forward Arrow -Right
    12: (30, 144, 255),   # Dodger Blue - Right Arrow
    13: (147, 112, 219)   # Medium Purple - Bicycle
}

# -----------------------------------
# Draw Detections
# -----------------------------------
def draw_lane_detections(img, results, conf_threshold=0.3, line_thickness=2, show_labels=True, overlay_alpha=0.3):
    """Draw lane detection results on the image with transparent overlays"""
    if results.boxes is None or len(results.boxes) == 0:
        return img, {}
    
    detection_counts = {}
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        if score < conf_threshold:
            continue
        
        cls_int = int(cls)
        class_name = CLASS_NAMES.get(cls_int, f'Class {cls_int}')
        
        # Count detections
        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        
        # Get color for this class
        color = LANE_COLORS.get(cls_int, (102, 126, 234))
        
        # Create transparent overlay fill inside bounding box
        overlay = img.copy()
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)  # Filled rectangle
        cv2.addWeighted(overlay, overlay_alpha, img, 1 - overlay_alpha, 0, img)
        
        # Draw border rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
        
        if show_labels:
            label = f"{class_name}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for label
            cv2.rectangle(img, (int(x1), int(y1) - label_h - 10), 
                         (int(x1) + label_w + 10, int(y1)), color, -1)
            
            # Label text
            cv2.putText(img, label, (int(x1) + 5, int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img, detection_counts

# -----------------------------------
# Sidebar Configuration
# -----------------------------------
with st.sidebar:
    st.markdown("# 🛣️ Lane Detection")
    st.markdown("---")
    
    st.markdown("### ⚙️ Detection Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    line_thickness = st.slider(
        "Line Thickness",
        min_value=1,
        max_value=5,
        value=2,
        help="Thickness of detection boxes"
    )
    
    show_labels = st.checkbox("Show Labels", value=True, help="Display class labels on detections")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    st.markdown("""
    **Model:** Road Lane Detector  
    **Framework:** YOLOv11  
    **Input:** Image/Video  
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎨 Legend")
    for cls_id, (name) in CLASS_NAMES.items():
        color = LANE_COLORS.get(cls_id, (102, 126, 234))
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        st.markdown(f'<span style="color:{hex_color};">●</span> {name}', unsafe_allow_html=True)

# -----------------------------------
# Main Content
# -----------------------------------
st.markdown('<h1 class="main-header"> AI Road Lane Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Computer Vision for Autonomous Driving & Road Safety</p>', unsafe_allow_html=True)

# -----------------------------------
# Input Selection Tabs
# -----------------------------------
tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Video Upload", "📹 Live Webcam"])

# -----------------------------------
# Tab 1: Image Upload
# -----------------------------------
with tab1:
    st.markdown("### Upload an image to detect road lanes")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_uploader",
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Original Image")
            img = Image.open(uploaded_image)
            st.image(img, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Detection Result")
            
            with st.spinner("🔍 Detecting lanes..."):
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                results = model.predict(img_bgr, verbose=False)[0]
                output_img, counts = draw_lane_detections(
                    img_bgr.copy(), 
                    results, 
                    conf_threshold=confidence_threshold,
                    line_thickness=line_thickness,
                    show_labels=show_labels
                )
                output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                
            st.image(output_rgb, use_container_width=True)
        
        # Detection Statistics
        st.markdown("---")
        st.markdown("### 📊 Detection Statistics")
        
        if counts:
            stat_cols = st.columns(len(counts) + 1)
            
            total = sum(counts.values())
            with stat_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total}</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                """, unsafe_allow_html=True)
            
            for idx, (class_name, count) in enumerate(counts.items(), 1):
                with stat_cols[idx]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{count}</div>
                        <div class="stat-label">{class_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No lanes detected in this image. Try adjusting the confidence threshold.")
        
        # Download Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            Image.fromarray(output_rgb).save(temp_img_file.name)
            st.download_button(
                label="⬇️ Download Result",
                data=open(temp_img_file.name, "rb").read(),
                file_name="lane_detection_result.png",
                mime="image/png"
            )

# -----------------------------------
# Tab 2: Video Upload
# -----------------------------------
with tab2:
    st.markdown("### Upload a video to detect road lanes")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video:
        # Save uploaded video temporarily
        tfile_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_input.write(uploaded_video.read())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Original Video")
            st.video(tfile_input.name)
        
        with col2:
            st.markdown("#### 🎯 Detection Output")
            video_preview_container = st.container()
            download_container = st.container()
        
        # Process Video Button
        if st.button("🚀 Start Lane Detection", key="process_video"):
            cap = cv2.VideoCapture(tfile_input.name)
            
            # Output video setup
            tfile_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_path = tfile_output.name
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = 20 if fps == 0 else fps
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            live_stats = st.empty()
            
            # Statistics tracking
            total_detections = 0
            processed_frames = 0
            class_counts = {}  # Track detections per class
            last_frame_rgb = None  # Store last processed frame
            
            # Use the preview container for live updates
            with video_preview_container:
                video_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.predict(frame, verbose=False)[0]
                frame_out, counts = draw_lane_detections(
                    frame.copy(), 
                    results,
                    conf_threshold=confidence_threshold,
                    line_thickness=line_thickness,
                    show_labels=show_labels
                )
                out.write(frame_out)
                
                # Update display
                frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                last_frame_rgb = frame_rgb  # Store for final display
                
                # Track per-class statistics
                for class_name, count in counts.items():
                    class_counts[class_name] = class_counts.get(class_name, 0) + count
                
                total_detections += sum(counts.values())
                processed_frames += 1
                
                progress = processed_frames / frame_count
                progress_bar.progress(progress)
                status_text.markdown(f'<p class="processing-text">🔄 Analyzing frame {processed_frames}/{frame_count}...</p>', unsafe_allow_html=True)
                
                live_stats.markdown(f"**Live Stats:** {total_detections} detections | Avg: {total_detections/processed_frames:.1f}/frame")
            
            cap.release()
            out.release()
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            live_stats.empty()
            video_placeholder.empty()
            
            # Show processed video in the preview area
            with video_preview_container:
                st.video(output_path)
            
            # Show download button below the preview
            with download_container:
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                    file_size_mb = len(video_bytes) / (1024 * 1024)
                    st.download_button(
                        label=f"⬇️ Download ({file_size_mb:.1f} MB)",
                        data=video_bytes,
                        file_name="lane_detection_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            # ========================================
            # FINAL OUTPUT SECTION
            # ========================================
            st.markdown("---")
            st.markdown("## 🎬 Analysis Complete!")
            
            # Success message with animation
            st.success(f"✅ Successfully processed **{processed_frames}** frames and detected **{total_detections}** lane markings!")
            
            # Video duration calculation
            duration_seconds = processed_frames / fps if fps > 0 else 0
            duration_str = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
            
            # ========================================
            # Video Metrics Row
            # ========================================
            st.markdown("### 📊 Video Analysis Summary")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{processed_frames}</div>
                    <div class="stat-label">Total Frames</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{duration_str}</div>
                    <div class="stat-label">Duration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total_detections}</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[3]:
                avg_per_frame = total_detections / processed_frames if processed_frames > 0 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{avg_per_frame:.1f}</div>
                    <div class="stat-label">Avg per Frame</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ========================================
            # Per-Class Detection Breakdown
            # ========================================
            if class_counts:
                st.markdown("### 🏷️ Detection Breakdown by Class")
                
                # Sort by count descending
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Create dynamic columns based on number of classes detected
                num_classes = len(sorted_classes)
                cols_per_row = min(4, num_classes)
                
                for i in range(0, num_classes, cols_per_row):
                    class_cols = st.columns(cols_per_row)
                    for j, col in enumerate(class_cols):
                        if i + j < num_classes:
                            class_name, count = sorted_classes[i + j]
                            percentage = (count / total_detections * 100) if total_detections > 0 else 0
                            with col:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div class="stat-number">{count}</div>
                                    <div class="stat-label">{class_name}</div>
                                    <div style="color: #667eea; font-size: 0.8rem;">{percentage:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
            
            # ========================================
            # Final Video Output
            # ========================================
            st.markdown("---")
            st.markdown("### 🎥 Processed Video Output")
            
            # Show last frame as thumbnail
            if last_frame_rgb is not None:
                st.markdown("**Preview (Last Frame):**")
                st.image(last_frame_rgb, use_container_width=True, caption="Final frame from processed video")
            
            # Full video player
            st.markdown("**Full Processed Video:**")
            st.video(output_path)
            
            # ========================================
            # Download Section
            # ========================================
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                    file_size_mb = len(video_bytes) / (1024 * 1024)
                    
                    st.download_button(
                        label=f"⬇️ Download Processed Video ({file_size_mb:.1f} MB)",
                        data=video_bytes,
                        file_name="lane_detection_output.mp4",
                        mime="video/mp4"
                    )

# -----------------------------------
# Tab 3: Live Webcam
# -----------------------------------
with tab3:
    st.markdown("### Real-time lane detection from webcam")
    
    st.info("📷 Click 'Start Webcam' to begin real-time lane detection. Press 'Stop' to end the session.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        start_btn = st.button("🎬 Start Webcam", key="start_webcam", type="primary")
    with col2:
        stop_btn = st.button("⏹️ Stop Webcam", key="stop_webcam")
    
    webcam_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    if start_btn:
        st.session_state.webcam_running = True
    
    if stop_btn:
        st.session_state.webcam_running = False
    
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera permissions.")
            st.session_state.webcam_running = False
        else:
            frame_count = 0
            total_detections = 0
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read from webcam")
                    break
                
                results = model.predict(frame, verbose=False)[0]
                frame_out, counts = draw_lane_detections(
                    frame.copy(), 
                    results,
                    conf_threshold=confidence_threshold,
                    line_thickness=line_thickness,
                    show_labels=show_labels
                )
                
                frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(frame_rgb, use_container_width=True)
                
                frame_count += 1
                total_detections += sum(counts.values())
                
                stats_placeholder.markdown(f"""
                **Frames Processed:** {frame_count} | **Total Detections:** {total_detections} | **Current Frame:** {sum(counts.values())} lanes
                """)
                
                if stop_btn:
                    break
            
            cap.release()

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    padding: 3rem 2rem; 
    margin-top: 2rem;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
">
    <p style="
        font-size: 2.5rem; 
        font-weight: 800; 
        color: #ffffff;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    ">Road Lane Detection System</p>
    <p style="
        font-size: 1.4rem; 
        color: #ffffff; 
        margin-bottom: 1.5rem;
        font-weight: 500;
    ">Powered by <span style="color: #667eea; font-weight: 700;">YOLOv11</span> & <span style="color: #764ba2; font-weight: 700;">Streamlit</span></p>
    <p style="
        font-size: 1.3rem; 
        color: #e0e0e0;
        padding-top: 1.5rem;
        border-top: 2px solid rgba(255, 255, 255, 0.15);
        font-weight: 500;
    ">👨‍💻 Built by <span style="
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 1.5rem;
    ">Syed Sarim Abbas</span></p>
    <p style="
        font-size: 1.1rem;
        color: #a8b2d1;
        margin-top: 0.5rem;
    ">for Autonomous Driving & Road Safety</p>
</div>
""", unsafe_allow_html=True)
