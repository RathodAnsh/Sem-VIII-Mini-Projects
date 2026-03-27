"""
app.py  — Student Drowsiness & Attention Monitor
─────────────────────────────────────────────────
Streamlit dashboard with live camera monitoring and session analytics.

Run:  streamlit run app.py
"""

import os
import time
import glob
import threading
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import pygame
import mediapipe as mp
from datetime import datetime
from collections import deque

# --- Initialize Audio ---
try:
    pygame.mixer.init()
    alarm_sound = None
    # Automatically find any .wav file in the 'sound' folder
    wav_files = glob.glob("sounds/*.wav")
    if wav_files:
        alarm_sound = pygame.mixer.Sound(wav_files[0])
except Exception as e:
    print(f"Audio init error: {e}")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drowsiness Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark, clean, modern UI ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0b0d14;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1118 !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f1118 0%, #151827 100%);
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.main-header h1 {
    color: #f0f4ff;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #0f1118;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 12px 12px 0 0;
}
.metric-card.green::before  { background: #22c55e; }
.metric-card.yellow::before { background: #f59e0b; }
.metric-card.red::before    { background: #ef4444; }
.metric-card.blue::before   { background: #3b82f6; }
.metric-card.purple::before { background: #a855f7; }

.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #f0f4ff;
    line-height: 1;
}
.metric-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.status-awake   { background: #052e1620; border: 1px solid #22c55e40; color: #22c55e; }
.status-drowsy  { background: #4c000020; border: 1px solid #ef444440; color: #ef4444; animation: pulse 1s infinite; }
.status-closing { background: #3d200020; border: 1px solid #f59e0b40; color: #f59e0b; }
.status-noface  { background: #1e1e2e;   border: 1px solid #334155;   color: #64748b; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}

/* Camera frame */
.camera-container {
    background: #0a0b0f;
    border: 1px solid #1e2130;
    border-radius: 16px;
    overflow: hidden;
    position: relative;
}
.camera-placeholder {
    background: #0a0b0f;
    border: 1px dashed #1e2130;
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    color: #334155;
}

/* Progress bar */
.progress-track {
    background: #1a1d2e;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.progress-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.3s ease;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1118;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2130;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #151827 !important;
    color: #e2e8f0 !important;
}

/* Buttons */
.stButton > button {
    background: #151827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
}
.stButton > button[kind="primary"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: white !important;
}

/* Divider */
hr { border-color: #1e2130 !important; }

/* Headings */
h1, h2, h3 { color: #e2e8f0 !important; }
p, span, label { color: #94a3b8; }

/* Slider */
.stSlider [data-testid="stSlider"] { color: #3b82f6; }

/* Alert box */
.alert-box {
    background: #1a050520;
    border: 1px solid #ef444440;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #ef4444;
    font-size: 0.85rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    animation: pulse 1.5s infinite;
}
.success-box {
    background: #05150920;
    border: 1px solid #22c55e40;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #22c55e;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "monitoring":      False,
        "total_frames":    0,
        "closed_frames":   0,
        "drowsy_episodes": 0,
        "attention_score": 100.0,
        "ear_value":       1.0,
        "status":          "NO FACE",
        "session_start":   None,
        "attention_history": [],
        "ear_history":     [],
        "frame":           None,
        "stop_event":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_time(secs: int) -> str:
    return f"{secs//60:02d}:{secs%60:02d}"


def attention_color(score: float) -> str:
    if score >= 70:  return "green"
    if score >= 40:  return "yellow"
    return "red"


def ear_bar_html(ear: float) -> str:
    pct   = min(int(ear / 0.4 * 100), 100)
    color = "#22c55e" if ear >= 0.25 else "#ef4444"
    return f"""
    <div class="metric-label">Eye Aspect Ratio (EAR)</div>
    <div style="display:flex;align-items:center;gap:10px">
      <div style="font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono',monospace;
                  color:#f0f4ff;width:56px">{ear:.3f}</div>
      <div class="progress-track" style="flex:1">
        <div class="progress-fill" style="width:{pct}%;background:{color}"></div>
      </div>
    </div>
    """


def attention_bar_html(score: float) -> str:
    color = "#22c55e" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
    return f"""
    <div class="metric-label">Attention Score</div>
    <div style="display:flex;align-items:center;gap:10px">
      <div style="font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono',monospace;
                  color:#f0f4ff;width:56px">{score:.0f}%</div>
      <div class="progress-track" style="flex:1">
        <div class="progress-fill" style="width:{score}%;background:{color}"></div>
      </div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <div style="font-size:2.5rem">👁️</div>
      <div style="color:#e2e8f0;font-size:1.1rem;font-weight:700;margin-top:0.3rem">
        Drowsiness Monitor
      </div>
      <div style="color:#475569;font-size:0.75rem;font-family:'JetBrains Mono',monospace;
                  margin-top:0.3rem">v1.0 — Transfer Learning</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Settings")

    camera_id = st.selectbox("Camera source", [0, 1, 2], index=0,
                              help="0 = built-in webcam, 1/2 = external")
                              
    # Increased default threshold to catch half-closed eyes better
    ear_thresh = st.slider("EAR threshold", 0.15, 0.40, 0.28, 0.01,
                           help="Increase this value if slight eye-closings are not being detected.")
                           
    # Hidden settings running in background
    alert_frames = 10  # Reduced from 20 to trigger alarm much faster
    use_model = True
    show_landmarks = True


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div style="font-size:2.2rem">👁️</div>
  <div>
    <h1>Student Drowsiness & Attention Monitor</h1>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_analytics = st.tabs(["🎥  Live Monitor", "📊  Analytics"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    # Control row
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 6])
    with ctrl_c1:
        start_btn = st.button("▶  Start Monitoring", type="primary", use_container_width=True)
    with ctrl_c2:
        stop_btn  = st.button("⏹  Stop",             use_container_width=True)
    with ctrl_c3:
        if st.session_state.monitoring:
            elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
            st.markdown(f"""
            <div class="success-box">
              ● Recording — {format_time(elapsed)} &nbsp;|&nbsp;
              Frames: {st.session_state.total_frames} &nbsp;|&nbsp;
              Episodes: {st.session_state.drowsy_episodes}
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Main layout
    cam_col, stats_col = st.columns([3, 1], gap="medium")

    with cam_col:
        frame_holder  = st.empty()
        status_holder = st.empty()

    with stats_col:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        alert_holder  = st.empty()
        ear_holder    = st.empty()
        att_holder    = st.empty()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label' style='margin-bottom:8px'>Session metrics</div>",
                    unsafe_allow_html=True)
        m1 = st.empty()
        m2 = st.empty()
        m3 = st.empty()
        m4 = st.empty()

    # ── Start logic ───────────────────────────────────────────────────────────
    if start_btn and not st.session_state.monitoring:
        # Reset session
        st.session_state.monitoring      = True
        st.session_state.total_frames    = 0
        st.session_state.closed_frames   = 0
        st.session_state.drowsy_episodes = 0
        st.session_state.attention_score = 100.0
        st.session_state.ear_value       = 1.0
        st.session_state.status          = "LOADING..."
        st.session_state.session_start   = time.time()
        st.session_state.attention_history = []
        st.session_state.ear_history       = []

    if stop_btn:
        st.session_state.monitoring = False

    # ── Live loop ─────────────────────────────────────────────────────────────
    if st.session_state.monitoring:
        try:
            from utils.ear_utils import (
                calculate_ear, get_eye_coords, crop_eye,
                LEFT_EYE_IDX, RIGHT_EYE_IDX,
                LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
            )

            # --- CLEAN, STANDARD MEDIAPIPE INITIALIZATION ---
            mp_face_mesh = mp.solutions.face_mesh
            mp_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            # ------------------------------------------------

            model = None
            if use_model and os.path.exists("models/eye_classifier.h5"):
                import tensorflow as tf
                model = tf.keras.models.load_model("models/eye_classifier.h5")

            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

            drowsy_count    = 0
            was_drowsy      = False
            frame_n         = 0
            attention_score = 100.0
            closed_frames   = 0

            CYAN = (220, 200, 60)
            RED  = (60, 60, 240)
            GREEN= (80, 210, 80)

            while st.session_state.monitoring:
                ret, frame = cap.read()
                if not ret:
                    frame_holder.warning("Camera not accessible. Check your camera connection.")
                    break

                frame_n += 1
                h, w     = frame.shape[:2]
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result   = mp_mesh.process(rgb)

                ear           = 1.0
                face_detected = False
                eye_closed    = False

                if result.multi_face_landmarks:
                    face_detected = True
                    lm = result.multi_face_landmarks[0].landmark

                    left_pts  = get_eye_coords(lm, LEFT_EYE_IDX,  w, h)
                    right_pts = get_eye_coords(lm, RIGHT_EYE_IDX, w, h)
                    ear_l     = calculate_ear(left_pts)
                    ear_r     = calculate_ear(right_pts)
                    ear       = (ear_l + ear_r) / 2.0

                    if show_landmarks:
                        for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
                            lx = int(lm[idx].x * w)
                            ly = int(lm[idx].y * h)
                            cv2.circle(frame, (lx, ly), 1, CYAN, -1)

                    model_open = 1.0
                    if model and frame_n % 3 == 0:
                        eye_crop = crop_eye(frame, lm, RIGHT_EYE_IDX, w, h)
                        if eye_crop is not None and eye_crop.size > 0:
                            ei  = cv2.resize(eye_crop, (224, 224))
                            ei  = cv2.cvtColor(ei, cv2.COLOR_BGR2RGB) / 255.0
                            ei  = np.expand_dims(ei, 0)
                            model_open = float(model.predict(ei, verbose=0)[0][0])

                    eye_closed = (ear < ear_thresh) or (model_open < 0.5)

                    if eye_closed:
                        drowsy_count  += 1
                        closed_frames += 1
                        attention_score = max(0.0, attention_score - 0.4)
                    else:
                        drowsy_count    = 0  # INSTANT RESET MATH
                        attention_score = min(100.0, attention_score + 0.15)
                else:
                    attention_score = max(0.0, attention_score - 0.05)

                is_drowsy = drowsy_count >= alert_frames
                
                # --- INSTANT AUDIO STOP FIX ---
                try:
                    if is_drowsy:
                        if alarm_sound and not pygame.mixer.Channel(0).get_busy():
                            pygame.mixer.Channel(0).play(alarm_sound)
                    else:
                        # INSTANTLY cut the sound the millisecond eyes open
                        pygame.mixer.Channel(0).stop() 
                except Exception as e:
                    pass
                # ------------------------------
                
                if is_drowsy and not was_drowsy:
                    st.session_state.drowsy_episodes += 1
                was_drowsy = is_drowsy

                st.session_state.total_frames    = frame_n
                st.session_state.closed_frames   = closed_frames
                st.session_state.attention_score = attention_score
                st.session_state.ear_value       = ear

                if frame_n % 15 == 0:
                    st.session_state.attention_history.append(
                        {"t": frame_n, "attention": round(attention_score, 1)})
                    st.session_state.ear_history.append(
                        {"t": frame_n, "ear": round(ear, 3)})

                # Draw minimal overlay on frame
                if is_drowsy:
                    ov = frame.copy()
                    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 180), -1)
                    cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
                    cv2.rectangle(frame, (0, 0), (w, h), RED, 5)
                    cv2.putText(frame, "DROWSY — WAKE UP!", (w//2 - 160, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2, cv2.LINE_AA)

                ear_color = GREEN if ear >= ear_thresh else RED
                cv2.putText(frame, f"EAR {ear:.3f}", (w-130, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1, cv2.LINE_AA)

                # Display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_holder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Status badge
                if not face_detected:
                    status_class, status_txt = "status-noface",  "● No face detected"
                elif is_drowsy:
                    status_class, status_txt = "status-drowsy",  "⚠ Drowsy — Alert!"
                elif ear < ear_thresh + 0.03:
                    status_class, status_txt = "status-closing", "◑ Eyes closing"
                else:
                    status_class, status_txt = "status-awake",   "● Awake & alert"

                status_holder.markdown(
                    f'<div style="text-align:center;padding:0.4rem 0">'
                    f'<span class="status-badge {status_class}">{status_txt}</span>'
                    f'</div>', unsafe_allow_html=True)

                # Right column metrics
                if is_drowsy:
                    alert_holder.markdown(
                        '<div class="alert-box">⚠ Drowsiness detected!</div>',
                        unsafe_allow_html=True)
                else:
                    alert_holder.empty()

                ear_holder.markdown(ear_bar_html(ear), unsafe_allow_html=True)
                att_holder.markdown(attention_bar_html(attention_score),
                                    unsafe_allow_html=True)

                elapsed = int(time.time() - st.session_state.session_start)
                open_pct = (1 - closed_frames / max(frame_n, 1)) * 100

                m1.markdown(f"""
                <div class="metric-card blue" style="margin-bottom:8px">
                  <div class="metric-label">Session time</div>
                  <div class="metric-value">{format_time(elapsed)}</div>
                </div>""", unsafe_allow_html=True)

                m2.markdown(f"""
                <div class="metric-card {'red' if st.session_state.drowsy_episodes > 0 else 'green'}"
                     style="margin-bottom:8px">
                  <div class="metric-label">Drowsy episodes</div>
                  <div class="metric-value">{st.session_state.drowsy_episodes}</div>
                </div>""", unsafe_allow_html=True)

                m3.markdown(f"""
                <div class="metric-card {attention_color(attention_score)}"
                     style="margin-bottom:8px">
                  <div class="metric-label">Attention</div>
                  <div class="metric-value">{attention_score:.0f}%</div>
                </div>""", unsafe_allow_html=True)

                m4.markdown(f"""
                <div class="metric-card blue">
                  <div class="metric-label">Eye open %</div>
                  <div class="metric-value">{open_pct:.0f}%</div>
                </div>""", unsafe_allow_html=True)

                time.sleep(0.03)

            cap.release()

        except Exception as e:
            st.error(f"Camera error: {e}")
            st.session_state.monitoring = False
    else:
        frame_holder.markdown("""
        <div class="camera-placeholder">
          <div style="font-size:4rem;margin-bottom:1rem">📷</div>
          <div style="color:#475569;font-size:1rem;font-weight:500">Camera not started</div>
          <div style="color:#334155;font-size:0.82rem;margin-top:0.5rem">
            Click <b style="color:#3b82f6">▶ Start Monitoring</b> to begin
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Session Analytics")

    if not st.session_state.attention_history:
        st.markdown("""
        <div class="camera-placeholder" style="padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">📈</div>
          <div style="color:#475569">No session data yet.</div>
          <div style="color:#334155;font-size:0.82rem;margin-top:0.4rem">
            Start monitoring to see live charts.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Summary cards
        s_c1, s_c2, s_c3, s_c4 = st.columns(4, gap="small")
        elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
        open_pct = (1 - st.session_state.closed_frames / max(st.session_state.total_frames, 1)) * 100
        avg_att  = np.mean([d["attention"] for d in st.session_state.attention_history])

        for col, label, value, accent in [
            (s_c1, "Session time",    format_time(elapsed), "blue"),
            (s_c2, "Avg attention",   f"{avg_att:.0f}%",    attention_color(avg_att)),
            (s_c3, "Drowsy episodes", str(st.session_state.drowsy_episodes), "red"),
            (s_c4, "Eye open",        f"{open_pct:.0f}%",   "green"),
        ]:
            col.markdown(f"""
            <div class="metric-card {accent}">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Training curves
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🧠 Training Results")
        tc1, tc2 = st.columns(2, gap="medium")
        with tc1:
            if os.path.exists("static/training_curves.png"):
                st.image("static/training_curves.png",
                         caption="Training & validation curves", use_container_width=True)
            else:
                st.info("Run `python train_model.py` to generate training curves.")
        with tc2:
            if os.path.exists("static/confusion_matrix.png"):
                st.image("static/confusion_matrix.png",
                         caption="Confusion matrix", use_container_width=True)
            else:
                st.info("Confusion matrix will appear here after training.")