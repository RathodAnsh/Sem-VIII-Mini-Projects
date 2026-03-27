"""
detector.py
───────────
Real-time drowsiness detection engine.
Combines MediaPipe face mesh + EAR geometry + MobileNetV2 classifier.

Usage (standalone):
    python detector.py
    python detector.py --no_model      # EAR-only mode (no trained model needed)
    python detector.py --camera 1      # use external camera
"""

import cv2
import time
import argparse
import threading
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

from utils.ear_utils import (
    calculate_ear, get_eye_coords, crop_eye,
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
    EAR_THRESHOLD, DROWSY_FRAME_COUNT,
)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN   = (80,  210, 80)
YELLOW  = (40,  210, 220)
RED     = (60,  60,  240)
WHITE   = (240, 240, 240)
GRAY    = (130, 130, 130)
DARK    = (20,  20,  30)
CYAN    = (220, 200, 60)

MODEL_PATH = "models/eye_classifier.h5"
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ── Alert ─────────────────────────────────────────────────────────────────────
class AlertManager:
    def __init__(self):
        self._playing   = False
        self._thread    = None

    def play(self):
        if self._playing:
            return
        self._playing = True
        if PYGAME_AVAILABLE:
            self._thread = threading.Thread(target=self._sound_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._playing = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def _sound_loop(self):
        try:
            pygame.mixer.music.load("sounds/alert.wav")
            pygame.mixer.music.play(-1)
        except Exception:
            pass


# ── Stats tracker ─────────────────────────────────────────────────────────────
class SessionStats:
    def __init__(self):
        self.start_time      = time.time()
        self.total_frames    = 0
        self.closed_frames   = 0
        self.drowsy_episodes = 0
        self.ear_history     = deque(maxlen=300)
        self.attention_hist  = deque(maxlen=300)
        self.attention_score = 100.0
        self._was_drowsy     = False

    def update(self, eye_closed: bool, ear: float, is_drowsy: bool):
        self.total_frames += 1
        self.ear_history.append(ear)
        if eye_closed:
            self.closed_frames  += 1
            self.attention_score = max(0.0, self.attention_score - 0.4)
        else:
            self.attention_score = min(100.0, self.attention_score + 0.15)
        if is_drowsy and not self._was_drowsy:
            self.drowsy_episodes += 1
        self._was_drowsy = is_drowsy
        self.attention_hist.append(self.attention_score)

    @property
    def session_time(self):
        secs = int(time.time() - self.start_time)
        return f"{secs//60:02d}:{secs%60:02d}"

    @property
    def eye_open_pct(self):
        if self.total_frames == 0:
            return 100.0
        return (1 - self.closed_frames / self.total_frames) * 100


# ── Main detector ─────────────────────────────────────────────────────────────
class DrowsinessDetector:
    def __init__(self, camera_id: int = 0, use_model: bool = True):
        self.camera_id    = camera_id
        self.use_model    = use_model and TF_AVAILABLE
        self.model        = None
        self.alert        = AlertManager()
        self.stats        = SessionStats()
        self.drowsy_count = 0

        # MediaPipe
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        if self.use_model:
            self._load_model()

    def _load_model(self):
        import os
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded.")
        else:
            print(f"[WARN] Model not found at {MODEL_PATH}. Running EAR-only mode.")
            self.use_model = False

    def _preprocess_eye(self, eye_img):
        eye_img = cv2.resize(eye_img, (224, 224))
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(eye_img / 255.0, axis=0)

    def _model_predict(self, eye_img):
        """Returns probability of eye being OPEN (1=open, 0=closed)."""
        if self.model is None or eye_img is None or eye_img.size == 0:
            return 1.0
        inp  = self._preprocess_eye(eye_img)
        prob = float(self.model.predict(inp, verbose=0)[0][0])
        return prob

    def _draw_overlay(self, frame, ear, attention, status,
                      is_drowsy, face_detected):
        h, w = frame.shape[:2]

        # Semi-transparent dark sidebar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # ── Status badge ──────────────────────────────────────────────────────
        if not face_detected:
            badge_col  = GRAY
            status_txt = "NO FACE"
        elif is_drowsy:
            badge_col  = RED
            status_txt = "DROWSY!"
        elif ear < EAR_THRESHOLD + 0.05:
            badge_col  = YELLOW
            status_txt = "CLOSING"
        else:
            badge_col  = GREEN
            status_txt = "AWAKE"

        cv2.rectangle(frame, (10, 10), (250, 55), badge_col, -1)
        cv2.rectangle(frame, (10, 10), (250, 55), WHITE, 1)
        cv2.putText(frame, status_txt, (20, 43),
                    FONT, 1.1, DARK, 2, cv2.LINE_AA)

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics = [
            ("EAR",        f"{ear:.3f}"),
            ("Attention",  f"{attention:.0f}%"),
            ("Episodes",   str(self.stats.drowsy_episodes)),
            ("Session",    self.stats.session_time),
            ("Eye open",   f"{self.stats.eye_open_pct:.0f}%"),
        ]
        y = 80
        for label, value in metrics:
            cv2.putText(frame, label, (14, y),
                        FONT, 0.45, GRAY, 1, cv2.LINE_AA)
            cv2.putText(frame, value, (14, y + 18),
                        FONT, 0.65, WHITE, 1, cv2.LINE_AA)
            y += 50

        # ── EAR bar ───────────────────────────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = 14, y, 220, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40,40,50), -1)
        fill = int(bar_w * min(ear / 0.4, 1.0))
        bar_color = GREEN if ear >= EAR_THRESHOLD else RED
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), bar_color, -1)
        cv2.putText(frame, "EAR", (bar_x, bar_y - 5),
                    FONT, 0.38, GRAY, 1, cv2.LINE_AA)
        y += 30

        # ── Attention bar ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (bar_x, y), (bar_x+bar_w, y+bar_h), (40,40,50), -1)
        att_fill  = int(bar_w * attention / 100)
        att_color = GREEN if attention > 60 else (YELLOW if attention > 30 else RED)
        cv2.rectangle(frame, (bar_x, y), (bar_x+att_fill, y+bar_h), att_color, -1)
        cv2.putText(frame, "Attention", (bar_x, y - 5),
                    FONT, 0.38, GRAY, 1, cv2.LINE_AA)

        # ── Drowsy alert overlay ───────────────────────────────────────────────
        if is_drowsy:
            alert_overlay = frame.copy()
            cv2.rectangle(alert_overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(alert_overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (0, 0), (w, h), RED, 6)
            txt  = "WAKE UP!"
            (tw, th), _ = cv2.getTextSize(txt, FONT, 1.8, 3)
            tx = (w - tw) // 2
            ty = h // 2 + th // 2
            cv2.putText(frame, txt, (tx+2, ty+2), FONT, 1.8, DARK,  4, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx,   ty),   FONT, 1.8, RED,   3, cv2.LINE_AA)

        # ── Timestamp ─────────────────────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 95, h - 12),
                    FONT, 0.45, GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit", (w // 2 - 75, h - 12),
                    FONT, 0.4, GRAY, 1, cv2.LINE_AA)

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nDrowsiness Monitor started. Press Q to quit.\n")
        frame_n = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_n += 1
            h, w    = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result  = self.mp_mesh.process(rgb)

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

                # Draw eye contours
                for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
                    lx = int(lm[idx].x * w)
                    ly = int(lm[idx].y * h)
                    cv2.circle(frame, (lx, ly), 1, CYAN, -1)

                # Model prediction every 3rd frame
                model_open = 1.0
                if self.use_model and frame_n % 3 == 0:
                    eye_crop   = crop_eye(frame, lm, RIGHT_EYE_IDX, w, h)
                    model_open = self._model_predict(eye_crop)

                eye_closed = (ear < EAR_THRESHOLD) or (model_open < 0.5)

                if eye_closed:
                    self.drowsy_count += 1
                else:
                    self.drowsy_count = max(0, self.drowsy_count - 1)

            is_drowsy = self.drowsy_count >= DROWSY_FRAME_COUNT
            self.stats.update(eye_closed, ear, is_drowsy)

            if is_drowsy:
                self.alert.play()
            else:
                self.alert.stop()

            frame = self._draw_overlay(
                frame, ear,
                self.stats.attention_score,
                "DROWSY" if is_drowsy else "AWAKE",
                is_drowsy, face_detected
            )

            cv2.imshow("Student Drowsiness Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.alert.stop()

        print("\n── Session Summary ──────────────────────")
        print(f"  Duration       : {self.stats.session_time}")
        print(f"  Total frames   : {self.stats.total_frames}")
        print(f"  Eye open       : {self.stats.eye_open_pct:.1f}%")
        print(f"  Drowsy episodes: {self.stats.drowsy_episodes}")
        print(f"  Avg attention  : {np.mean(list(self.stats.attention_hist)):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",   type=int,  default=0)
    parser.add_argument("--no_model", action="store_true",
                        help="Run EAR-only (no trained model)")
    args = parser.parse_args()
    DrowsinessDetector(args.camera, use_model=not args.no_model).run()
