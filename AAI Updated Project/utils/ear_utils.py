import numpy as np

# ── Eye Aspect Ratio ─────────────────────────────────────────────────────────
def calculate_ear(eye_landmarks):
    """
    Calculates Eye Aspect Ratio (EAR) from 6 landmark points.

    Points layout:
        p1 ──────────── p4
          p2          p5
        p3 ──────────── p6

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    When eye is open  → EAR ~0.30
    When eye is closed → EAR ~0.20 or less
    """
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_landmarks]
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ── MediaPipe landmark indices for eyes ──────────────────────────────────────
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# For drawing eye contour (more points)
LEFT_EYE_CONTOUR  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                      173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                      466, 388, 387, 386, 385, 384, 398]

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD       = 0.25   # below this → eye closing
DROWSY_FRAME_COUNT  = 20     # consecutive closed frames before alert
BLINK_MIN_FRAMES    = 2      # ignore blinks shorter than this
ATTENTION_DECAY     = 0.3    # how fast attention score drops per closed frame
ATTENTION_RECOVER   = 0.1    # how fast it recovers per open frame


def get_eye_coords(landmarks, indices, frame_w, frame_h):
    """Extract (x, y) pixel coordinates for given landmark indices."""
    return [
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
        for i in indices
    ]


def crop_eye(frame, landmarks, indices, frame_w, frame_h, pad=12):
    """Crop eye region from frame with padding."""
    pts = get_eye_coords(landmarks, indices, frame_w, frame_h)
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    x1  = max(0, min(xs) - pad)
    x2  = min(frame_w, max(xs) + pad)
    y1  = max(0, min(ys) - pad)
    y2  = min(frame_h, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
