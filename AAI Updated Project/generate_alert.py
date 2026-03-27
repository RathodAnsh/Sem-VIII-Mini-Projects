"""
generate_alert.py
─────────────────
Generates the alert beep sound file.
Run once: python generate_alert.py
"""

import os
import numpy as np
from scipy.io.wavfile import write


def generate_alert_sound(path: str = "sounds/alert.wav"):
    os.makedirs("sounds", exist_ok=True)
    sample_rate = 44100
    duration    = 0.5    # seconds per beep
    freqs       = [880, 1100, 880, 1100]  # alternating tones
    silence     = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
    chunks      = []

    for freq in freqs:
        t     = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        beep  = np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Apply fade in/out to avoid clicking
        fade  = int(sample_rate * 0.02)
        beep[:fade]  *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        chunks.append(beep)
        chunks.append(silence)

    wave = (np.concatenate(chunks) * 32767).astype(np.int16)
    write(path, sample_rate, wave)
    print(f"Alert sound saved → {path}")


if __name__ == "__main__":
    generate_alert_sound()
