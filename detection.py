# Nombre: Jose Alberto Artiles
# Matrícula: 23-MISN-2-013
# Archivo: detection.py
# Descripción: Detección de manos con MediaPipe y utilidades para landmarks.

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, max_num_hands=1, detection_conf=0.6, track_conf=0.6):
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf
        )

    def process(self, frame):
        """
        Input: BGR frame (OpenCV)
        Output: (px, py, coords) or None, and the frame with landmarks drawn (BGR)
        coords: numpy array shape (42,) with (x1,y1,x2,y2,...normalized)
        """
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None, frame
        lm = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        px = int(lm.landmark[0].x * w)
        py = int(lm.landmark[0].y * h)
        coords = []
        for p in lm.landmark:
            coords.append(p.x)
            coords.append(p.y)
        coords = np.array(coords, dtype=np.float32)
        return (px, py, coords), frame
