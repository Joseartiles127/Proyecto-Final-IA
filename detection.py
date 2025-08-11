import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def detect_hand_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            if max(x_coords) < 0.4:
                return "left"
            elif min(x_coords) > 0.6:
                return "right"
    return "none"
