# Nombre: Jose Alberto
# Matrícula: 23-MISN-2-013
# Archivo: dataset_builder.py
# Descripción: Graba landmarks desde la webcam y guarda dataset_gestos.npz
# Uso: python dataset_builder.py

import cv2
import numpy as np
from detection import HandDetector
import os

detector = HandDetector()
cap = cv2.VideoCapture(0)
labels_map = {
    0: "idle",
    1: "left",
    2: "right",
    3: "grab"
}
X = []
y = []

print("Instrucciones:")
print("Presiona la tecla numérica para comenzar a grabar cada clase:")
for k,v in labels_map.items():
    print(f"{k}: {v}")
print("Presiona 'q' para salir y guardar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out, vis = detector.process(frame)
    display = cv2.flip(vis, 1)
    cv2.imshow("Dataset builder (presiona 0-3 para etiquetar, q para salir)", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
        label = int(chr(key))
        # Captura varias muestras rápidas
        for i in range(12):
            ret2, frame2 = cap.read()
            if not ret2:
                continue
            r, f2 = detector.process(frame2)
            if r is not None:
                px, py, coords = r
                X.append(coords)
                y.append(label)
        print(f"Añadidas ~12 muestras para {labels_map[label]}")

cap.release()
cv2.destroyAllWindows()

X = np.stack(X) if len(X)>0 else np.zeros((0,42), dtype=np.float32)
y = np.array(y, dtype=np.int64)
np.savez_compressed("dataset_gestos.npz", X=X, y=y)
print("Guardado dataset_gestos.npz con", len(y), "muestras.")
