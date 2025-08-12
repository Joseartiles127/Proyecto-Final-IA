# Nombre: Jose Alberto Artiles - Matricula: 23-MISN-2-013
# Juego controlado mediante la cámara

import gradio as gr
import cv2
import mediapipe as mp
from detection import detect_hand_gesture
from PIL import Image

# Cargar imágenes del juego
fondo = Image.open("resources/virtual-en-superficie-oscura.jpg")

def jugar(frame):
    gesto = detect_hand_gesture(frame)
    
    # Aquí va la lógica para mover objetos del juego según el gesto
    if gesto == "left":
        accion = "Mover izquierda"
    elif gesto == "right":
        accion = "Mover derecha"
    else:
        accion = "Quieto"
    
    return frame, f"Gesto detectado: {accion}"

iface = gr.Interface(
    fn=jugar,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=["image", "text"],
    live=True,
    title="Juego controlado mediante la cámara",
    description="Controla el juego moviendo tus manos frente a la cámara."
)

if __name__ == "__main__":
    iface.launch()
