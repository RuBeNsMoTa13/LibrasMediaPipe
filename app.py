import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import gradio as gr

# Caminho do modelo (coloque `gesture_recognizer.task` aqui ou ajuste)
MODEL_PATH = 'gesture_recognizer.task'

# Inicializar reconhecedor (modo IMAGE para processar frames individuais)
recognizer = None
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Aviso: não foi possível inicializar o recognizer: {e}")
    print("Verifique se 'gesture_recognizer.task' existe ou se a versão do MediaPipe suporta Tasks API.")


def process_frame(image: np.ndarray) -> np.ndarray:
    """Recebe imagem RGB (HWC, uint8), retorna imagem anotada RGB."""
    if image is None:
        return None

    # Gradio fornece imagem em RGB
    rgb_frame = image

    # Criar mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Fazer a predição (modo IMAGE usa `recognize`)
    result = None
    if recognizer is not None:
        try:
            result = recognizer.recognize(mp_image)
        except Exception:
            # fallback para recognize_for_video com timestamp 0
            try:
                result = recognizer.recognize_for_video(mp_image, 0)
            except Exception:
                result = None

    # Desenhar sobre a imagem (usar BGR para OpenCV)
    frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    h, w, _ = frame_bgr.shape

    if result and getattr(result, 'hand_landmarks', None):
        for hand_landmarks in result.hand_landmarks:
            # pontos
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)

            # conexões simples (padrão 0-20)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            for start, end in connections:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar gesto detectado (top-1)
    if result and getattr(result, 'gestures', None):
        try:
            top_gesture = result.gestures[0][0]
            label = top_gesture.category_name
            score = top_gesture.score
            text = f"{label} ({score*100:.1f}%)"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass

    # Converter de volta para RGB para o Gradio
    annotated = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return annotated


def main():
    iface = gr.Interface(
        fn=process_frame,
        inputs=gr.Image(sources=['webcam'], type='numpy'),
        outputs='image',
        live=True,
        title='Detecção de LIBRAS - MediaPipe',
        description='Webcam -> reconhece gestos e desenha landmarks.'
    )
    iface.launch(server_name='0.0.0.0', server_port=7860)


if __name__ == '__main__':
    main()
