import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'gesture_recognizer.task' # Certifique-se que o nome está correto

# Inicializar o reconhecedor de gestos
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO # Otimizado para webcam
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

print("Iniciando detecção... Pressione 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverter a imagem para parecer um espelho
    frame = cv2.flip(frame, 1)
    
    # Converter BGR (OpenCV) para RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Obter o timestamp atual em milissegundos
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Realizar a detecção
    result = recognizer.recognize_for_video(mp_image, timestamp)

    # Desenhar os landmarks da mão
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            # Desenhar os pontos e conexões
            for i, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Desenhar as conexões dos dedos
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
                (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
                (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
                (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
                (0, 17), (17, 18), (18, 19), (19, 20)  # Mínimo
            ]
            for start, end in connections:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar o resultado na tela
    if result.gestures:
        # Pega o gesto com maior confiança
        top_gesture = result.gestures[0][0]
        label = top_gesture.category_name
        score = top_gesture.score

        # Desenhar o texto no frame
        text = f"Sinal: {label} ({score*100:.1f}%)"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibir o vídeo
    cv2.imshow('LIBRAS - MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()