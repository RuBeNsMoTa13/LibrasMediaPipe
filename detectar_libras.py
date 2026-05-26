import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'gesture_recognizer.task' # Certifique-se que o nome está correto
recognized_text = ""  # texto acumulado com letras
last_seen_token = None  # ultimo token visto
last_legend_update_time = 0.0  # instante do ultimo sinal consolidado
LEGEND_CLEAR_SECONDS = 3 # limpa a legenda apos esse tempo sem novos sinais
LETTER_DELAY_SECONDS = 0.7  # intervalo minimo entre letras
last_letter_commit_time = 0.0  # instante da ultima letra adicionada
CONFIDENCE_THRESHOLD = 0.75  # só registrar contexto se a inferência >= 75%

# TTS (texto-para-fala) opcional - usa pyttsx3 se instalado
USE_TTS = False
try:
    import pyttsx3
    USE_TTS = True
except Exception:
    USE_TTS = False

last_spoken_text = ""

def speak_async(text: str):
    """Executa TTS de forma assíncrona para não bloquear o loop principal."""
    if not USE_TTS or not text:
        return

    def worker(tts_text: str):
        try:
            engine = pyttsx3.init()
            engine.say(tts_text)
            engine.runAndWait()
        except Exception as e:
            print("Aviso: falha no TTS:", e)

    threading.Thread(target=worker, args=(text,), daemon=True).start()
# (LETTER_DELAY_SECONDS e last_letter_commit_time já definidos acima)

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


def update_recognized_text(token, current_text, previous_seen, now_seconds, previous_commit_time):
    """Atualiza o texto acumulado quando um novo token estável aparece."""
    if token is None:
        return current_text, None, previous_commit_time, False

    normalized = token.strip().lower()
    if not normalized:
        return current_text, previous_seen, previous_commit_time, False

    if normalized == previous_seen:
        return current_text, previous_seen, previous_commit_time, False

    if normalized in {"space", "espaco", "espaço"}:
        if current_text and not current_text.endswith(" "):
            current_text += " "
            return current_text, normalized, previous_commit_time, True
        return current_text, normalized, previous_commit_time, False

    if normalized in {"del", "delete", "apagar", "backspace"}:
        new_text = current_text[:-1]
        return new_text, normalized, previous_commit_time, new_text != current_text

    if len(normalized) == 1 and normalized.isalpha():
        if (now_seconds - previous_commit_time) < LETTER_DELAY_SECONDS:
            # Ignora por enquanto sem travar a mesma letra para futuras tentativas.
            return current_text, previous_seen, previous_commit_time, False
        current_text += normalized
        return current_text, normalized, now_seconds, True

    return current_text, normalized, previous_commit_time, False

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

    now_seconds = timestamp / 1000.0

    # Realizar a detecção
    result = recognizer.recognize_for_video(mp_image, timestamp)

    # Desenhar os landmarks da mão
    if result.hand_landmarks:
        h, w = frame.shape[:2]
        for hand_landmarks in result.hand_landmarks:
            # Desenhar os pontos e conexões
            for landmark in hand_landmarks:
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

        # Atualiza contexto apenas se confiança suficiente
        if score >= CONFIDENCE_THRESHOLD:
            recognized_text, last_seen_token, last_letter_commit_time, text_updated = update_recognized_text(
                label,
                recognized_text,
                last_seen_token,
                now_seconds,
                last_letter_commit_time,
            )
            if text_updated:
                last_legend_update_time = now_seconds
        else:
            text_updated = False

        # (não falar aqui) — vamos falar somente quando a legenda expirar (inatividade)

    if recognized_text and last_legend_update_time and (now_seconds - last_legend_update_time) >= LEGEND_CLEAR_SECONDS:
        # Quando houver inatividade suficiente, falar o último contexto antes de limpar
        if USE_TTS and recognized_text and recognized_text != last_spoken_text:
            speak_async(recognized_text)
            last_spoken_text = recognized_text
        recognized_text = ""
        last_seen_token = None
        last_legend_update_time = 0.0

    current_label = f"Sinal: {label} ({score*100:.1f}%)" if result.gestures else "Sinal: sem gesto"
    bottom_label = f"Legenda: {recognized_text}" if recognized_text else "Legenda:"

    # Desenhar o sinal atual no topo
    cv2.putText(frame, current_label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Desenhar a legenda acumulada embaixo
    size = cv2.getTextSize(bottom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    (bw, bh) = size[0]
    bottom_x = 20
    bottom_y = frame.shape[0] - 20
    cv2.rectangle(frame, (bottom_x - 4, bottom_y - bh - 6), (bottom_x + bw + 6, bottom_y + 6), (0, 0, 0), -1)
    cv2.putText(frame, bottom_label, (bottom_x, bottom_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Exibir o vídeo
    cv2.imshow('LIBRAS - MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()