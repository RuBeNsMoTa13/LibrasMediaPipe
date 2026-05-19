import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import gradio as gr
import time
import threading
import queue
import sys
# OTIMIZAÇÕES CRÍTICAS PARA FPS:
# - Processar apenas 1 frame a cada N frames (major boost)
# - Downscale agressivo para detecção
# - Cache de resultados
# - Desenhar apenas quando há novos resultados
DOWNSCALE_WIDTH = 96  # Aumentado de 160 -> 96 (4x mais rápido!)
PROCESS_EVERY_N = 3    # Processa 1 a cada 3 frames
DRAW_EVERY_N = 1       # Desenha mais frequentemente (usa cache)
GRADIO_OUTPUT_WIDTH = 360
frame_count = 0
process_frame_count = 0
USE_DRAWING_UTILS = False
RUN_WITH_GRADIO = True

last_gradio_output = None
last_gradio_process_time = 0.0
last_fps_time = 0.0
fps_frames = 0
fps_text = "0.0 FPS"
latest_overlay = {
    "hand_landmarks": None,
    "label_text": None,
    "fps_text": "0.0 FPS",
    "timestamp": 0,
}
overlay_lock = threading.Lock()

# Cache de resultados para frames que não processamos
cached_result = None
cached_label = None
cached_landmarks = None

# Caminho do modelo (coloque `gesture_recognizer.task` aqui ou ajuste)
MODEL_PATH = 'gesture_recognizer.task'

# Inicializar reconhecedor (modo VIDEO para processar frames de webcam)
recognizer = None
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Aviso: não foi possível inicializar o recognizer: {e}")
    print("Verifique se 'gesture_recognizer.task' existe ou se a versão do MediaPipe suporta Tasks API.")


def process_frame(image: np.ndarray) -> np.ndarray:
    """Recebe imagem RGB (HWC, uint8), retorna imagem anotada RGB."""
    global last_gradio_output, last_gradio_process_time, last_fps_time, fps_frames
    global fps_text, frame_count, process_frame_count, cached_result, cached_label, cached_landmarks
    
    if image is None:
        return None

    now = time.time()
    frame_count += 1
    
    # OTIMIZAÇÃO 1: Processar apenas a cada N frames
    should_process = (frame_count % PROCESS_EVERY_N) == 0
    
    rgb_frame = image
    h, w, _ = rgb_frame.shape
    
    label_text = cached_label  # Usa resultado anterior por padrão
    hand_landmarks_list = cached_landmarks
    
    if should_process:
        # Redimensionar AGRESSIVAMENTE para detecção
        small_w = DOWNSCALE_WIDTH if w > DOWNSCALE_WIDTH else w
        scale = small_w / w
        small_h = max(1, int(h * scale))
        small_rgb = cv2.resize(rgb_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)

        result = None
        if recognizer is not None:
            try:
                timestamp = int(now * 1000)
                result = recognizer.recognize_for_video(mp_image, timestamp)
                process_frame_count += 1
            except Exception as e:
                print(f"Aviso: recognize_for_video falhou: {e}")

        # Atualizar cache
        if result:
            if getattr(result, 'gestures', None):
                try:
                    top_gesture = result.gestures[0][0]
                    label_text = f"{top_gesture.category_name} ({top_gesture.score*100:.1f}%)"
                    cached_label = label_text
                except Exception:
                    pass
            
            if getattr(result, 'hand_landmarks', None):
                hand_landmarks_list = result.hand_landmarks
                cached_landmarks = hand_landmarks_list
    
    # RENDERIZAR frame (sempre, com dados cacheados)
    frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Desenhar landmarks cacheados
    if hand_landmarks_list:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        for hand_landmarks in hand_landmarks_list:
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)
            for start, end in connections:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Calcular FPS
    fps_frames += 1
    elapsed = now - last_fps_time if last_fps_time else 0.0
    if last_fps_time == 0.0:
        last_fps_time = now
    elif elapsed >= 0.5:
        fps_text = f"{(fps_frames / elapsed):.1f} FPS (process: {process_frame_count}/{frame_count})"
        last_fps_time = now
        fps_frames = 0
        process_frame_count = 0

    # Desenhar FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(fps_text, font, 0.7, 1)
    cv2.putText(frame_bgr, fps_text, (w - 10 - tw, 10 + th), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    
    if label_text:
        cv2.putText(frame_bgr, label_text, (10, h - 10), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    # Redimensionar saída (reduzir overhead de renderização)
    out_h, out_w = frame_bgr.shape[:2]
    if out_w > GRADIO_OUTPUT_WIDTH:
        out_scale = GRADIO_OUTPUT_WIDTH / out_w
        out_h = max(1, int(out_h * out_scale))
        frame_bgr = cv2.resize(frame_bgr, (GRADIO_OUTPUT_WIDTH, out_h), interpolation=cv2.INTER_AREA)

    output = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    last_gradio_output = output
    return output


def render_frame(frame_bgr: np.ndarray, result) -> np.ndarray:
    """Desenha landmarks/textos sobre um frame BGR e devolve BGR."""
    h, w, _ = frame_bgr.shape

    # Usar cache global
    if cached_landmarks:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        for hand_landmarks in cached_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)
            for start, end in connections:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fps_scale = 0.7
    fps_thickness = 1
    (tw, th), _ = cv2.getTextSize("0.00 FPS", font, fps_scale, fps_thickness)
    margin = 10
    fx = w - margin - tw
    fy = margin + th
    
    # Desenhar FPS
    cv2.putText(frame_bgr, fps_text, (fx, fy), font, fps_scale, (0, 255, 255), fps_thickness, cv2.LINE_AA)

    if cached_label:
        cv2.putText(frame_bgr, cached_label, (10, h - 10), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    return frame_bgr


def main():
    print(f"Recognizer inicializado: {recognizer is not None}")
    if recognizer is None:
        print("ERRO: Recognizer não foi inicializado!")

    if not RUN_WITH_GRADIO:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERRO: não foi possível abrir a webcam.")
            sys.exit(1)

        # tenta reduzir a latência da webcam
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        last_fps_time = time.time()
        fps_frames = 0
        last_fps_text = "0.0 FPS"
        last_label_text = None
        last_landmarks = None
        last_draw_frame = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_bgr = frame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            
            # OTIMIZAÇÃO: Processar apenas cada N frames
            should_process = (frame_idx % PROCESS_EVERY_N) == 0
            
            if should_process:
                small_w = DOWNSCALE_WIDTH if w > DOWNSCALE_WIDTH else w
                scale = small_w / w
                small_h = max(1, int(h * scale))
                small_rgb = cv2.resize(frame_rgb, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
                result = None
                if recognizer is not None:
                    try:
                        timestamp = int(time.time() * 1000)
                        result = recognizer.recognize_for_video(mp_image, timestamp)
                    except Exception as e:
                        print(f"Aviso(opencv): recognize_for_video falhou: {e}")

                if result and getattr(result, 'gestures', None):
                    try:
                        top_gesture = result.gestures[0][0]
                        last_label_text = f"{top_gesture.category_name} ({top_gesture.score*100:.1f}%)"
                    except Exception:
                        pass

                if result and getattr(result, 'hand_landmarks', None):
                    last_landmarks = result.hand_landmarks
            
            fps_frames += 1
            now = time.time()
            elapsed = now - last_fps_time
            if elapsed >= 0.5:
                last_fps_text = f"{fps_frames / elapsed:.1f} FPS"
                fps_frames = 0
                last_fps_time = now

            shown = render_frame(frame_bgr.copy(), None)
            cv2.imshow('Detecção de LIBRAS - MediaPipe', shown)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return
    with gr.Blocks(title='Detecção de LIBRAS - MediaPipe') as demo:
        gr.Markdown("# Detecção de LIBRAS - MediaPipe")
        gr.Markdown("Webcam em tempo real com reconhecimento de gestos e landmarks")
        
        with gr.Row():
            input_video = gr.Image(sources=['webcam'], type='numpy', label='Webcam', streaming=True)
            output_video = gr.Image(label='Resultado', type='numpy')
        
        input_video.stream(
            fn=process_frame,
            inputs=input_video,
            outputs=output_video
        )

    demo.launch(server_name='127.0.0.1', server_port=7860)


if __name__ == '__main__':
    main()
