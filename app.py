import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import gradio as gr
from fastrtc import WebRTC
import time
import threading
import queue
import sys
# OTIMIZAÇÕES CRÍTICAS PARA FPS:
# - Processar apenas 1 frame a cada N frames (major boost)
# - Downscale agressivo para detecção
# - Cache de resultados
# - Desenhar apenas quando há novos resultados
DOWNSCALE_WIDTH = 192  # Menos agressivo para melhorar o reconhecimento
PROCESS_EVERY_N = 3   # Processa 1 a cada 3 frames
DRAW_EVERY_N = 1       # Desenha mais frequentemente (usa cache)
GRADIO_OUTPUT_WIDTH = None
WEBRTC_DISPLAY_WIDTH = 640
WEBRTC_INPUT_IS_BGR = False
WEBRTC_COLOR_FALLBACK = True
frame_count = 0
process_frame_count = 0
USE_DRAWING_UTILS = False
RUN_WITH_GRADIO = True

last_gradio_output = None
last_gradio_process_time = 0.0
last_fps_time = 0.0
fps_frames = 0
fps_text = "0.0 FPS"
LOG_EVERY_SECONDS = 1.0
last_log_time = 0.0
log_window_frames = 0
log_window_processed = 0
log_window_frame_time = 0.0
log_window_proc_time = 0.0
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


def extract_top_gesture(result) -> tuple[str, float] | None:
    gestures = getattr(result, "gestures", None)
    if not gestures:
        return None
    first = gestures[0]
    if hasattr(first, "categories"):
        categories = first.categories
    elif isinstance(first, list):
        categories = first
    else:
        categories = None
    if not categories:
        return None
    top = categories[0]
    return top.category_name, top.score


def process_frame(image: np.ndarray) -> np.ndarray:
    """Recebe imagem (HWC), retorna imagem anotada no mesmo formato de entrada."""
    global last_gradio_output, last_gradio_process_time, last_fps_time, fps_frames
    global fps_text, frame_count, process_frame_count, cached_result, cached_label, cached_landmarks
    global last_log_time, log_window_frames, log_window_processed, log_window_frame_time, log_window_proc_time
    
    if image is None:
        return None

    if isinstance(image, tuple):
        image = image[0]
    elif isinstance(image, dict) and "frame" in image:
        image = image["frame"]

    now = time.time()
    t0 = time.perf_counter()
    frame_count += 1
    log_window_frames += 1
    
    # OTIMIZAÇÃO 1: Processar apenas a cada N frames
    should_process = (frame_count % PROCESS_EVERY_N) == 0
    
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    if WEBRTC_INPUT_IS_BGR:
        frame_bgr = image
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = image
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    h, w, _ = rgb_frame.shape
    
    label_text = cached_label  # Usa resultado anterior por padrão
    debug_label = "Sem gesto"
    hand_landmarks_list = cached_landmarks
    
    if should_process:
        t_proc_start = time.perf_counter()
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
            best = extract_top_gesture(result)
            if WEBRTC_COLOR_FALLBACK:
                alt_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                alt_small_w = DOWNSCALE_WIDTH if w > DOWNSCALE_WIDTH else w
                alt_scale = alt_small_w / w
                alt_small_h = max(1, int(h * alt_scale))
                alt_small_rgb = cv2.resize(alt_rgb, (alt_small_w, alt_small_h), interpolation=cv2.INTER_LINEAR)
                alt_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=alt_small_rgb)
                try:
                    alt_result = recognizer.recognize_for_video(alt_mp_image, timestamp)
                    alt_best = extract_top_gesture(alt_result)
                    if alt_best and (not best or alt_best[1] > best[1]):
                        best = alt_best
                except Exception:
                    pass

            if best:
                label_text = f"{best[0]} ({best[1]*100:.1f}%)"
                cached_label = label_text
                debug_label = f"Gesto: {best[0]}"
            else:
                cached_label = None
                label_text = None
                debug_label = "Sem gesto"
            
            if getattr(result, 'hand_landmarks', None):
                hand_landmarks_list = result.hand_landmarks
                cached_landmarks = hand_landmarks_list
            else:
                cached_landmarks = None
                hand_landmarks_list = None
        else:
            cached_landmarks = None
            hand_landmarks_list = None
        log_window_processed += 1
        log_window_proc_time += time.perf_counter() - t_proc_start
    
    # RENDERIZAR frame (sempre, com dados cacheados)
    
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
    log_window_frame_time += time.perf_counter() - t0
    elapsed = now - last_fps_time if last_fps_time else 0.0
    if last_fps_time == 0.0:
        last_fps_time = now
    elif elapsed >= 0.5:
        fps_text = f"{(fps_frames / elapsed):.1f} FPS"
        last_fps_time = now
        fps_frames = 0
        process_frame_count = 0

    # Log em tempo real no terminal (uma linha atualizada)
    log_elapsed = now - last_log_time if last_log_time else 0.0
    if last_log_time == 0.0:
        last_log_time = now
    elif log_elapsed >= LOG_EVERY_SECONDS:
        avg_frame_ms = (log_window_frame_time / max(log_window_frames, 1)) * 1000.0
        avg_proc_ms = (log_window_proc_time / max(log_window_processed, 1)) * 1000.0
        sys.stdout.write(
            f"fps={fps_text} | frame_ms={avg_frame_ms:.1f} | proc_ms={avg_proc_ms:.1f} | processed={log_window_processed}/{log_window_frames}\n"
        )
        sys.stdout.flush()
        last_log_time = now
        log_window_frames = 0
        log_window_processed = 0
        log_window_frame_time = 0.0
        log_window_proc_time = 0.0

    # Desenhar FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(fps_text, font, 0.7, 1)
    cv2.putText(frame_bgr, fps_text, (w - 10 - tw, 10 + th), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    
    label_out = f"Sinal: {label_text}" if label_text else debug_label
    (lw, lh), _ = cv2.getTextSize(label_out, font, 0.8, 2)
    label_x = 10
    label_y = 30
    cv2.rectangle(frame_bgr, (label_x - 4, label_y - lh - 6), (label_x + lw + 6, label_y + 6), (0, 0, 0), -1)
    color = (0, 255, 0) if label_text else (0, 0, 255)
    cv2.putText(frame_bgr, label_out, (label_x, label_y), font, 0.8, color, 2, cv2.LINE_AA)

    # Manter resolucao original da webcam no output

    output = frame_bgr if WEBRTC_INPUT_IS_BGR else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
    with gr.Blocks(
        title='Detecção de LIBRAS - MediaPipe',
        css=(
            "@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Fraunces:opsz,wght@9..144,600&display=swap');"
            ":root {--bg1: #f7f2e8; --bg2: #e7f0ff; --ink: #1e1e1e; --muted: #5a5a5a; --card: #ffffff;}"
            "body {background: radial-gradient(1200px 600px at 20% 10%, var(--bg2), transparent),"
            "linear-gradient(180deg, var(--bg1), #ffffff 55%); color: var(--ink);}"
            ".gradio-container {max-width: 920px; margin: 0 auto; padding: 24px 16px 40px;}"
            "#page-header {text-align: center; margin: 10px 0 6px; font-family: 'Fraunces', serif;"
            "font-size: 34px; letter-spacing: 0.3px;}"
            "#page-subtitle {text-align: center; margin: 0 0 22px; color: var(--muted);"
            "font-family: 'Space Grotesk', sans-serif; font-size: 15px;}"
            "#webrtc-wrap {display: flex; justify-content: center;}"
            f"#webrtc-box {{width: min(92vw, {WEBRTC_DISPLAY_WIDTH}px); padding: 16px; border-radius: 18px;"
            "background: var(--card); box-shadow: 0 12px 30px rgba(28, 45, 80, 0.12);"
            "border: 1px solid rgba(30, 30, 30, 0.06);}}"
            "#webrtc {width: 100%; position: relative; overflow: hidden;}"
            "#webrtc video, #webrtc canvas {width: 100%; height: auto; border-radius: 12px;"
            "position: static !important; max-width: 100% !important; object-fit: contain;}"
            "#webrtc .label, #webrtc label, .gradio-label {font-family: 'Space Grotesk', sans-serif;}"
        )
    ) as demo:
        gr.Markdown("<h1 id='page-header'>Detecção de LIBRAS - MediaPipe</h1>")
        gr.Markdown("<p id='page-subtitle'>Webcam em tempo real com reconhecimento de gestos e landmarks</p>")
        
        with gr.Row(elem_id='webrtc-wrap'):
            with gr.Column(elem_id='webrtc-box'):
                input_video = WebRTC(
                    label='Webcam',
                    mode='send-receive',
                    modality='video',
                    elem_id='webrtc'
                )

        input_video.stream(fn=process_frame, inputs=input_video, outputs=input_video)

    demo.launch(server_name='127.0.0.1', server_port=7860)


if __name__ == '__main__':
    main()
