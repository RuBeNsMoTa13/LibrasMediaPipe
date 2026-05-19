import cv2  # OpenCV para captura/desenho
import mediapipe as mp  # MediaPipe base
from mediapipe.tasks import python  # BaseOptions do Tasks
from mediapipe.tasks.python import vision  # GestureRecognizer
import numpy as np  # arrays e manipulacao de imagem
import gradio as gr  # UI
try:
    from fastrtc import WebRTC  # WebRTC no Gradio
except ImportError:
    from fastrtc.webrtc import WebRTC  # fallback para versoes antigas
import time  # tempo e FPS
import sys  # stdout
# OTIMIZAÇÕES CRÍTICAS PARA FPS:
# - Processar apenas 1 frame a cada N frames (major boost)
# - Downscale agressivo para detecção
# - Cache de resultados
# - Desenhar apenas quando há novos resultados
## Parametros de performance/qualidade
DOWNSCALE_WIDTH = 192  # Largura para deteccao (menor = mais rapido)
PROCESS_EVERY_N = 3   # Processa 1 a cada N frames
WEBRTC_DISPLAY_WIDTH = 640  # Largura visual da caixa
WEBRTC_INPUT_IS_BGR = False  # Cor esperada do frame do WebRTC
WEBRTC_COLOR_FALLBACK = True  # Tenta outra ordem de cor se falhar
frame_count = 0  # contador total de frames

last_fps_time = 0.0  # referencia para FPS
fps_frames = 0  # frames na janela de FPS
fps_text = "0.0 FPS"  # texto de FPS
LOG_EVERY_SECONDS = 1.0  # intervalo de log
last_log_time = 0.0  # referencia do log
log_window_frames = 0  # frames na janela de log
log_window_processed = 0  # processados na janela
log_window_frame_time = 0.0  # tempo total de frame
log_window_proc_time = 0.0  # tempo total de processamento
# Cache de resultados para frames que não processamos
cached_label = None  # ultimo label
cached_landmarks = None  # ultimos landmarks

# Caminho do modelo (coloque `gesture_recognizer.task` aqui ou ajuste)
MODEL_PATH = 'gesture_recognizer.task'  # arquivo do modelo

# Inicializar reconhecedor (modo VIDEO para processar frames de webcam)
recognizer = None  # instancia do recognizer
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)  # caminho do modelo
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO  # modo video (webcam)
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)  # cria reconhecedor
except Exception as e:
    print(f"Aviso: não foi possível inicializar o recognizer: {e}")
    print("Verifique se 'gesture_recognizer.task' existe ou se a versão do MediaPipe suporta Tasks API.")


def extract_top_gesture(result) -> tuple[str, float] | None:
    """Extrai o melhor gesto do resultado do MediaPipe."""
    gestures = getattr(result, "gestures", None)  # lista de gestos
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
    global last_fps_time, fps_frames
    global fps_text, frame_count, cached_label, cached_landmarks
    global last_log_time, log_window_frames, log_window_processed, log_window_frame_time, log_window_proc_time
    
    if image is None:
        return None

    if isinstance(image, tuple):
        image = image[0]
    elif isinstance(image, dict) and "frame" in image:
        image = image["frame"]

    now = time.time()  # tempo atual
    t0 = time.perf_counter()  # cronometro
    frame_count += 1  # incrementa frames
    log_window_frames += 1  # janela de log
    
    # OTIMIZAÇÃO 1: Processar apenas a cada N frames
    should_process = (frame_count % PROCESS_EVERY_N) == 0  # throttling
    
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    if WEBRTC_INPUT_IS_BGR:
        frame_bgr = image
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = image
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    h, w, _ = rgb_frame.shape  # dimensoes
    
    label_text = cached_label  # Usa resultado anterior por padrão
    debug_label = "Sem gesto"  # texto default
    hand_landmarks_list = cached_landmarks  # landmarks cacheados
    
    if should_process:
        t_proc_start = time.perf_counter()  # cronometro de processamento
        # Redimensionar AGRESSIVAMENTE para detecção
        small_w = DOWNSCALE_WIDTH if w > DOWNSCALE_WIDTH else w  # largura alvo
        scale = small_w / w  # escala
        small_h = max(1, int(h * scale))  # altura alvo
        small_rgb = cv2.resize(rgb_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)  # resize

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)  # MP image

        result = None  # resultado default
        if recognizer is not None:
            try:
                timestamp = int(now * 1000)
                result = recognizer.recognize_for_video(mp_image, timestamp)
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
        log_window_processed += 1  # contador de processados
        log_window_proc_time += time.perf_counter() - t_proc_start  # tempo proc
    
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
    fps_frames += 1  # conta frames
    log_window_frame_time += time.perf_counter() - t0  # tempo total
    elapsed = now - last_fps_time if last_fps_time else 0.0  # delta
    if last_fps_time == 0.0:
        last_fps_time = now
    elif elapsed >= 0.5:
        fps_text = f"{(fps_frames / elapsed):.1f} FPS"  # texto de FPS
        last_fps_time = now
        fps_frames = 0

    # Log em tempo real no terminal (uma linha atualizada)
    log_elapsed = now - last_log_time if last_log_time else 0.0  # delta log
    if last_log_time == 0.0:
        last_log_time = now
    elif log_elapsed >= LOG_EVERY_SECONDS:
        avg_frame_ms = (log_window_frame_time / max(log_window_frames, 1)) * 1000.0  # ms/frame
        avg_proc_ms = (log_window_proc_time / max(log_window_processed, 1)) * 1000.0  # ms/proc
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
    font = cv2.FONT_HERSHEY_SIMPLEX  # fonte
    (tw, th), _ = cv2.getTextSize(fps_text, font, 0.7, 1)
    cv2.putText(frame_bgr, fps_text, (w - 10 - tw, 10 + th), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    
    label_out = f"Sinal: {label_text}" if label_text else debug_label  # texto do gesto
    (lw, lh), _ = cv2.getTextSize(label_out, font, 0.8, 2)
    label_x = 10
    label_y = 30
    cv2.rectangle(frame_bgr, (label_x - 4, label_y - lh - 6), (label_x + lw + 6, label_y + 6), (0, 0, 0), -1)
    color = (0, 255, 0) if label_text else (0, 0, 255)  # cor do texto
    cv2.putText(frame_bgr, label_out, (label_x, label_y), font, 0.8, color, 2, cv2.LINE_AA)

    # Manter resolucao original da webcam no output

    output = frame_bgr if WEBRTC_INPUT_IS_BGR else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # saida
    return output


def main():
    """Entrada principal."""
    print(f"Recognizer inicializado: {recognizer is not None}")
    if recognizer is None:
        print("ERRO: Recognizer não foi inicializado!")
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
        gr.Markdown("<h1 id='page-header'>Detecção de LIBRAS - MediaPipe</h1>")  # titulo
        gr.Markdown("<p id='page-subtitle'>Webcam em tempo real com reconhecimento de gestos e landmarks</p>")  # subtitulo
        
        with gr.Row(elem_id='webrtc-wrap'):
            with gr.Column(elem_id='webrtc-box'):
                input_video = WebRTC(
                    label='Webcam',
                    mode='send-receive',
                    modality='video',
                    elem_id='webrtc'
                )  # componente WebRTC

        input_video.stream(fn=process_frame, inputs=input_video, outputs=input_video)  # stream

    demo.launch(server_name='127.0.0.1', server_port=7860)  # inicia server


if __name__ == '__main__':
    main()  # entrypoint
