import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import threading
import numpy as np

# --- CONFIGURAÇÕES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(ROOT_DIR / 'models' / 'gesture_recognizer.task')

recognized_text = ""  # texto acumulado com letras
last_seen_token = None  # ultimo token visto
last_legend_update_time = 0.0  # instante do ultimo sinal consolidado
LEGEND_CLEAR_SECONDS = 5  # limpa a legenda apos esse tempo sem novos sinais no modo automatico
LETTER_DELAY_SECONDS = 0.7  # intervalo minimo entre letras
last_letter_commit_time = 0.0  # instante da ultima letra adicionada
CONFIDENCE_THRESHOLD = 0.50  # limiar padrão ajustado para 50% (permite letras como U, V, etc.)

# Modo de voz: False = manual (apenas ao teclar Enter), True = automático (ao pausar por 5s)
AUTO_SPEAK = False

# Controle do modal de ajuda / atalhos
show_help = False

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

from PIL import Image, ImageDraw, ImageFont

# --- SISTEMA DE TIPOGRAFIA MODERNA E PALETA DE ALTO CONTRASTE ---
FONT_REGULAR_PATH = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/segoeuib.ttf"
FONT_FALLBACK_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_FALLBACK_BOLD = "C:/Windows/Fonts/arialbd.ttf"

USE_PIL_FONT = False
font_large_bold = None
font_med_bold = None
font_med = None
font_small = None

try:
    bold_path = FONT_BOLD_PATH if Path(FONT_BOLD_PATH).exists() else FONT_FALLBACK_BOLD
    reg_path = FONT_REGULAR_PATH if Path(FONT_REGULAR_PATH).exists() else FONT_FALLBACK_PATH
    font_large_bold = ImageFont.truetype(bold_path, 30)
    font_med_bold = ImageFont.truetype(bold_path, 22)
    font_med = ImageFont.truetype(reg_path, 18)
    font_small = ImageFont.truetype(reg_path, 16)
    USE_PIL_FONT = True
except Exception as e:
    print("Aviso: Falha ao carregar fonte TrueType, utilizando fontes padrão OpenCV:", e)
    USE_PIL_FONT = False

# Cores vibrantes em formato BGR para máxima legibilidade
COLOR_EMERALD = (118, 230, 0)      # Verde neon para sinais confirmados [OK]
COLOR_AMBER   = (0, 145, 255)      # Laranja vivo para sinais abaixo do limiar
COLOR_SILVER  = (220, 216, 207)    # Cinza prateado suave para "sem gesto"
COLOR_CYAN    = (255, 229, 0)      # Ciano elétrico para limiar e destaques
COLOR_GOLD    = (0, 215, 255)      # Dourado brilhante para modo manual / atalhos
COLOR_YELLOW  = (85, 238, 255)     # Amarelo puro ultra visível para a legenda soletrada
COLOR_WHITE   = (245, 245, 245)    # Branco gelo para textos e atalhos
COLOR_MUTED   = (165, 160, 155)    # Cinza médio para textos secundários


def draw_help_modal(frame):
    """Desenha um pop-up modal centralizado e responsivo em HD com tipografia nítida."""
    h, w = frame.shape[:2]
    
    # 1. Escurecer o fundo com transparência
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # 2. Caixa do modal centralizada e proporcional ao canvas HD
    modal_w = min(740, int(w * 0.90))
    modal_h = min(490, int(h * 0.88))
    x1 = (w - modal_w) // 2
    y1 = (h - modal_h) // 2
    x2 = x1 + modal_w
    y2 = y1 + modal_h

    # Fundo do card
    cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 20, 18), -1)
    # Borda externa ciano vibrante
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CYAN, 2)
    # Borda interna sutil
    cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (60, 52, 46), 1)

    # Linha divisória do cabeçalho
    cv2.line(frame, (x1 + 20, y1 + 52), (x2 - 20, y1 + 52), (70, 65, 60), 1)
    # Linha divisória inferior
    cv2.line(frame, (x1 + 20, y2 - 42), (x2 - 20, y2 - 42), (70, 65, 60), 1)

    shortcuts = [
        ("[Enter]", "Falar legenda acumulada (TTS) e limpar texto"),
        ("[V]", "Alternar modo de voz (Manual <-> Automático)"),
        ("[+] / [=]", "Aumentar limiar de confiança (+5%)"),
        ("[-] / [_]", "Diminuir limiar de confiança (-5%)"),
        ("[C]", "Limpar a legenda acumulada"),
        ("[D/Backsp]", "Apagar o último caractere digitado"),
        ("[Espaço]", "Inserir um espaço entre palavras"),
        ("[H]", "Abrir / fechar este manual de atalhos"),
        ("[Q]", "Fechar e sair da aplicação"),
    ]

    if USE_PIL_FONT:
        # Renderização via Pillow em HD com Segoe UI e acentuação nativa
        box_slice = frame[y1:y2, x1:x2]
        pil_box = Image.fromarray(box_slice)
        draw = ImageDraw.Draw(pil_box)

        # Cabeçalho
        title = "MANUAL DE ATALHOS - LIBRAS"
        bbox_t = font_large_bold.getbbox(title)
        tw = bbox_t[2] - bbox_t[0]
        draw.text(((modal_w - tw) // 2, 12), title, font=font_large_bold, fill=COLOR_CYAN)

        start_y = 68
        line_spacing = max(28, int((modal_h - 130) / len(shortcuts)))
        key_col_w = 160

        for i, (key_label, desc) in enumerate(shortcuts):
            cy = start_y + i * line_spacing
            draw.text((28, cy), key_label, font=font_med_bold, fill=COLOR_GOLD)
            draw.text((key_col_w, cy), f": {desc}", font=font_med, fill=COLOR_WHITE)

        footer = "Pressione [H] ou [ESC] para retornar ao vídeo"
        bbox_f = font_small.getbbox(footer)
        fw = bbox_f[2] - bbox_f[0]
        draw.text(((modal_w - fw) // 2, modal_h - 32), footer, font=font_small, fill=COLOR_MUTED)

        frame[y1:y2, x1:x2] = np.asarray(pil_box)
    else:
        # Fallback OpenCV
        title = "MANUAL DE ATALHOS - LIBRAS"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(frame, title, (x1 + (modal_w - tw) // 2, y1 + 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_CYAN, 2, cv2.LINE_AA)
        start_y = y1 + 80
        line_spacing = max(28, int((modal_h - 135) / len(shortcuts)))
        key_col_w = 160
        for i, (key_label, desc) in enumerate(shortcuts):
            cy = start_y + i * line_spacing
            cv2.putText(frame, key_label, (x1 + 25, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_GOLD, 1, cv2.LINE_AA)
            cv2.putText(frame, f": {desc}", (x1 + key_col_w, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
        footer = "Pressione [H] ou [ESC] para retornar ao video"
        (fw, _), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
        cv2.putText(frame, footer, (x1 + (modal_w - fw) // 2, y2 - 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, COLOR_MUTED, 1, cv2.LINE_AA)

# Inicializar o reconhecedor de gestos
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO # Otimizado para webcam
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Configuração da Janela Responsiva HD (1280x720)
WINDOW_NAME = 'LIBRAS - MediaPipe'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)
# Tentar configurar resolução HD se a câmera suportar (fallback automático)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando detecção...")
print("Janela redimensionável ativa. Arraste as bordas ou maximize a janela livremente.")
print("Pressione [H] na janela para abrir o Manual de Atalhos a qualquer momento.")
print("Atalhos disponíveis na janela:")
print("  [H]        : Abrir / Fechar Manual de Atalhos (Pop-up)")
print("  [Enter]    : Falar a legenda acumulada (TTS) e limpar")
print("  [V]        : Alternar modo de voz (Manual [Enter] <-> Automático)")
print("  [+] ou [=] : Aumentar limiar de confiança (+5%)")
print("  [-] ou [_] : Diminuir limiar de confiança (-5%)")
print("  [C]        : Limpar legenda acumulada")
print("  [D/Backsp] : Apagar última letra")
print("  [Espaço]   : Adicionar espaço")
print("  [Q]        : Sair")


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

    # Redimensionar o frame de exibição para alta definição HD (1280x720)
    # Isso evita serrilhamento de texto e garante linhas suaves de alta resolução
    DISPLAY_W = 1280
    DISPLAY_H = 720
    if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H:
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    h, w = frame.shape[:2]

    # Desenhar os landmarks da mão sobre o canvas HD
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Desenhar os pontos das articulações
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            
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
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Variáveis de estado do frame
    gesture_detected = False
    label = ""
    score = 0.0

    # Resetar token anterior quando a mão não estiver no enquadramento
    if not result.hand_landmarks:
        last_seen_token = None

    # Processar os gestos reconhecidos (pausa a digitação na legenda se o menu de ajuda estiver aberto)
    if result.gestures and not show_help:
        top_gesture = result.gestures[0][0]
        label = top_gesture.category_name
        score = top_gesture.score

        if label and label.lower() != "none":
            gesture_detected = True

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
            # Gesto classificado como "None" (posição neutra da mão)
            last_seen_token = None

    # Fala e limpeza automática por inatividade (apenas quando o modo automático estiver ativo)
    if AUTO_SPEAK and recognized_text and last_legend_update_time and (now_seconds - last_legend_update_time) >= LEGEND_CLEAR_SECONDS:
        if USE_TTS and recognized_text and recognized_text != last_spoken_text:
            speak_async(recognized_text)
            last_spoken_text = recognized_text
        recognized_text = ""
        last_seen_token = None
        last_legend_update_time = 0.0

    threshold_percent = int(CONFIDENCE_THRESHOLD * 100)

    # --- 1. BARRA SUPERIOR UNIFICADA (HUD de Status e Controles) ---
    top_bar_h = 76
    cv2.rectangle(frame, (0, 0), (w, top_bar_h), (20, 18, 15), -1)
    cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (55, 48, 42), 1)

    # Linha 1 Superior: Sinal à esquerda e Limiar à direita
    if gesture_detected:
        if score >= CONFIDENCE_THRESHOLD:
            current_label = f"Sinal: {label} ({score*100:.1f}%) [OK]"
            label_color = COLOR_EMERALD  # Verde neon vibrante
        else:
            current_label = f"Sinal: {label} ({score*100:.1f}%) [< {threshold_percent}%]"
            label_color = COLOR_AMBER  # Laranja vivo
    else:
        current_label = "Sinal: sem gesto"
        label_color = COLOR_SILVER

    thresh_text = f"Limiar: {threshold_percent}% [+/-]"
    sub_left = "Reconhecimento ativo" if not show_help else "MODO AJUDA ABERTO"
    top_shortcuts = "[H] Manual de Atalhos  |  [C] Limpar  |  [Q] Sair"

    if USE_PIL_FONT:
        top_slice = frame[0:top_bar_h, 0:w]
        pil_top = Image.fromarray(top_slice)
        draw_top = ImageDraw.Draw(pil_top)

        draw_top.text((20, 10), current_label, font=font_med_bold, fill=label_color)

        bbox_th = font_med_bold.getbbox(thresh_text)
        tw_th = bbox_th[2] - bbox_th[0]
        draw_top.text((w - tw_th - 20, 10), thresh_text, font=font_med_bold, fill=COLOR_CYAN)

        draw_top.text((20, 44), sub_left, font=font_small, fill=COLOR_MUTED)

        bbox_sc = font_small.getbbox(top_shortcuts)
        tw_sc = bbox_sc[2] - bbox_sc[0]
        draw_top.text((w - tw_sc - 20, 44), top_shortcuts, font=font_small, fill=COLOR_WHITE)

        frame[0:top_bar_h, 0:w] = np.asarray(pil_top)
    else:
        cv2.putText(frame, current_label, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, label_color, 1, cv2.LINE_AA)
        (tw_thresh, _), _ = cv2.getTextSize(thresh_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.putText(frame, thresh_text, (w - tw_thresh - 20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, sub_left, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.50, COLOR_MUTED, 1, cv2.LINE_AA)
        (tw_sc, _), _ = cv2.getTextSize(top_shortcuts, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        cv2.putText(frame, top_shortcuts, (w - tw_sc - 20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)

    # --- 2. BARRA INFERIOR UNIFICADA (Voz e Legenda Acumulada) ---
    bottom_bar_h = 86
    bottom_y_start = h - bottom_bar_h
    cv2.rectangle(frame, (0, bottom_y_start), (w, h), (20, 18, 15), -1)
    cv2.line(frame, (0, bottom_y_start), (w, bottom_y_start), (55, 48, 42), 1)

    if AUTO_SPEAK:
        voice_status = f"Voz: [AUTO - fala em {LEGEND_CLEAR_SECONDS}s]  (tecle V p/ Manual)"
        voice_color = COLOR_CYAN
    else:
        voice_status = "Voz: [MANUAL - tecle Enter p/ falar]  (tecle V p/ Auto)"
        voice_color = COLOR_GOLD

    edit_hints = "[D/Backsp] Apagar letra  |  [Espaço] Espaço"

    if USE_PIL_FONT:
        bot_slice = frame[bottom_y_start:h, 0:w]
        pil_bot = Image.fromarray(bot_slice)
        draw_bot = ImageDraw.Draw(pil_bot)

        draw_bot.text((20, 10), voice_status, font=font_small, fill=voice_color)

        bbox_eh = font_small.getbbox(edit_hints)
        tw_eh = bbox_eh[2] - bbox_eh[0]
        draw_bot.text((w - tw_eh - 20, 10), edit_hints, font=font_small, fill=COLOR_MUTED)

        # Legenda com grande destaque
        draw_bot.text((20, 42), "Legenda:", font=font_large_bold, fill=COLOR_CYAN)
        bbox_leg = font_large_bold.getbbox("Legenda: ")
        leg_offset = 20 + (bbox_leg[2] - bbox_leg[0])
        if recognized_text:
            draw_bot.text((leg_offset, 42), recognized_text, font=font_large_bold, fill=COLOR_YELLOW)
        else:
            draw_bot.text((leg_offset, 46), "(aguardando sinal...)", font=font_med, fill=COLOR_MUTED)

        frame[bottom_y_start:h, 0:w] = np.asarray(pil_bot)
    else:
        cv2.putText(frame, voice_status, (20, h - 50), cv2.FONT_HERSHEY_DUPLEX, 0.55, voice_color, 1, cv2.LINE_AA)
        (tw_eh, _), _ = cv2.getTextSize(edit_hints, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
        cv2.putText(frame, edit_hints, (w - tw_eh - 20, h - 50), cv2.FONT_HERSHEY_DUPLEX, 0.50, COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, "Legenda: ", (20, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.85, COLOR_CYAN, 2, cv2.LINE_AA)
        (leg_tw, _), _ = cv2.getTextSize("Legenda: ", cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        if recognized_text:
            cv2.putText(frame, recognized_text, (20 + leg_tw, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.90, COLOR_YELLOW, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "(aguardando sinal...)", (20 + leg_tw, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.70, COLOR_MUTED, 1, cv2.LINE_AA)

    # Desenhar o pop-up do Manual de Atalhos sobre o frame se ativado
    if show_help:
        draw_help_modal(frame)

    # Exibir o vídeo na janela responsiva
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('h'), ord('H')):  # Alternar pop-up de ajuda
        show_help = not show_help
        print(f"Manual de atalhos: {'Aberto' if show_help else 'Fechado'}")
    elif key == 27:  # Tecla ESC fecha a ajuda se estiver aberta
        if show_help:
            show_help = False
            print("Manual de atalhos fechado.")
    elif key in (13, 10):  # Tecla Enter (aciona fala manual da legenda)
        if recognized_text:
            if USE_TTS:
                speak_async(recognized_text)
                last_spoken_text = recognized_text
                print(f"Voz disparada (Enter): '{recognized_text}'")
            else:
                print(f"Texto da legenda: '{recognized_text}' (pyttsx3 não disponível)")
            recognized_text = ""
            last_seen_token = None
            last_legend_update_time = 0.0
        else:
            print("Legenda vazia para falar.")
    elif key in (ord('v'), ord('V')):  # Alternar entre modo Manual e Automático
        AUTO_SPEAK = not AUTO_SPEAK
        mode_str = f"AUTOMÁTICO (fala após {LEGEND_CLEAR_SECONDS}s de inatividade)" if AUTO_SPEAK else "MANUAL (fala apenas ao teclar Enter)"
        print(f"Modo de voz alternado para: {mode_str}")
    elif key in (ord('+'), ord('=')):
        CONFIDENCE_THRESHOLD = min(0.95, round(CONFIDENCE_THRESHOLD + 0.05, 2))
        print(f"Limiar de confiança ajustado para: {int(CONFIDENCE_THRESHOLD * 100)}%")
    elif key in (ord('-'), ord('_')):
        CONFIDENCE_THRESHOLD = max(0.20, round(CONFIDENCE_THRESHOLD - 0.05, 2))
        print(f"Limiar de confiança ajustado para: {int(CONFIDENCE_THRESHOLD * 100)}%")
    elif key in (ord('c'), ord('C')):
        recognized_text = ""
        last_seen_token = None
        last_legend_update_time = 0.0
        print("Legenda limpa manualmente.")
    elif key == 8 or key in (ord('d'), ord('D')):  # Backspace ou 'd' para apagar
        if recognized_text:
            recognized_text = recognized_text[:-1]
            last_seen_token = None
            last_legend_update_time = now_seconds
            print(f"Caractere apagado. Legenda atual: '{recognized_text}'")
    elif key == 32:  # Barra de espaço
        if recognized_text and not recognized_text.endswith(" "):
            recognized_text += " "
            last_legend_update_time = now_seconds

cap.release()
cv2.destroyAllWindows()