import argparse
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import threading
import time
import numpy as np

# --- CONFIGURAÇÕES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(ROOT_DIR / 'models' / 'gesture_recognizer.task')

# --- ESCOLHA DO CLASSIFICADOR ---
# task: o gesture_recognizer.task classifica a letra (padrão)
# snn, rf, svm: o MediaPipe extrai a mão, a rede de dentro do .task faz o resumo de
#   128 números (classificadores/resumo.py) e o modelo escolhido classifica.
#   Os três são treinados e salvos em models/ por src/treino/treinar_modelos.py.
# A tecla M passa pelos modelos disponíveis nesta ordem.
MODEL_NAMES = {
    "task": "MediaPipe .task",
    "snn": "SNN",
    "rf": "Random Forest",
    "svm": "SVM",
}
parser = argparse.ArgumentParser(description="Reconhecimento do alfabeto manual de LIBRAS na webcam")
parser.add_argument("--modelo", choices=list(MODEL_NAMES), default="task",
                    help="classificador usado nas letras (padrão: task)")
args = parser.parse_args()

sys.path.insert(0, str(ROOT_DIR / 'src'))
classifiers = {}  # modelos carregados: nome -> objeto com prever(resultado do MediaPipe)
try:
    from classificadores.snn import ClassificadorSNN, CAMINHO_SNN
    if CAMINHO_SNN.exists():
        classifiers["snn"] = ClassificadorSNN()
except ImportError as e:
    print("Aviso: SNN indisponível (pip install torch snntorch ai-edge-litert):", e)
try:
    from classificadores.classicos import ClassificadorClassico, CAMINHO_RF, CAMINHO_SVM
    for key, path in (("rf", CAMINHO_RF), ("svm", CAMINHO_SVM)):
        if path.exists():
            classifiers[key] = ClassificadorClassico(path)
except ImportError as e:
    print("Aviso: Random Forest e SVM indisponíveis (pip install scikit-learn ai-edge-litert):", e)

available_models = ["task"] + [m for m in MODEL_NAMES if m in classifiers]
if args.modelo not in available_models:
    print(f"Aviso: modelo '{args.modelo}' indisponível. Rode src/treino/treinar_modelos.py "
          "para treinar e salvar os modelos. Usando o MediaPipe .task.")
active_model = args.modelo if args.modelo in available_models else "task"

recognized_text = ""  # texto acumulado com letras
last_seen_token = None  # ultimo token visto
last_legend_update_time = 0.0  # instante do ultimo sinal consolidado
LEGEND_CLEAR_SECONDS = 5  # limpa a legenda apos esse tempo sem novos sinais no modo automatico
LETTER_DELAY_SECONDS = 0.7  # intervalo minimo entre letras
last_letter_commit_time = 0.0  # instante da ultima letra adicionada
HOLD_SECONDS = 1.0  # a letra precisa ficar estável por esse tempo para entrar na legenda
hold_candidate = None  # letra sendo confirmada (barra de progresso)
hold_start = 0.0  # instante em que a letra candidata apareceu
hold_done_letter = None  # última letra confirmada, mostrada em verde por um instante
hold_done_until = 0.0
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hud import HudState, draw_hud, draw_hand, hit, SHORTCUTS_DESKTOP
from corretor import Corretor

# Nomes curtos dos modelos no rodapé do HUD
MODEL_SHORT = {"task": ".task", "snn": "SNN", "rf": "RF", "svm": "SVM"}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
    (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20)  # Mínimo
]
HISTORY_MAX = 50  # frases guardadas no histórico

# Estado do HUD que não depende do modelo
show_history = True
history = []  # [(frase, "HH:MM")], a mais nova primeiro
hud_regions = []  # áreas clicáveis do último quadro desenhado
mouse_events = []  # cliques e rolagens do mouse esperando o próximo quadro
is_fullscreen = False

# Corretor automático: ao terminar uma palavra (espaço), sugere a grafia certa; Tab aceita
corretor = Corretor()
autocorrect_on = True
suggestion = None  # (posição da palavra na frase, palavra escrita, correção) esperando Tab


def check_last_word(text):
    """Procura correção para a palavra que acabou de ser fechada com espaço."""
    word = text.rstrip().split(" ")[-1]
    if not autocorrect_on or not word:
        return None
    fix = corretor.suggest(word)
    if fix is None:
        return None
    return (len(text.rstrip()) - len(word), word, fix)


def suggestion_still_valid(text, sugg):
    return sugg is not None and text[sugg[0]:sugg[0] + len(sugg[1])] == sugg[1]


def on_mouse(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_events.append(("click", x, y, 0))
    elif event == cv2.EVENT_MOUSEWHEEL:
        mouse_events.append(("wheel", x, y, cv2.getMouseWheelDelta(flags)))


def remember_phrase(text):
    """Guarda a frase falada no topo do histórico."""
    history.insert(0, (text, time.strftime("%H:%M")))
    del history[HISTORY_MAX:]


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
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

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
print("  [M]        : Trocar classificador (" + ", ".join(MODEL_NAMES[m] for m in available_models) + ")")
print(f"Classificador ativo: {MODEL_NAMES[active_model]}")
print("  [+] ou [=] : Aumentar limiar de confiança (+5%)")
print("  [-] ou [_] : Diminuir limiar de confiança (-5%)")
print("  [C]        : Limpar legenda acumulada")
print("  [D/Backsp] : Apagar última letra")
print("  [Espaço]   : Adicionar espaço")
print("  [T]        : Mostrar / esconder o histórico de frases")
print("  [F]        : Tela cheia")
print("  [Tab]      : Aceitar a correção sugerida")
print("  [A]        : Ligar / desligar o corretor automático")
print("  Os botões do vídeo também funcionam com o mouse.")
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

def speak_phrase():
    """Fala a frase escrita, guarda no histórico e limpa a legenda (Enter, botão Falar ou modo automático)."""
    global recognized_text, last_seen_token, last_legend_update_time, last_spoken_text
    phrase = recognized_text.strip()
    if not phrase:
        print("Legenda vazia para falar.")
        return
    if USE_TTS:
        speak_async(phrase)
        print(f"Voz disparada: '{phrase}'")
    else:
        print(f"Texto da legenda: '{phrase}' (pyttsx3 não disponível)")
    last_spoken_text = phrase
    remember_phrase(phrase)
    recognized_text = ""
    last_seen_token = None
    last_legend_update_time = 0.0


def do_action(name):
    """Ações dos atalhos de teclado e dos botões do HUD."""
    global show_help, show_history, AUTO_SPEAK, CONFIDENCE_THRESHOLD, active_model, is_fullscreen
    global recognized_text, last_seen_token, last_legend_update_time, hist_scroll
    global autocorrect_on, suggestion
    if name == "help":
        show_help = not show_help
        print(f"Manual de atalhos: {'Aberto' if show_help else 'Fechado'}")
    elif name == "speak":
        speak_phrase()
    elif name == "model":
        if len(available_models) == 1:
            print("Só o MediaPipe .task está disponível: rode src/treino/treinar_modelos.py para treinar os outros.")
        else:
            active_model = available_models[(available_models.index(active_model) + 1) % len(available_models)]
            last_seen_token = None
            print(f"Classificador alternado para: {MODEL_NAMES[active_model]}")
    elif name == "voice":
        AUTO_SPEAK = not AUTO_SPEAK
        mode_str = f"AUTOMÁTICO (fala após {LEGEND_CLEAR_SECONDS}s de inatividade)" if AUTO_SPEAK else "MANUAL (fala apenas ao teclar Enter)"
        print(f"Modo de voz alternado para: {mode_str}")
    elif name in ("thr+", "thr-"):
        step = 0.05 if name == "thr+" else -0.05
        CONFIDENCE_THRESHOLD = min(0.95, max(0.20, round(CONFIDENCE_THRESHOLD + step, 2)))
        print(f"Limiar de confiança ajustado para: {int(CONFIDENCE_THRESHOLD * 100)}%")
    elif name == "clear":
        recognized_text = ""
        last_seen_token = None
        last_legend_update_time = 0.0
        print("Legenda limpa manualmente.")
    elif name == "del":
        if recognized_text:
            recognized_text = recognized_text[:-1]
            last_seen_token = None
            last_legend_update_time = now_seconds
            print(f"Caractere apagado. Legenda atual: '{recognized_text}'")
    elif name == "space":
        if recognized_text and not recognized_text.endswith(" "):
            recognized_text += " "
            last_legend_update_time = now_seconds
            suggestion = check_last_word(recognized_text)
    elif name == "hist":
        show_history = not show_history
        hist_scroll = 0
    elif name == "accept":
        if suggestion_still_valid(recognized_text, suggestion):
            start, word, fix = suggestion
            recognized_text = recognized_text[:start] + fix + recognized_text[start + len(word):]
            print(f"Correção aceita: '{word}' -> '{fix}'")
        suggestion = None
    elif name == "dismiss":
        suggestion = None
    elif name == "autocorrect":
        autocorrect_on = not autocorrect_on
        suggestion = None
        print(f"Corretor automático: {'ligado' if autocorrect_on else 'desligado'}")
    elif name == "full":
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)


ACTIONS = {"help", "speak", "model", "voice", "thr+", "thr-", "clear", "del", "space", "hist", "full",
           "accept", "dismiss", "autocorrect"}
KEY_ACTIONS = {
    13: "speak", 10: "speak",
    ord('m'): "model", ord('M'): "model",
    ord('v'): "voice", ord('V'): "voice",
    ord('+'): "thr+", ord('='): "thr+",
    ord('-'): "thr-", ord('_'): "thr-",
    ord('c'): "clear", ord('C'): "clear",
    8: "del", ord('d'): "del", ord('D'): "del",
    32: "space",
    ord('t'): "hist", ord('T'): "hist",
    ord('f'): "full", ord('F'): "full",
    9: "accept",
    ord('a'): "autocorrect", ord('A'): "autocorrect",
}
hist_scroll = 0
fps, infer_ms, last_frame_time = 0.0, 0.0, 0.0

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

    # Realizar a detecção (o tempo aparece no rodapé do HUD)
    infer_start = time.perf_counter()
    result = recognizer.recognize_for_video(mp_image, timestamp)
    infer_ms = infer_ms * 0.9 + (time.perf_counter() - infer_start) * 1000 * 0.1

    # Redimensionar o frame de exibição para alta definição HD (1280x720)
    # Isso evita serrilhamento de texto e garante linhas suaves de alta resolução
    DISPLAY_W = 1280
    DISPLAY_H = 720
    if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H:
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    h, w = frame.shape[:2]

    # Desenhar o esqueleto da mão sobre o canvas HD
    for hand_landmarks in result.hand_landmarks or []:
        draw_hand(frame, hand_landmarks, HAND_CONNECTIONS)

    # Variáveis de estado do frame
    hold_now = time.monotonic()
    hold_progress = 0.0
    gesture_detected = False
    label = ""
    score = 0.0

    # Resetar token anterior quando a mão não estiver no enquadramento
    if not result.hand_landmarks:
        last_seen_token = None

    # Classificar a letra com o modelo ativo
    top_label, top_score = None, 0.0
    if active_model in classifiers and result.hand_landmarks:
        # SNN, Random Forest ou SVM recebem os 21 landmarks que o MediaPipe acabou de extrair
        top_label, top_score = classifiers[active_model].prever(result)
    elif active_model == "task" and result.gestures:
        top_label = result.gestures[0][0].category_name
        top_score = result.gestures[0][0].score

    # Processar os gestos reconhecidos (pausa a digitação na legenda se o menu de ajuda estiver aberto)
    if top_label is not None and not show_help:
        label = top_label
        score = top_score

        if label and label.lower() != "none":
            gesture_detected = True

            # Atualiza contexto apenas se confiança suficiente e se a letra ficar
            # estável por HOLD_SECONDS (a barra de progresso mostra quanto falta)
            if score >= CONFIDENCE_THRESHOLD:
                token = label.strip().lower()
                if token == last_seen_token:
                    hold_candidate = None  # já entrou: troque de letra ou abaixe a mão para repetir
                else:
                    if token != hold_candidate:
                        hold_candidate, hold_start = token, hold_now
                    hold_progress = min(1.0, (hold_now - hold_start) / HOLD_SECONDS)
                    if hold_progress >= 1.0:
                        recognized_text, last_seen_token, last_letter_commit_time, text_updated = update_recognized_text(
                            label,
                            recognized_text,
                            last_seen_token,
                            now_seconds,
                            last_letter_commit_time,
                        )
                        if text_updated:
                            last_legend_update_time = now_seconds
                            if recognized_text.endswith(" "):  # sinal de espaço: fecha a palavra
                                suggestion = check_last_word(recognized_text)
                        if last_seen_token == token:
                            hold_done_letter, hold_done_until = token, hold_now + 0.4
                            hold_candidate = None
            else:
                hold_candidate = None
        else:
            # Gesto classificado como "None" (posição neutra da mão)
            last_seen_token = None
            hold_candidate = None
    else:
        hold_candidate = None

    # Fala e limpeza automática por inatividade (apenas quando o modo automático estiver ativo)
    if AUTO_SPEAK and recognized_text and last_legend_update_time and (now_seconds - last_legend_update_time) >= LEGEND_CLEAR_SECONDS:
        speak_phrase()

    if not suggestion_still_valid(recognized_text, suggestion):  # palavra apagada ou frase falada
        suggestion = None

    # FPS para o rodapé do HUD
    frame_now = time.monotonic()
    if last_frame_time:
        fps = fps * 0.9 + 0.1 / max(1e-3, frame_now - last_frame_time)
    last_frame_time = frame_now

    # --- HUD: webcam em tela cheia e controles pequenos por cima (igual à versão web) ---
    hud_state = HudState(
        letter=(label.upper() if len(label) == 1 else label) if gesture_detected else None,
        score=score,
        threshold=CONFIDENCE_THRESHOLD,
        holding=hold_candidate is not None and not show_help,
        hold_progress=hold_progress,
        done=hold_done_letter is not None and hold_now < hold_done_until,
        text=recognized_text,
        history=history,
        show_history=show_history,
        hist_scroll=hist_scroll,
        model_name=MODEL_SHORT[active_model],
        auto_speak=AUTO_SPEAK,
        auto_seconds=LEGEND_CLEAR_SECONDS,
        stats=f"{fps:.0f} FPS · {infer_ms:.0f} ms",
        show_help=show_help,
        shortcuts=SHORTCUTS_DESKTOP,
        autocorrect=autocorrect_on,
        suggestion=(suggestion[1], suggestion[2]) if suggestion else None,
    )
    hud_regions = draw_hud(frame, hud_state)
    hist_scroll = hud_state.hist_scroll

    # Exibir o vídeo na janela responsiva
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break  # janela fechada no X

    # Cliques e rolagem do mouse nos botões do HUD
    while mouse_events:
        kind, mx, my, delta = mouse_events.pop(0)
        target = hit(hud_regions, mx, my)
        if kind == "wheel":
            if target is not None and (target == "histcard" or isinstance(target, tuple)):
                hist_scroll = max(0, hist_scroll + (1 if delta < 0 else -1))
        elif show_help:
            if target != "modal":
                do_action("help")  # clique no X ou fora do manual fecha
        elif isinstance(target, tuple) and target[0] == "repeat":
            if target[1] < len(history):
                phrase = history[target[1]][0]
                print(f"Repetindo: '{phrase}'")
                speak_async(phrase)
        elif target in ACTIONS:
            do_action(target)

    if key == ord('q'):
        break
    elif key in (ord('h'), ord('H')):  # Alternar pop-up de ajuda
        do_action("help")
    elif key == 27:  # Tecla ESC fecha a ajuda se estiver aberta, senão ignora a sugestão do corretor
        if show_help:
            do_action("help")
        else:
            do_action("dismiss")
    elif show_help:
        pass  # com o manual aberto, só H e Esc funcionam
    elif key in KEY_ACTIONS:
        do_action(KEY_ACTIONS[key])

cap.release()
cv2.destroyAllWindows()