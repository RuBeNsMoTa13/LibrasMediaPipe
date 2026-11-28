"""HUD do app desktop: a webcam ocupa a tela toda e os controles ficam pequenos por cima.

É o mesmo desenho da versão web (web/index.html), feito com OpenCV (formas) e Pillow (textos).
Tudo é desenhado num quadro de 1280x720. draw_hud devolve as áreas clicáveis, para o
detectar_libras.py saber qual botão o mouse apertou.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # sem Pillow, os textos saem com a fonte do OpenCV
    Image = ImageDraw = ImageFont = None

W, H = 1280, 720
MARGIN = 14
PANEL_ALPHA = 0.65      # transparência dos cartões, para a mão aparecer por trás
MAX_HIST_ROWS = 5       # o histórico para de crescer aqui e passa a rolar

# Cores em BGR (o OpenCV usa BGR; o Pillow recebe as mesmas tuplas porque o quadro é BGR)
ACCENT = (248, 189, 56)   # azul claro #38bdf8
GREEN = (153, 211, 52)    # #34d399, letra acima do limiar
AMBER = (36, 191, 251)    # #fbbf24, letra abaixo do limiar
TEXT = (247, 244, 242)    # #f2f4f7
SOFT = (211, 202, 195)    # #c3cad3
MUTED = (174, 163, 154)   # #9aa3ae
PANEL = (18, 14, 12)      # #0c0e12, fundo dos cartões
BUTTON = (50, 45, 41)     # botões dentro dos cartões
TRACK = (70, 64, 60)      # trilho das barras de progresso
LINE = (62, 57, 53)       # divisórias
DARK = (16, 13, 11)       # texto em cima do azul
MODAL = (28, 23, 20)      # #14171c, fundo do manual de atalhos

FONT_FILES = {
    False: ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
_fonts = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = None
        if ImageFont is not None:
            for path in FONT_FILES[bold]:
                if Path(path).exists():
                    _fonts[key] = ImageFont.truetype(path, size)
                    break
    return _fonts[key]


SHORTCUTS_DESKTOP = [
    ("Enter", "Falar a frase e guardar no histórico"),
    ("V", "Voz manual ou automática (fala após 5 s)"),
    ("M", "Trocar o modelo (.task, SNN, RF, SVM)"),
    ("+ / −", "Aumentar ou diminuir o limiar (5%)"),
    ("Espaço", "Inserir espaço entre palavras"),
    ("D / Backspace", "Apagar a última letra"),
    ("C", "Limpar a frase"),
    ("Tab", "Aceitar a correção sugerida"),
    ("A", "Ligar ou desligar o corretor automático"),
    ("T", "Mostrar ou esconder o histórico"),
    ("F", "Tela cheia"),
    ("H / Esc", "Abrir ou fechar este manual"),
    ("Q", "Sair"),
]


@dataclass
class HudState:
    letter: str = None          # letra que o modelo está vendo agora (None = sem gesto)
    score: float = 0.0
    threshold: float = 0.5
    holding: bool = False       # a letra está sendo confirmada (barra enchendo)
    hold_progress: float = 0.0
    done: bool = False          # a letra acabou de entrar (barra verde por um instante)
    text: str = ""              # frase sendo escrita
    history: list = field(default_factory=list)  # [(frase, "HH:MM")], a mais nova primeiro
    show_history: bool = True
    hist_scroll: int = 0
    model_name: str = ".task"
    auto_speak: bool = False
    auto_seconds: int = 5
    stats: str = ""             # "30 FPS · 12 ms"
    autocorrect: bool = True    # corretor automático ligado (tecla A)
    suggestion: tuple = None    # (palavra escrita, correção sugerida) esperando Tab ou o botão Aceitar
    show_help: bool = False
    shortcuts: list = field(default_factory=lambda: list(SHORTCUTS_DESKTOP))


class _Texts:
    """Junta os textos do quadro e desenha todos numa passada só do Pillow (mais rápido)."""

    def __init__(self):
        self.items = []

    @staticmethod
    def width(s, size, bold=False):
        f = _font(size, bold)
        if f is not None:
            return f.getlength(s)
        scale = size / 30
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0][0]

    def add(self, x, y, s, size, color, bold=False, anchor="lm"):
        self.items.append((x, y, s, size, color, bold, anchor))

    def flush(self, frame):
        if not self.items:
            return
        if _font(14) is not None:
            pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil)
            for x, y, s, size, color, bold, anchor in self.items:
                draw.text((x, y), s, font=_font(size, bold), fill=color, anchor=anchor)
            frame[:] = np.asarray(pil)
        else:
            for x, y, s, size, color, bold, anchor in self.items:
                scale = size / 30
                (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
                if anchor[0] == "r":
                    x -= tw
                elif anchor[0] == "m":
                    x -= tw / 2
                cv2.putText(frame, s, (int(x), int(y + th / 2)), cv2.FONT_HERSHEY_DUPLEX, scale,
                            color, 2 if bold else 1, cv2.LINE_AA)
        self.items = []


def rrect(img, x1, y1, x2, y2, r, color):
    """Retângulo de cantos arredondados, preenchido."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = int(min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


# Ícones desenhados numa grade 24x24 (os mesmos traços dos SVG da versão web)
ICONS = {
    "speaker": [("poly", [(11, 5), (6, 9), (3, 9), (3, 15), (6, 15), (11, 19)], True),
                ("arc", (12, 12), 5), ("arc", (12, 12), 9)],
    "space": [("poly", [(4, 10), (4, 14), (20, 14), (20, 10)], False)],
    "back": [("poly", [(21, 5), (9, 5), (3, 12), (9, 19), (21, 19)], True),
             ("poly", [(17, 9), (12, 15)], False), ("poly", [(12, 9), (17, 15)], False)],
    "trash": [("poly", [(4, 7), (20, 7)], False), ("poly", [(9, 7), (9, 4), (15, 4), (15, 7)], False),
              ("poly", [(6, 7), (7, 20), (17, 20), (18, 7)], False)],
    "chat": [("poly", [(3, 21), (3, 5), (3.6, 3.6), (5, 3), (19, 3), (20.4, 3.6), (21, 5), (21, 15),
                       (20.4, 16.4), (19, 17), (7, 17)], True)],
    "full": [("poly", [(4, 9), (4, 4), (9, 4)], False), ("poly", [(20, 9), (20, 4), (15, 4)], False),
             ("poly", [(4, 15), (4, 20), (9, 20)], False), ("poly", [(20, 15), (20, 20), (15, 20)], False)],
    "close": [("poly", [(6, 6), (18, 18)], False), ("poly", [(18, 6), (6, 18)], False)],
}


def draw_icon(img, name, x, y, size, color, thickness=2):
    s = size / 24
    for part in ICONS[name]:
        if part[0] == "poly":
            pts = np.int32([[(x + px * s) * 16, (y + py * s) * 16] for px, py in part[1]])
            cv2.polylines(img, [pts], part[2], color, thickness, cv2.LINE_AA, shift=4)
        else:
            (cx, cy), r = part[1], part[2]
            center = (int((x + cx * s) * 16), int((y + cy * s) * 16))
            radius = int(r * s * 16)
            cv2.ellipse(img, center, (radius, radius), 0, -45, 45, color, thickness, cv2.LINE_AA, shift=4)


def draw_hand(frame, landmarks, connections, accent=ACCENT):
    """Esqueleto da mão: linhas claras e pontos na cor de destaque."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (225, 225, 225), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 5, accent, -1, cv2.LINE_AA)


def _fit_tail(s, size, max_w):
    """Corta o começo da frase quando ela não cabe (mostra sempre o fim, onde se está escrevendo)."""
    if _Texts.width(s, size) <= max_w:
        return s
    while s and _Texts.width("…" + s, size) > max_w:
        s = s[1:]
    return "…" + s


def _fit_head(s, size, max_w):
    if _Texts.width(s, size) <= max_w:
        return s
    while s and _Texts.width(s + "…", size) > max_w:
        s = s[:-1]
    return s + "…"


def draw_hud(frame, st: HudState, accent=ACCENT):
    """Desenha o HUD sobre o quadro (1280x720, BGR) e devolve [(nome, (x1, y1, x2, y2))] clicáveis."""
    regions = []
    texts = _Texts()
    overlay = frame.copy()      # cartões semitransparentes: desenhados aqui e misturados de uma vez
    solids = []                 # formas opacas desenhadas depois da mistura

    # --- Cartão do sinal (canto superior esquerdo) ---
    thr = round(st.threshold * 100)
    if st.letter is None:
        letter, status, status_color = "–", "sem gesto", MUTED
    elif st.score >= st.threshold:
        letter, status, status_color = st.letter, f"{st.score * 100:.0f}% · OK", GREEN
    else:
        letter, status, status_color = st.letter, f"{st.score * 100:.0f}% · abaixo de {thr}%", AMBER
    lw = texts.width(letter, 28, True)
    sw = texts.width(status, 15, True)
    cw = max(180, 12 + lw + 10 + sw + 14)
    cx1, cy1, cx2, cy2 = MARGIN, MARGIN, MARGIN + cw, MARGIN + 62
    rrect(overlay, cx1, cy1, cx2, cy2, 10, PANEL)
    texts.add(cx1 + 12, cy1 + 25, letter, 28, TEXT, True)
    texts.add(cx1 + 12 + lw + 10, cy1 + 26, status, 15, status_color, True)
    bx1, bx2, by = cx1 + 12, cx2 - 12, cy1 + 48
    p = 1.0 if st.done else (st.hold_progress if st.holding else 0.0)
    solids.append(lambda f: rrect(f, bx1, by, bx2, by + 5, 2, TRACK))
    if p > 0:
        fill_color = GREEN if st.done else accent
        solids.append(lambda f: rrect(f, bx1, by, bx1 + max(5, (bx2 - bx1) * p), by + 5, 2, fill_color))

    # --- Botões do canto superior direito: histórico, atalhos, tela cheia ---
    size = 38
    for i, name in enumerate(("full", "help", "hist")):
        x2 = W - MARGIN - i * (size + 6)
        x1 = x2 - size
        rrect(overlay, x1, MARGIN, x2, MARGIN + size, 10, PANEL)
        regions.append((name, (x1, MARGIN, x2, MARGIN + size)))
        if name == "full":
            solids.append(lambda f, x1=x1: draw_icon(f, "full", x1 + 10, MARGIN + 10, 18, TEXT))
        elif name == "help":
            texts.add(x1 + size / 2, MARGIN + size / 2, "?", 18, TEXT, True, "mm")
        else:
            color = accent if st.show_history else TEXT
            solids.append(lambda f, x1=x1, color=color: draw_icon(f, "chat", x1 + 9, MARGIN + 9, 20, color))

    # --- Histórico (logo abaixo do botão que mostra e esconde) ---
    if st.show_history:
        hw = 280
        hx2 = W - MARGIN
        hx1 = hx2 - hw
        hy1 = MARGIN + size + 8
        n = len(st.history)
        scroll = max(0, min(st.hist_scroll, n - MAX_HIST_ROWS)) if n > MAX_HIST_ROWS else 0
        st.hist_scroll = scroll
        visible = st.history[scroll:scroll + MAX_HIST_ROWS]
        row_h = 44
        rows_h = row_h * len(visible) if visible else 40
        footer_h = 26 if n > MAX_HIST_ROWS else 0
        hy2 = hy1 + 32 + rows_h + footer_h + 6
        rrect(overlay, hx1, hy1, hx2, hy2, 14, PANEL)
        regions.append(("histcard", (hx1, hy1, hx2, hy2)))
        texts.add(hx1 + 14, hy1 + 18, "HISTÓRICO", 12, MUTED, True)
        texts.add(hx2 - 14, hy1 + 18, f"{n} frase" + ("" if n == 1 else "s"), 12, MUTED, False, "rm")
        if not visible:
            texts.add(hx1 + 14, hy1 + 32 + 20, "As frases faladas aparecem aqui", 13, MUTED)
        for i, (phrase, hour) in enumerate(visible):
            ry = hy1 + 32 + i * row_h
            solids.append(lambda f, ry=ry: cv2.line(f, (hx1 + 10, ry), (hx2 - 10, ry), LINE, 1, cv2.LINE_AA))
            texts.add(hx1 + 14, ry + 16, _fit_head(phrase, 15, hw - 14 - 50), 15, TEXT)
            texts.add(hx1 + 14, ry + 33, hour, 11, MUTED)
            rx1, ry1 = hx2 - 10 - 32, ry + 6
            solids.append(lambda f, rx1=rx1, ry1=ry1: rrect(f, rx1, ry1, rx1 + 32, ry1 + 32, 8, BUTTON))
            solids.append(lambda f, rx1=rx1, ry1=ry1: draw_icon(f, "speaker", rx1 + 7, ry1 + 7, 18, SOFT))
            regions.append((("repeat", scroll + i), (rx1, ry1, rx1 + 32, ry1 + 32)))
        if footer_h:
            fy = hy1 + 32 + rows_h
            solids.append(lambda f, fy=fy: cv2.line(f, (hx1 + 10, fy), (hx2 - 10, fy), LINE, 1, cv2.LINE_AA))
            shown_to = scroll + len(visible)
            texts.add(hx1 + 14, fy + 14, f"{scroll + 1}–{shown_to} de {n} · role com o mouse", 11, MUTED)

    # --- Frase sendo escrita (barra fina embaixo) ---
    shown = st.text
    cand = st.letter.lower() if (st.holding and st.letter and len(st.letter) == 1) else ""
    if not shown and not cand:
        area_w = 250
    else:
        shown = _fit_tail(shown, 22, 520)
        area_w = max(250, texts.width(shown, 22) + texts.width(cand, 22) + 4)
    speak_label_w = texts.width("Falar", 15, True)
    falar_w = 14 + 18 + 7 + speak_label_w + 14
    total = 18 + area_w + 10 + 3 * 34 + 2 * 6 + 8 + falar_w + 7
    x1 = int((W - total) / 2)
    y1, y2 = H - 40 - 50, H - 40
    ymid = (y1 + y2) / 2
    rrect(overlay, x1, y1, x1 + total, y2, 14, PANEL)
    tx = x1 + 18
    if not shown and not cand:
        texts.add(tx, ymid, "faça um sinal para escrever", 17, MUTED)
    else:
        texts.add(tx, ymid, shown, 22, TEXT)
        if cand:
            ccx = tx + texts.width(shown, 22)
            cwid = max(10, texts.width(cand, 22))
            texts.add(ccx, ymid, cand, 22, accent)
            solids.append(lambda f, ccx=ccx, cwid=cwid: rrect(f, ccx, ymid + 13, ccx + cwid, ymid + 16, 1, TRACK))
            solids.append(lambda f, ccx=ccx, cwid=cwid: rrect(
                f, ccx, ymid + 13, ccx + max(2, cwid * st.hold_progress), ymid + 16, 1, accent))
    bx = x1 + 18 + area_w + 10
    for name, icon in (("space", "space"), ("del", "back"), ("clear", "trash")):
        solids.append(lambda f, bx=bx: rrect(f, bx, y1 + 8, bx + 34, y1 + 42, 8, BUTTON))
        solids.append(lambda f, bx=bx, icon=icon: draw_icon(f, icon, bx + 8, y1 + 16, 18, TEXT))
        regions.append((name, (bx, y1 + 8, bx + 34, y1 + 42)))
        bx += 34 + 6
    fx = bx - 6 + 8
    solids.append(lambda f: rrect(f, fx, y1 + 6, fx + falar_w, y1 + 44, 10, accent))
    solids.append(lambda f: draw_icon(f, "speaker", fx + 14, y1 + 16, 18, DARK))
    texts.add(fx + 14 + 18 + 7, ymid, "Falar", 15, DARK, True)
    regions.append(("speak", (fx, y1 + 6, fx + falar_w, y1 + 44)))

    # --- Sugestão do corretor, logo acima da frase ---
    if st.suggestion:
        typed, fix = st.suggestion
        lab_w = texts.width("corretor", 12, True)
        typed_w = texts.width(typed, 16)
        fix_w = texts.width(fix, 17, True)
        acc_w = texts.width("Aceitar · Tab", 13, True) + 24
        sw_total = 14 + lab_w + 12 + typed_w + 10 + 16 + 10 + fix_w + 14 + acc_w + 6 + 28 + 6
        sx1 = int((W - sw_total) / 2)
        sy1, sy2 = y1 - 8 - 40, y1 - 8
        smid = (sy1 + sy2) / 2
        rrect(overlay, sx1, sy1, sx1 + sw_total, sy2, 12, PANEL)
        x = sx1 + 14
        texts.add(x, smid, "corretor", 12, MUTED, True)
        x += lab_w + 12
        texts.add(x, smid, typed, 16, SOFT)
        solids.append(lambda f, x=x: cv2.line(f, (int(x), int(smid)), (int(x + typed_w), int(smid)), SOFT, 1, cv2.LINE_AA))
        x += typed_w + 10
        solids.append(lambda f, x=x: cv2.arrowedLine(f, (int(x), int(smid)), (int(x + 16), int(smid)), MUTED, 2,
                                                       cv2.LINE_AA, tipLength=0.4))
        x += 16 + 10
        texts.add(x, smid, fix, 17, accent, True)
        x += fix_w + 14
        ax = x
        solids.append(lambda f: rrect(f, ax, sy1 + 6, ax + acc_w, sy2 - 6, 8, accent))
        texts.add(ax + 12, smid, "Aceitar · Tab", 13, DARK, True)
        regions.append(("accept", (ax, sy1 + 6, ax + acc_w, sy2 - 6)))
        dx = ax + acc_w + 6
        solids.append(lambda f: rrect(f, dx, sy1 + 6, dx + 28, sy2 - 6, 8, BUTTON))
        solids.append(lambda f: draw_icon(f, "close", dx + 6, sy1 + 12, 16, TEXT))
        regions.append(("dismiss", (dx, sy1 + 6, dx + 28, sy2 - 6)))

    # --- Rodapé: modelo, limiar, voz e desempenho ---
    fy1, fy2 = H - 10 - 24, H - 10
    fmid = (fy1 + fy2) / 2
    x = MARGIN

    def chip(label, color=SOFT, region=None, bold=False):
        nonlocal x
        cw = texts.width(label, 13, bold) + 20
        rrect(overlay, x, fy1, x + cw, fy2, 7, PANEL)
        texts.add(x + 10, fmid, label, 13, color, bold)
        if region:
            regions.append((region, (x, fy1, x + cw, fy2)))
        x += cw + 5

    chip(f"modelo {st.model_name} (M)", region="model")
    chip("−", TEXT, "thr-", True)
    chip(f"limiar {thr}%")
    chip("+", TEXT, "thr+", True)
    if st.auto_speak:
        chip(f"voz auto {st.auto_seconds} s (V)", accent, "voice")
    else:
        chip("voz manual (V)", SOFT, "voice")
    if st.autocorrect:
        chip("corretor ligado (A)", accent, "autocorrect")
    else:
        chip("corretor desligado (A)", SOFT, "autocorrect")
    if st.stats:
        sw = texts.width(st.stats, 13) + 20
        rrect(overlay, W - MARGIN - sw, fy1, W - MARGIN, fy2, 7, PANEL)
        texts.add(W - MARGIN - 10, fmid, st.stats, 13, SOFT, False, "rm")

    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, dst=frame)
    for draw in solids:
        draw(frame)
    texts.flush(frame)

    if st.show_help:
        return draw_help(frame, st.shortcuts, accent)
    return regions


def draw_help(frame, shortcuts, accent=ACCENT):
    """Manual de atalhos no meio da tela (o reconhecimento fica pausado enquanto está aberto)."""
    texts = _Texts()
    dim = frame.copy()
    cv2.rectangle(dim, (0, 0), (W, H), (13, 10, 8), -1)
    cv2.addWeighted(dim, 0.78, frame, 0.22, 0, dst=frame)
    mw = 600
    mh = 84 + len(shortcuts) * 32 + 44
    x1, y1 = (W - mw) // 2, (H - mh) // 2
    x2, y2 = x1 + mw, y1 + mh
    rrect(frame, x1 - 1, y1 - 1, x2 + 1, y2 + 1, 20, LINE)
    rrect(frame, x1, y1, x2, y2, 20, MODAL)
    texts.add(x1 + 28, y1 + 38, "Atalhos de teclado", 24, TEXT, True)
    close = (x2 - 24 - 38, y1 + 19, x2 - 24, y1 + 57)
    rrect(frame, *close, 10, BUTTON)
    draw_icon(frame, "close", close[0] + 10, close[1] + 10, 18, TEXT)
    for i, (key, desc) in enumerate(shortcuts):
        ry = y1 + 84 + i * 32 + 12
        texts.add(x1 + 28, ry, key, 16, accent, True)
        texts.add(x1 + 28 + 160, ry, desc, 16, TEXT)
    texts.add(x1 + 28, y2 - 26, "O reconhecimento fica pausado enquanto o manual está aberto.", 13, MUTED)
    texts.flush(frame)
    return [("modal", (x1, y1, x2, y2)), ("close_help", close)]


def hit(regions, x, y):
    """Nome da área clicada em (x, y), ou None (as últimas da lista ficam por cima)."""
    for name, (x1, y1, x2, y2) in reversed(regions):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None
