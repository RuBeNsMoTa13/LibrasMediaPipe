"""
Gera a figura do resumo de 128 números da mão (Figura 5 do TCC).

O resumo é a saída da rede gesture_embedder do Google, que vem dentro do
gesture_recognizer.task (ver src/classificadores/resumo.py). Para cada letra
escolhida, o script pega uma foto de teste em que o MediaPipe encontra a mão,
calcula o resumo e desenha:

  (a) a foto da primeira letra com os 21 pontos e os primeiros números do
      resumo, para mostrar que a mão vira uma lista de 128 números;
  (b) gráficos de linha com os 128 números de duas letras sobrepostos.

Cada lista é dividida pelo seu comprimento (norma L2) antes de desenhar, para
comparar só o formato. A semelhança é a similaridade do cosseno entre as duas
listas: 100% = mesmo formato, 0% = sem relação.

Gráfico salvo em results/figures/resumo_128.png

Uso: python src/avaliacao/grafico_resumo.py [LETRA_BASE LETRA_PARECIDA LETRA_DIFERENTE]
     (padrão: M N B)
Dependências: pip install matplotlib mediapipe opencv-python ai-edge-litert
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SAIDA = ROOT_DIR / "results" / "figures"
SAIDA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "src"))
from classificadores.resumo import MODELO, PASTA_TESTE, resumir_resultado  # noqa: E402

# Ligações entre os 21 pontos da mão, para desenhar o esqueleto
LIGACOES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
            (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
            (18, 19), (19, 20), (0, 17)]
CORES = ["#1d4ed8", "#16a34a", "#c2410c"]


def primeira_mao(recognizer, letra):
    """Primeira foto de teste da letra em que o MediaPipe encontra a mão."""
    pasta = PASTA_TESTE / letra
    for nome in sorted(os.listdir(pasta), key=lambda s: int(Path(s).stem)):
        img = cv2.cvtColor(cv2.imread(str(pasta / nome)), cv2.COLOR_BGR2RGB)
        res = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        if res.hand_landmarks:
            pontos = np.array([[p.x, p.y] for p in res.hand_landmarks[0]])
            return img, pontos, resumir_resultado(res)[0]
    raise RuntimeError(f"Nenhuma mão encontrada nas fotos de {letra}")


def desenhar_mao(ax, img, pontos, tamanho=256):
    img = cv2.resize(img, (tamanho, tamanho), interpolation=cv2.INTER_CUBIC)
    ax.imshow(img)
    p = pontos * tamanho
    for a, b in LIGACOES:
        ax.plot(p[[a, b], 0], p[[a, b], 1], color="#22c55e", lw=1.5)
    ax.scatter(p[:, 0], p[:, 1], s=12, c="#ef4444", edgecolors="white", linewidths=0.6, zorder=3)
    ax.axis("off")


def semelhanca(u, v):
    """Similaridade do cosseno entre duas listas de números."""
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def main():
    letras = [l.upper() for l in sys.argv[1:4]] or ["M", "N", "B"]
    recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.IMAGE))
    maos = {l: primeira_mao(recognizer, l) for l in letras}
    recognizer.close()

    base, parecida, diferente = letras
    fig = plt.figure(figsize=(8, 9.2))

    # (a) a mão vira uma lista de 128 números
    img, pontos, resumo = maos[base]
    ax = fig.add_axes([0.03, 0.68, 0.22, 0.22])
    desenhar_mao(ax, img, pontos)
    ax.set_title(f"Foto da letra {base}\ncom os 21 pontos", fontsize=10)
    virgula = lambda v: f"{v:.1f}".replace(".", ",")
    texto = "\n".join([f"número {i + 1}:   {virgula(resumo[i])}" for i in range(4)]
                       + ["      ...", f"número 128: {virgula(resumo[127])}"])
    fig.text(0.33, 0.82, "gesture_embedder\n(rede do Google)  →", fontsize=11, va="center")
    fig.text(0.62, 0.82, "Resumo da mão\n\n" + texto, fontsize=11, va="center", family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#dbeafe", ec="#1d4ed8"))
    fig.text(0.02, 0.975, "a) A rede do Google transforma a mão numa lista de 128 números",
             fontsize=12, weight="bold", va="top")

    # (b) duas letras sobrepostas em gráficos de linha
    fig.text(0.02, 0.655, "b) Comparando as listas: cada linha liga os 128 números de uma mão",
             fontsize=12, weight="bold", va="top")
    unit = {l: maos[l][2] / np.linalg.norm(maos[l][2]) for l in letras}
    lim = max(np.abs(u).max() for u in unit.values()) * 1.08
    x = np.arange(1, 129)
    for r, outra in enumerate([parecida, diferente]):
        y = 0.385 - r * 0.265
        par = [base, outra]
        for k, letra in enumerate(par):
            ax = fig.add_axes([0.02, y + 0.115 - k * 0.115, 0.1, 0.1])
            desenhar_mao(ax, maos[letra][0], maos[letra][1], tamanho=160)
            cor = CORES[letras.index(letra)]
            ax.text(1.08, 0.5, letra, transform=ax.transAxes, fontsize=14, weight="bold",
                    va="center", color=cor)
        g = fig.add_axes([0.2, y, 0.78, 0.22])
        for letra in par:
            g.plot(x, unit[letra], color=CORES[letras.index(letra)], lw=1.6)
        g.axhline(0, color="gray", lw=0.6)
        g.set_xlim(1, 128)
        g.set_ylim(-lim, lim)
        g.set_yticks([])
        g.set_xticks([1, 32, 64, 96, 128])
        g.spines[["top", "right", "left"]].set_visible(False)
        s = semelhanca(maos[base][2], maos[outra][2])
        g.set_title(f"{base} e {outra}: semelhança de {max(s, 0) * 100:.0f}%", fontsize=11,
                    weight="bold", loc="left")
        if r == 0:
            g.set_xticklabels([])
        else:
            g.set_xlabel("posição do número na lista (1 a 128)")
        print(f"Semelhança {base}-{outra}: {s:.3f}")

    caminho = SAIDA / "resumo_128.png"
    fig.savefig(caminho, dpi=250)
    print(f"Gráfico salvo em {caminho}")


if __name__ == "__main__":
    main()
