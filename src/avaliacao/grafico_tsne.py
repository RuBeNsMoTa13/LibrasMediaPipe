"""
Gera o gráfico t-SNE dos resumos de 128 números das mãos (Figura 6 do TCC).

O t-SNE não é um classificador: ele só "achata" os 128 números de cada mão em
2 números, para dar para desenhar, mantendo perto no gráfico as mãos que eram
parecidas. Cada ponto é uma foto de data/libras (treino e teste) em que o
MediaPipe encontrou a mão, e a cor é a letra verdadeira.

Gráfico salvo em results/figures/:
  - tsne_letras.png   à esquerda todas as letras; à direita só as letras que os
                      modelos mais confundem (M, N, T, F, U, R, V), com as
                      outras em cinza

Uso: python src/avaliacao/grafico_tsne.py
Dependências: pip install scikit-learn matplotlib mediapipe opencv-python ai-edge-litert
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SAIDA = ROOT_DIR / "results" / "figures"
SAIDA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "src"))
from classificadores.resumo import carregar_dados  # noqa: E402

SEED = 42
# Pares e trios de letras com formato parecido
PARECIDAS = ["M", "N", "T", "F", "U", "R", "V"]
# Afasta os rótulos de F e T, que caem quase no mesmo lugar
DESLOCAR = {"F": np.array([-4.0, 0.0]), "T": np.array([4.0, 0.0])}


def main():
    treino, teste = carregar_dados()
    X = np.vstack([treino["resumo"], teste["resumo"]])
    y = np.concatenate([treino["y"], teste["y"]])
    print(f"Calculando o t-SNE de {len(X)} mãos...")

    X = StandardScaler().fit_transform(X)
    pontos = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=SEED).fit_transform(X)

    letras = sorted(set(y))
    # 21 cores sem os dois cinzas do tab20, que ficam para as letras em segundo plano
    cores = [c for i, c in enumerate(plt.cm.tab20.colors) if i not in (14, 15)]
    cores += [plt.cm.Dark2(i) for i in (0, 3, 5)]
    cor_de = {letra: cores[i] for i, letra in enumerate(letras)}

    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(16, 7.5))

    # Esquerda: todas as letras, com o nome no centro de cada nuvem
    for letra in letras:
        m = y == letra
        esq.scatter(pontos[m, 0], pontos[m, 1], s=4, color=cor_de[letra], alpha=0.6)
        cx, cy = np.median(pontos[m], axis=0)
        esq.text(cx, cy, letra, fontsize=12, weight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    esq.set_title("Todas as 21 letras")

    # Direita: só as letras parecidas em cor, o resto em cinza
    outras = ~np.isin(y, PARECIDAS)
    dir_.scatter(pontos[outras, 0], pontos[outras, 1], s=3, color="#d0d0d0", alpha=0.5)
    destaque = plt.cm.Dark2(np.arange(len(PARECIDAS)))
    for i, letra in enumerate(PARECIDAS):
        m = y == letra
        dir_.scatter(pontos[m, 0], pontos[m, 1], s=6, color=destaque[i], alpha=0.8, label=letra)
        cx, cy = np.median(pontos[m], axis=0) + DESLOCAR.get(letra, 0)
        dir_.text(cx, cy, letra, fontsize=13, weight="bold", ha="center", va="center",
                  color=destaque[i],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=destaque[i], alpha=0.9))
    dir_.legend(title="Letras parecidas", markerscale=3, loc="best")
    dir_.set_title("Letras que os modelos mais confundem")

    for ax in (esq, dir_):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Resumos de 128 números das mãos projetados em 2D pelo t-SNE", fontsize=14)
    fig.tight_layout()

    caminho = SAIDA / "tsne_letras.png"
    fig.savefig(caminho, dpi=200)
    print(f"Gráfico salvo em {caminho}")


if __name__ == "__main__":
    main()
