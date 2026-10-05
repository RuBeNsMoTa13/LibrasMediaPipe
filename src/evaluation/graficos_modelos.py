"""
Gera os gráficos de comparação entre os quatro classificadores do projeto:
MediaPipe (gesture_recognizer.task), Random Forest, SVM e SNN.

Gráficos salvos em results/figures/:
  - distribuicao_amostras.png      imagens por letra no treino e no teste
  - deteccao_maos_por_letra.png    % das fotos de teste em que o MediaPipe achou a mão
  - snn_curva_perda.png            perda da SNN a cada época
  - snn_curva_acuracia.png         acurácia da SNN (treino e teste) a cada época
  - comparacao_modelos.png         acurácia, precisão, recall e F1 dos quatro modelos
  - f1_por_letra.png               F1 de cada letra em cada modelo (mapa de calor)
  - matriz_confusao_rf.png / _svm.png / _snn.png
  - rf_importancia_landmarks.png   quanto cada ponto da mão pesa no Random Forest

Todos os modelos são avaliados nas MESMAS fotos de teste: as de data/libras/test
em que o MediaPipe encontrou uma mão. O treino usa o cache de landmarks criado por
testar_snn.py (results/tables/landmarks_libras.npz) e repete a mesma receita:
features normalizadas, mãos espelhadas, mesmas sementes.

Uso: python src/evaluation/graficos_modelos.py
Dependências: pip install torch snntorch scikit-learn matplotlib seaborn mediapipe opencv-python
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from snntorch import functional as SF
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

warnings.filterwarnings("ignore")

# --- 1. CONFIGURAÇÃO ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PASTA_TREINO = ROOT_DIR / "data" / "libras" / "train"
PASTA_TESTE = ROOT_DIR / "data" / "libras" / "test"
MODELO = ROOT_DIR / "models" / "gesture_recognizer.task"
CACHE = ROOT_DIR / "results" / "tables" / "landmarks_libras.npz"
SAIDA = ROOT_DIR / "results" / "figures"
SAIDA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "src"))
from classificadores.landmarks import normalizar, espelhar, com_angulos  # noqa: E402
from classificadores.snn import SNN  # noqa: E402

EPOCAS = 50
LR = 2e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Uma cor fixa por modelo, igual em todos os gráficos
CORES = {"MediaPipe": "#2a78d6", "Random Forest": "#eb6834",
         "SVM": "#1baf7a", "SNN": "#eda100"}
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRADE, "axes.labelcolor": TINTA_2,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "legend.frameon": False, "font.size": 11,
})

NOMES_LANDMARKS = [
    "Punho", "Polegar 1", "Polegar 2", "Polegar 3", "Polegar ponta",
    "Indicador 1", "Indicador 2", "Indicador 3", "Indicador ponta",
    "Médio 1", "Médio 2", "Médio 3", "Médio ponta",
    "Anelar 1", "Anelar 2", "Anelar 3", "Anelar ponta",
    "Mínimo 1", "Mínimo 2", "Mínimo 3", "Mínimo ponta",
]


def salvar(nome):
    plt.tight_layout()
    caminho = SAIDA / nome
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"  salvo: {caminho.relative_to(ROOT_DIR)}")


# --- 2. TESTE: MEDIAPIPE NAS FOTOS (previsão + landmarks da mesma foto) ---
def avaliar_teste_com_mediapipe():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.IMAGE,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    X, y, pred_mp = [], [], []
    total_por_letra, detectadas_por_letra = {}, {}
    for letra in sorted(os.listdir(PASTA_TESTE)):
        pasta = PASTA_TESTE / letra
        if not pasta.is_dir():
            continue
        letra = letra.upper()
        total_por_letra[letra] = detectadas_por_letra[letra] = 0
        for nome in sorted(os.listdir(pasta)):
            img = cv2.imread(str(pasta / nome))
            if img is None:
                continue
            total_por_letra[letra] += 1
            res = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.hand_landmarks:
                continue
            detectadas_por_letra[letra] += 1
            X.append([c for lm in res.hand_landmarks[0] for c in (lm.x, lm.y, lm.z)])
            y.append(letra)
            pred_mp.append(res.gestures[0][0].category_name.upper() if res.gestures else "NENHUM")
    recognizer.close()
    return (np.array(X, dtype=np.float32), np.array(y), np.array(pred_mp),
            total_por_letra, detectadas_por_letra)


# --- 3. SNN COM HISTÓRICO POR ÉPOCA ---
def treinar_snn_com_historico(X_treino, y_treino, X_teste, y_teste, num_classes):
    modelo = SNN(entradas=X_treino.shape[1], saidas=num_classes)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    perda_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.1)
    Xt, yt = torch.from_numpy(X_treino), torch.from_numpy(y_treino).long()
    Xv, yv = torch.from_numpy(X_teste), torch.from_numpy(y_teste).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    hist = {"perda": [], "acc_treino": [], "acc_teste": []}
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perda_total = 0.0
        for xb, yb in loader:
            perda = perda_fn(modelo(xb), yb)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * len(xb)
        modelo.eval()
        with torch.no_grad():
            acc_t = (modelo(Xt).sum(0).argmax(1) == yt).float().mean().item()
            acc_v = (modelo(Xv).sum(0).argmax(1) == yv).float().mean().item()
        hist["perda"].append(perda_total / len(Xt))
        hist["acc_treino"].append(acc_t)
        hist["acc_teste"].append(acc_v)
        if epoca % 10 == 0 or epoca == 1:
            print(f"  época {epoca:3d} | perda {hist['perda'][-1]:.4f} | "
                  f"acc treino {acc_t:.4f} | acc teste {acc_v:.4f}")
    with torch.no_grad():
        pred = modelo(Xv).sum(0).argmax(1).numpy()
    return pred, hist


def metricas(y_true, y_pred):
    return [accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="weighted", zero_division=0),
            recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f1_score(y_true, y_pred, average="weighted", zero_division=0)]


# --- 4. GRÁFICOS ---
def grafico_distribuicao(classes):
    treino = [len(os.listdir(PASTA_TREINO / c)) for c in classes]
    teste = [len(os.listdir(PASTA_TESTE / c)) for c in classes]
    x = np.arange(len(classes))
    plt.figure(figsize=(11, 4.5))
    plt.bar(x - 0.2, treino, 0.38, color="#2a78d6", label=f"Treino ({sum(treino)})")
    plt.bar(x + 0.2, teste, 0.38, color="#eb6834", label=f"Teste ({sum(teste)})")
    plt.xticks(x, classes)
    plt.grid(axis="x", visible=False)
    plt.ylabel("Imagens")
    plt.ylim(0, max(treino) * 1.2)
    plt.title("Imagens por letra no dataset")
    plt.legend(loc="upper center", ncol=2)
    salvar("distribuicao_amostras.png")


def grafico_deteccao(classes, total, detectadas):
    taxa = [100 * detectadas[c] / total[c] for c in classes]
    plt.figure(figsize=(11, 4.5))
    plt.bar(classes, taxa, 0.6, color="#2a78d6")
    for i, t in enumerate(taxa):
        plt.text(i, t + 1, f"{t:.0f}", ha="center", fontsize=8, color=TINTA_2)
    plt.ylim(0, 110)
    plt.grid(axis="x", visible=False)
    plt.ylabel("Fotos com mão detectada (%)")
    geral = 100 * sum(detectadas.values()) / sum(total.values())
    plt.title(f"Detecção da mão pelo MediaPipe nas fotos de teste (geral: {geral:.1f}%)")
    salvar("deteccao_maos_por_letra.png")


def graficos_snn(hist):
    epocas = range(1, len(hist["perda"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, hist["perda"], color="#eda100", linewidth=2)
    plt.xlabel("Época")
    plt.ylabel("Perda (MSE da contagem de spikes)")
    plt.title("SNN: perda no treinamento")
    salvar("snn_curva_perda.png")

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, hist["acc_treino"], color="#2a78d6", linewidth=2,
             label=f"Treino (final {hist['acc_treino'][-1]:.3f})")
    plt.plot(epocas, hist["acc_teste"], color="#eb6834", linewidth=2,
             label=f"Teste (final {hist['acc_teste'][-1]:.3f})")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.title("SNN: acurácia por época")
    plt.legend(loc="lower right")
    salvar("snn_curva_acuracia.png")


def grafico_comparacao(resultados):
    nomes_metricas = ["Acurácia", "Precisão", "Recall", "F1"]
    modelos = list(resultados)
    x = np.arange(len(nomes_metricas))
    largura = 0.8 / len(modelos)
    plt.figure(figsize=(11, 5.5))
    for i, m in enumerate(modelos):
        pos = x - 0.4 + largura * (i + 0.5)
        plt.bar(pos, resultados[m], largura * 0.92, color=CORES[m], label=m)
        for p, v in zip(pos, resultados[m]):
            plt.text(p, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, color=TINTA_2)
    menor = min(min(v) for v in resultados.values())
    plt.ylim(max(0, menor - 0.1), 1.03)
    plt.xticks(x, nomes_metricas)
    plt.grid(axis="x", visible=False)
    plt.ylabel("Valor (média ponderada pelas letras)")
    plt.title("Comparação dos modelos nas mesmas fotos de teste")
    plt.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    salvar("comparacao_modelos.png")


def grafico_f1_por_letra(y_true, previsoes, classes):
    tabela = np.array([f1_score(y_true, previsoes[m], labels=classes, average=None,
                                zero_division=0) for m in previsoes])
    plt.figure(figsize=(12, 3.8))
    sns.heatmap(tabela, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=classes, yticklabels=list(previsoes),
                linewidths=1, linecolor="white", annot_kws={"fontsize": 8},
                cbar_kws={"label": "F1"})
    plt.grid(False)
    plt.yticks(rotation=0)
    plt.title("F1 por letra em cada modelo")
    salvar("f1_por_letra.png")


def grafico_matriz(y_true, y_pred, classes, nome, arquivo):
    matriz = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.grid(False)
    plt.title(f"Matriz de Confusão - {nome}")
    plt.ylabel("Letra Correta (Realidade)")
    plt.xlabel("Letra Prevista (Modelo)")
    salvar(arquivo)


def grafico_importancia_rf(rf):
    # 63 features = (x, y, z) de 21 pontos; soma as três coordenadas de cada ponto.
    # O punho fica de fora: a normalização o coloca sempre na origem (importância 0).
    imp = rf.feature_importances_[:63].reshape(21, 3).sum(axis=1) * 100
    ordem = [i for i in np.argsort(imp) if i != 0]
    plt.figure(figsize=(8, 7))
    plt.barh([NOMES_LANDMARKS[i] for i in ordem], imp[ordem], 0.65, color="#eb6834")
    for j, i in enumerate(ordem):
        plt.text(imp[i] + 0.1, j, f"{imp[i]:.1f}%", va="center", fontsize=8, color=TINTA_2)
    plt.grid(axis="y", visible=False)
    plt.xlabel("Importância no Random Forest (%)")
    plt.title("Pontos da mão que mais pesam na decisão")
    salvar("rf_importancia_landmarks.png")


# --- 5. EXECUÇÃO ---
if not CACHE.exists():
    sys.exit(f"Cache {CACHE} não encontrado. Rode antes: python src/evaluation/testar_snn.py")
d = np.load(CACHE)
X_treino, y_treino = d["X_treino"], d["y_treino"]
classes = sorted(set(y_treino))
indice = {c: i for i, c in enumerate(classes)}

print("Rodando o MediaPipe nas fotos de teste...")
X_teste, y_teste, pred_mp, total, detectadas = avaliar_teste_com_mediapipe()
print(f"  mãos detectadas: {len(X_teste)} de {sum(total.values())} fotos")

X_treino_n = normalizar(X_treino)
X_teste_n = normalizar(X_teste)
y_treino_i = np.array([indice[c] for c in y_treino])
y_teste_i = np.array([indice[c] for c in y_teste])
X_treino_aug = com_angulos(np.concatenate([X_treino_n, espelhar(X_treino_n)]))
X_teste_n = com_angulos(X_teste_n)
y_treino_aug = np.concatenate([y_treino_i, y_treino_i])

print("Treinando a SNN...")
pred_snn, hist = treinar_snn_com_historico(X_treino_aug, y_treino_aug, X_teste_n,
                                           y_teste_i, len(classes))
print("Treinando Random Forest e SVM...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_treino_aug, y_treino_aug)
svm = SVC(kernel="rbf", random_state=SEED).fit(X_treino_aug, y_treino_aug)

para_letra = np.array(classes)
previsoes = {
    "MediaPipe": pred_mp,
    "Random Forest": para_letra[rf.predict(X_teste_n)],
    "SVM": para_letra[svm.predict(X_teste_n)],
    "SNN": para_letra[pred_snn],
}
resultados = {m: metricas(y_teste, p) for m, p in previsoes.items()}

print("\nModelo          acurácia  precisão  recall    F1")
for m, v in resultados.items():
    print(f"{m:<15} " + "  ".join(f"{x:.3f}   " for x in v))

print("\nGerando gráficos...")
grafico_distribuicao(classes)
grafico_deteccao(classes, total, detectadas)
graficos_snn(hist)
grafico_comparacao(resultados)
grafico_f1_por_letra(y_teste, previsoes, classes)
for nome, arquivo in [("Random Forest", "matriz_confusao_rf.png"),
                      ("SVM", "matriz_confusao_svm.png"),
                      ("SNN", "matriz_confusao_snn.png")]:
    grafico_matriz(y_teste, previsoes[nome], classes, nome, arquivo)
grafico_importancia_rf(rf)
