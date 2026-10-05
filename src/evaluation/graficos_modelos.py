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

Todos os modelos são avaliados nas MESMAS fotos de teste: as de data/libras/test
em que o MediaPipe encontrou uma mão. RF, SVM e SNN usam o mesmo resumo de 128
números da mão que o .task (src/classificadores/resumo.py) e repetem a receita de
testar_snn.py: mãos espelhadas, mesma padronização, mesmas sementes.

Uso: python src/evaluation/graficos_modelos.py
Dependências: pip install torch snntorch scikit-learn matplotlib seaborn mediapipe opencv-python ai-edge-litert
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

warnings.filterwarnings("ignore")

# --- 1. CONFIGURAÇÃO ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PASTA_TREINO = ROOT_DIR / "data" / "libras" / "train"
PASTA_TESTE = ROOT_DIR / "data" / "libras" / "test"
SAIDA = ROOT_DIR / "results" / "figures"
SAIDA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "src"))
from classificadores.resumo import carregar_dados, espelhar, resumir  # noqa: E402
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



def salvar(nome):
    plt.tight_layout()
    caminho = SAIDA / nome
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"  salvo: {caminho.relative_to(ROOT_DIR)}")


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


# --- 5. EXECUÇÃO ---
treino, teste = carregar_dados()
classes = sorted(set(treino["y"]))
indice = {c: i for i, c in enumerate(classes)}
y_teste = teste["y"]
pred_mp = teste["pred_task"]
total = dict(zip(teste["letras"], teste["total"]))
detectadas = {c: int((y_teste == c).sum()) for c in classes}
print(f"  mãos detectadas no teste: {len(y_teste)} de {sum(total.values())} fotos")

y_treino_i = np.array([indice[c] for c in treino["y"]])
y_teste_i = np.array([indice[c] for c in y_teste])
X_treino_aug = np.concatenate([
    treino["resumo"], resumir(*espelhar(treino["pontos"], treino["mundo"], treino["direita"]))])
y_treino_aug = np.concatenate([y_treino_i, y_treino_i])
X_teste = teste["resumo"]
media, desvio = X_treino_aug.mean(0), X_treino_aug.std(0) + 1e-6
def padronizar(X):
    return ((X - media) / desvio).astype(np.float32)

print("Treinando a SNN...")
pred_snn, hist = treinar_snn_com_historico(padronizar(X_treino_aug), y_treino_aug,
                                           padronizar(X_teste), y_teste_i, len(classes))
print("Treinando Random Forest e SVM...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_treino_aug, y_treino_aug)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED))
svm.fit(X_treino_aug, y_treino_aug)

para_letra = np.array(classes)
previsoes = {
    "MediaPipe": pred_mp,
    "Random Forest": para_letra[rf.predict(X_teste)],
    "SVM": para_letra[svm.predict(X_teste)],
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
