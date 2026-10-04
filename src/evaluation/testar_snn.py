"""
Teste de uma Spiking Neural Network (SNN) sobre os landmarks do MediaPipe.

Fluxo:
  1. Extrai os 21 landmarks (63 coordenadas) das imagens de data/libras/train e
     data/libras/test, do mesmo jeito que comparar_modelos.py, e guarda tudo em
     cache (results/tables/landmarks_libras.npz) para não reprocessar as fotos.
  2. Normaliza cada mão (punho na origem, escala pelo maior ponto) para a rede
     não depender da posição ou do tamanho da mão na imagem.
  3. Treina uma SNN com neurônios LIF (snnTorch). As 63 coordenadas entram como
     corrente constante durante NUM_PASSOS instantes de tempo; a letra prevista
     é o neurônio de saída que mais disparou.
  4. Treina Random Forest e SVM nas MESMAS features e imprime as quatro
     métricas de todos no formato da tabela LaTeX.

Dependências extras: pip install torch snntorch scikit-learn
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# --- 1. CONFIGURAÇÃO ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PASTA_TREINO = ROOT_DIR / "data" / "libras" / "train"
PASTA_TESTE = ROOT_DIR / "data" / "libras" / "test"
MODELO = ROOT_DIR / "models" / "gesture_recognizer.task"
CACHE = ROOT_DIR / "results" / "tables" / "landmarks_libras.npz"

NUM_PASSOS = 25     # quantos instantes de tempo a rede "observa" cada mão
OCULTOS = 128       # neurônios em cada camada escondida
BETA = 0.9          # fator de decaimento da membrana dos neurônios LIF
EPOCAS = 50
LR = 2e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# --- 2. EXTRAÇÃO DOS LANDMARKS (com cache) ---
def extrair_landmarks_da_pasta(recognizer, caminho_base):
    import cv2
    import mediapipe as mp

    X, y = [], []
    for letra in sorted(os.listdir(caminho_base)):
        pasta_letra = os.path.join(caminho_base, letra)
        if not os.path.isdir(pasta_letra):
            continue
        arquivos = os.listdir(pasta_letra)
        print(f" -> Processando letra '{letra}' ({len(arquivos)} imagens)...")
        for nome_arquivo in arquivos:
            imagem_cv2 = cv2.imread(os.path.join(pasta_letra, nome_arquivo))
            if imagem_cv2 is None:
                continue
            imagem_rgb = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
            resultado = recognizer.recognize(mp_image)
            if resultado.hand_landmarks:
                pontos = []
                for landmark in resultado.hand_landmarks[0]:
                    pontos.extend([landmark.x, landmark.y, landmark.z])
                X.append(pontos)
                y.append(letra.upper())
    return np.array(X, dtype=np.float32), np.array(y)


def carregar_dados():
    if CACHE.exists():
        print(f"Usando landmarks em cache: {CACHE}")
        d = np.load(CACHE)
        return d["X_treino"], d["y_treino"], d["X_teste"], d["y_teste"]

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.IMAGE,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    print("\nExtraindo landmarks do TREINO...")
    X_treino, y_treino = extrair_landmarks_da_pasta(recognizer, PASTA_TREINO)
    print("\nExtraindo landmarks do TESTE...")
    X_teste, y_teste = extrair_landmarks_da_pasta(recognizer, PASTA_TESTE)
    recognizer.close()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X_treino=X_treino, y_treino=y_treino,
                        X_teste=X_teste, y_teste=y_teste)
    print(f"Landmarks salvos em {CACHE}")
    return X_treino, y_treino, X_teste, y_teste


# --- 3. NORMALIZAÇÃO DA MÃO ---
def normalizar(X):
    """Punho (landmark 0) na origem e escala pelo ponto mais distante dele."""
    pontos = X.reshape(-1, 21, 3)
    pontos = pontos - pontos[:, :1, :]
    escala = np.linalg.norm(pontos, axis=2).max(axis=1).reshape(-1, 1, 1)
    return (pontos / np.maximum(escala, 1e-6)).reshape(-1, 63).astype(np.float32)


# --- 4. A REDE SPIKING ---
class SNN(nn.Module):
    def __init__(self, entradas, ocultos, saidas):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(entradas, ocultos)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=grad)
        self.fc2 = nn.Linear(ocultos, ocultos)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=grad)
        self.fc3 = nn.Linear(ocultos, saidas)
        self.lif3 = snn.Leaky(beta=BETA, spike_grad=grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spikes_saida = []
        # Codificação direta: a mesma entrada é injetada em todos os passos de tempo
        for _ in range(NUM_PASSOS):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            spikes_saida.append(spk3)
        return torch.stack(spikes_saida)  # [passos, batch, classes]


def treinar_snn(X_treino, y_treino, X_teste, num_classes):
    modelo = SNN(X_treino.shape[1], OCULTOS, num_classes)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    # Pede ~80% de disparos no neurônio da letra certa e ~10% nos demais.
    # Com ce_rate_loss a mesma rede travava em ~87% de acurácia.
    perda_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.1)

    Xt = torch.from_numpy(X_treino)
    yt = torch.from_numpy(y_treino).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perda_total = 0.0
        for xb, yb in loader:
            spikes = modelo(xb)
            perda = perda_fn(spikes, yb)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * len(xb)
        if epoca % 10 == 0 or epoca == 1:
            modelo.eval()
            with torch.no_grad():
                acc = (modelo(Xt).sum(0).argmax(1) == yt).float().mean().item()
            print(f"Época {epoca:3d} | perda {perda_total / len(Xt):.4f} | acurácia treino {acc:.4f}")

    modelo.eval()
    with torch.no_grad():
        spikes = modelo(torch.from_numpy(X_teste))
        pred = spikes.sum(0).argmax(1).numpy()
        media_spikes = spikes.sum().item() / len(X_teste)
    return pred, media_spikes


def metricas(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="weighted"),
            recall_score(y_true, y_pred, average="weighted"),
            f1_score(y_true, y_pred, average="weighted"))


# --- 5. EXECUÇÃO ---
X_treino, y_treino, X_teste, y_teste = carregar_dados()
print(f"\nMãos detectadas: treino {len(X_treino)} | teste {len(X_teste)}")

X_treino_n = normalizar(X_treino)
X_teste_n = normalizar(X_teste)

classes = sorted(set(y_treino))
indice = {c: i for i, c in enumerate(classes)}
y_treino_i = np.array([indice[c] for c in y_treino])
y_teste_i = np.array([indice[c] for c in y_teste])

print("\nTreinando a SNN (snnTorch, neurônios LIF)...")
inicio = time.time()
pred_snn, media_spikes = treinar_snn(X_treino_n, y_treino_i, X_teste_n, len(classes))
print(f"Tempo de treino da SNN: {time.time() - inicio:.1f}s | "
      f"spikes de saída por amostra: {media_spikes:.1f}")
m_snn = metricas(y_teste_i, pred_snn)

print("\nTreinando Random Forest e SVM nas mesmas features normalizadas...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_treino_n, y_treino_i)
m_rf = metricas(y_teste_i, rf.predict(X_teste_n))
svm = SVC(kernel="rbf", random_state=SEED).fit(X_treino_n, y_treino_i)
m_svm = metricas(y_teste_i, svm.predict(X_teste_n))

print("\n" + "=" * 65)
print("VALORES PARA A TABELA NO LATEX (acurácia, precisão, recall, F1)")
print("=" * 65)
for nome, m in [("Random Forest", m_rf), ("Support Vector Machine", m_svm),
                ("Spiking Neural Network", m_snn)]:
    print(f"{nome:<24}& {m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f} & {m[3]:.3f} \\\\")
print("=" * 65)
