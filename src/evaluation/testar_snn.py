"""
Teste de uma Spiking Neural Network (SNN) sobre os landmarks do MediaPipe.

Fluxo:
  1. Extrai os 21 landmarks (63 coordenadas) das imagens de data/libras/train e
     data/libras/test, do mesmo jeito que comparar_modelos.py, e guarda tudo em
     cache (results/tables/landmarks_libras.npz) para não reprocessar as fotos.
  2. Normaliza cada mão (punho na origem, escala pelo maior ponto) para a rede
     não depender da posição ou do tamanho da mão na imagem.
  3. Treina uma SNN com neurônios LIF (snnTorch, definida em
     src/classificadores/snn.py), com cada mão também espelhada. As 63
     coordenadas entram como corrente constante durante NUM_PASSOS instantes de
     tempo; a letra prevista é o neurônio de saída que mais disparou.
  4. Treina Random Forest e SVM nas MESMAS features e imprime as quatro
     métricas de todos no formato da tabela LaTeX.
  5. Salva os três modelos em models/ (snn_libras.pt, rf_libras.pkl e
     svm_libras.pkl) para o app da webcam, que alterna entre eles com a tecla M.

Dependências extras: pip install torch snntorch scikit-learn
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
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

sys.path.insert(0, str(ROOT_DIR / "src"))
from classificadores.landmarks import normalizar, espelhar  # noqa: E402
from classificadores.snn import SNN, CAMINHO_SNN, salvar_snn  # noqa: E402
from classificadores.classicos import CAMINHO_RF, CAMINHO_SVM, salvar_classico  # noqa: E402

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


def treinar_snn(X_treino, y_treino, X_teste, num_classes):
    modelo = SNN(saidas=num_classes)
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
    return modelo, pred, media_spikes


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
# O app da webcam espelha o frame (efeito espelho), e a pessoa pode usar a outra
# mão. Por isso os modelos treinam também com cada mão espelhada no eixo x.
X_teste_esp = espelhar(X_teste_n)
X_treino_aug = np.concatenate([X_treino_n, espelhar(X_treino_n)])
y_treino_aug = np.concatenate([y_treino_i, y_treino_i])
modelo_snn, pred_snn, media_spikes = treinar_snn(X_treino_aug, y_treino_aug, X_teste_n, len(classes))
print(f"Tempo de treino da SNN: {time.time() - inicio:.1f}s | "
      f"spikes de saída por amostra: {media_spikes:.1f}")
m_snn = metricas(y_teste_i, pred_snn)
with torch.no_grad():
    pred_esp = modelo_snn(torch.from_numpy(X_teste_esp)).sum(0).argmax(1).numpy()
print(f"Acurácia da SNN no teste espelhado (como na webcam): {accuracy_score(y_teste_i, pred_esp):.3f}")
salvar_snn(modelo_snn, classes)
print(f"Modelo SNN salvo em {CAMINHO_SNN}")

print("\nTreinando Random Forest e SVM nas mesmas features normalizadas...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_treino_aug, y_treino_aug)
m_rf = metricas(y_teste_i, rf.predict(X_teste_n))
# probability=True para o app mostrar a confiança de cada letra
svm = SVC(kernel="rbf", probability=True, random_state=SEED).fit(X_treino_aug, y_treino_aug)
m_svm = metricas(y_teste_i, svm.predict(X_teste_n))
for nome, modelo, caminho in [("Random Forest", rf, CAMINHO_RF), ("SVM", svm, CAMINHO_SVM)]:
    acc_esp = accuracy_score(y_teste_i, modelo.predict(X_teste_esp))
    print(f"Acurácia do {nome} no teste espelhado (como na webcam): {acc_esp:.3f}")
    salvar_classico(modelo, classes, caminho)
    print(f"Modelo {nome} salvo em {caminho}")

print("\n" + "=" * 65)
print("VALORES PARA A TABELA NO LATEX (acurácia, precisão, recall, F1)")
print("=" * 65)
for nome, m in [("Random Forest", m_rf), ("Support Vector Machine", m_svm),
                ("Spiking Neural Network", m_snn)]:
    print(f"{nome:<24}& {m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f} & {m[3]:.3f} \\\\")
print("=" * 65)
