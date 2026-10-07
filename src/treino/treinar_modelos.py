"""
Treina e testa Random Forest, SVM e uma Spiking Neural Network (SNN) sobre o
mesmo "resumo" de 128 números da mão que o gesture_recognizer.task usa, e
compara os três com o próprio .task nas mesmas fotos de teste.

Fluxo:
  1. Extrai de cada foto de data/libras/train e data/libras/test os landmarks
     da imagem, os landmarks 3D e a mão (direita/esquerda), com o próprio .task,
     e guarda em cache (results/tables/resumo_libras.npz). Fotos em que o
     MediaPipe não encontra a mão ficam de fora.
  2. Passa cada mão pela rede gesture_embedder que vem dentro do .task
     (src/classificadores/resumo.py), que devolve 128 números. Antes, RF,
     SVM e SNN usavam os pontos da mão centralizados no pulso.
  3. Treina a SNN (neurônios LIF, snnTorch), o Random Forest e o SVM nesses
     resumos, com cada mão também espelhada, e imprime as quatro métricas dos
     quatro modelos numa tabela separada por tabulação, pronta para colar no
     Word (Modelo CDI). A linha do .task usa a letra
     que ele mesmo deu a cada foto na extração do passo 1.
  4. Salva os três modelos em models/ (snn_libras.pt, rf_libras.pkl e
     svm_libras.pkl) para o app da webcam, que alterna entre eles com a tecla M.

Uso: python src/treino/treinar_modelos.py
Dependências extras: pip install torch snntorch scikit-learn ai-edge-litert
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from snntorch import functional as SF
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# --- 1. CONFIGURAÇÃO ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(ROOT_DIR / "src"))
from classificadores.resumo import carregar_dados, espelhar, resumir  # noqa: E402
from classificadores.snn import SNN, CAMINHO_SNN, salvar_snn  # noqa: E402
from classificadores.classicos import CAMINHO_RF, CAMINHO_SVM, salvar_classico  # noqa: E402

EPOCAS = 50
LR = 2e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def treinar_snn(X_treino, y_treino, X_teste, num_classes):
    modelo = SNN(entradas=X_treino.shape[1], saidas=num_classes)
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
treino, teste = carregar_dados()
print(f"\nMãos detectadas: treino {len(treino['y'])} | teste {len(teste['y'])}")

classes = sorted(set(treino["y"]))
indice = {c: i for i, c in enumerate(classes)}
y_treino_i = np.array([indice[c] for c in treino["y"]])
y_teste_i = np.array([indice[c] for c in teste["y"]])

# O app da webcam espelha o frame (efeito espelho), e a pessoa pode usar a outra
# mão. Por isso os modelos treinam também com cada mão espelhada.
X_treino_aug = np.concatenate([
    treino["resumo"], resumir(*espelhar(treino["pontos"], treino["mundo"], treino["direita"]))])
y_treino_aug = np.concatenate([y_treino_i, y_treino_i])
X_teste = teste["resumo"]
X_teste_esp = resumir(*espelhar(teste["pontos"], teste["mundo"], teste["direita"]))

# A SNN recebe o resumo padronizado (média 0, desvio 1 em cada número)
media, desvio = X_treino_aug.mean(0), X_treino_aug.std(0) + 1e-6
def padronizar(X):
    return ((X - media) / desvio).astype(np.float32)

print("\nTreinando a SNN (snnTorch, neurônios LIF)...")
inicio = time.time()
modelo_snn, pred_snn, media_spikes = treinar_snn(
    padronizar(X_treino_aug), y_treino_aug, padronizar(X_teste), len(classes))
print(f"Tempo de treino da SNN: {time.time() - inicio:.1f}s | "
      f"spikes de saída por amostra: {media_spikes:.1f}")
m_snn = metricas(y_teste_i, pred_snn)
with torch.no_grad():
    pred_esp = modelo_snn(torch.from_numpy(padronizar(X_teste_esp))).sum(0).argmax(1).numpy()
print(f"Acurácia da SNN no teste espelhado (como na webcam): {accuracy_score(y_teste_i, pred_esp):.3f}")
salvar_snn(modelo_snn, classes, media, desvio)
print(f"Modelo SNN salvo em {CAMINHO_SNN}")

print("\nTreinando Random Forest e SVM nos mesmos resumos...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_treino_aug, y_treino_aug)
m_rf = metricas(y_teste_i, rf.predict(X_teste))
# SVM com os números padronizados; probability=True para o app mostrar a confiança
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED))
svm.fit(X_treino_aug, y_treino_aug)
m_svm = metricas(y_teste_i, svm.predict(X_teste))
for nome, modelo, caminho in [("Random Forest", rf, CAMINHO_RF), ("SVM", svm, CAMINHO_SVM)]:
    acc_esp = accuracy_score(y_teste_i, modelo.predict(X_teste_esp))
    print(f"Acurácia do {nome} no teste espelhado (como na webcam): {acc_esp:.3f}")
    salvar_classico(modelo, classes, caminho)
    print(f"Modelo {nome} salvo em {caminho}")

# O .task já classificou cada foto na extração (pred_task): mesmas fotos de teste
m_task = metricas(teste["y"], teste["pred_task"])

print("\n" + "=" * 65)
print("TABELA PARA O TCC (copie e cole no Word: as colunas são separadas por tabulação)")
print("=" * 65)
print("Modelo\tAcurácia\tPrecisão\tRecall\tF1")
for nome, m in [("MediaPipe (.task)", m_task), ("Random Forest", m_rf),
                ("Support Vector Machine", m_svm), ("Spiking Neural Network", m_snn)]:
    print(nome + "\t" + "\t".join(f"{v:.3f}".replace(".", ",") for v in m))
print("=" * 65)
