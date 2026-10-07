"""
Treina e testa Random Forest, SVM e uma Spiking Neural Network (SNN) sobre o
mesmo "resumo" de 128 números da mão que o gesture_recognizer.task usa, e
compara os três com o próprio .task nas mesmas fotos de teste.

Fluxo:
  1. Extrai de cada foto de train/ e test/ do dataset do Kaggle (baixado na
     primeira vez pelo kagglehub, ou a pasta da variável LIBRAS_DADOS) os
     landmarks da imagem, os landmarks 3D e a mão (direita/esquerda), com o
     próprio .task, e guarda em cache (results/tables/resumo_libras_kaggle.npz).
     Fotos em que o MediaPipe não encontra a mão ficam de fora.
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
  5. Anota o treino em results/treinos/classificadores.json: data, quantidade de
     fotos, configurações de cada modelo, perda da SNN a cada época, métricas e
     versões das bibliotecas, para o TCC não depender de números copiados à mão.

Uso: python src/treino/treinar_modelos.py
Dependências extras: pip install torch snntorch scikit-learn ai-edge-litert kagglehub
"""
import json
import platform
import sys
import time
import warnings
from datetime import datetime
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
from classificadores.resumo import carregar_dados, espelhar, pasta_dados, resumir  # noqa: E402
from classificadores.snn import SNN, CAMINHO_SNN, salvar_snn  # noqa: E402
from classificadores.classicos import CAMINHO_RF, CAMINHO_SVM, salvar_classico  # noqa: E402

EPOCAS = 50
LR = 2e-3
LOTE = 64
SEED = 42
REGISTRO = ROOT_DIR / "results" / "treinos" / "classificadores.json"

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
        torch.utils.data.TensorDataset(Xt, yt), batch_size=LOTE, shuffle=True)

    historico = []
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
        historico.append({"epoca": epoca, "perda": round(perda_total / len(Xt), 6)})
        if epoca % 10 == 0 or epoca == 1:
            modelo.eval()
            with torch.no_grad():
                acc = (modelo(Xt).sum(0).argmax(1) == yt).float().mean().item()
            historico[-1]["acuracia_treino"] = round(acc, 6)
            print(f"Época {epoca:3d} | perda {perda_total / len(Xt):.4f} | acurácia treino {acc:.4f}")

    modelo.eval()
    with torch.no_grad():
        spikes = modelo(torch.from_numpy(X_teste))
        pred = spikes.sum(0).argmax(1).numpy()
        media_spikes = spikes.sum().item() / len(X_teste)
    return modelo, pred, media_spikes, historico


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
modelo_snn, pred_snn, media_spikes, historico_snn = treinar_snn(
    padronizar(X_treino_aug), y_treino_aug, padronizar(X_teste), len(classes))
print(f"Tempo de treino da SNN: {time.time() - inicio:.1f}s | "
      f"spikes de saída por amostra: {media_spikes:.1f}")
m_snn = metricas(y_teste_i, pred_snn)
with torch.no_grad():
    pred_esp = modelo_snn(torch.from_numpy(padronizar(X_teste_esp))).sum(0).argmax(1).numpy()
acc_esp_snn = accuracy_score(y_teste_i, pred_esp)
print(f"Acurácia da SNN no teste espelhado (como na webcam): {acc_esp_snn:.3f}")
salvar_snn(modelo_snn, classes, media, desvio)
print(f"Modelo SNN salvo em {CAMINHO_SNN}")

print("\nTreinando Random Forest e SVM nos mesmos resumos...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_treino_aug, y_treino_aug)
m_rf = metricas(y_teste_i, rf.predict(X_teste))
# SVM com os números padronizados; probability=True para o app mostrar a confiança
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED))
svm.fit(X_treino_aug, y_treino_aug)
m_svm = metricas(y_teste_i, svm.predict(X_teste))
acc_esp = {"Spiking Neural Network": acc_esp_snn}
for nome, modelo, caminho in [("Random Forest", rf, CAMINHO_RF), ("SVM", svm, CAMINHO_SVM)]:
    acc_esp[nome] = accuracy_score(y_teste_i, modelo.predict(X_teste_esp))
    print(f"Acurácia do {nome} no teste espelhado (como na webcam): {acc_esp[nome]:.3f}")
    salvar_classico(modelo, classes, caminho)
    print(f"Modelo {nome} salvo em {caminho}")

# O .task já classificou cada foto na extração (pred_task): mesmas fotos de teste
m_task = metricas(teste["y"], teste["pred_task"])

print("\n" + "=" * 65)
print("TABELA PARA O TCC (copie e cole no Word: as colunas são separadas por tabulação)")
print("=" * 65)
print("Modelo\tAcurácia\tPrecisão\tRecall\tF1")
tabela = [("MediaPipe (.task)", m_task), ("Random Forest", m_rf),
          ("Support Vector Machine", m_svm), ("Spiking Neural Network", m_snn)]
for nome, m in tabela:
    print(nome + "\t" + "\t".join(f"{v:.3f}".replace(".", ",") for v in m))
print("=" * 65)

# --- 6. REGISTRO DO TREINO ---
def versao(modulo):
    try:
        return __import__(modulo).__version__
    except Exception:
        return None

registro = {
    "data": datetime.now().astimezone().isoformat(timespec="seconds"),
    "dataset": "Kaggle williansoliveira/libras, divisão train/test do próprio dataset",
    "pasta_dados": str(pasta_dados()),
    "semente": SEED,
    "fotos": {"treino": int(sum(treino["total"])), "teste": int(sum(teste["total"])),
              "treino_com_mao": len(treino["y"]), "teste_com_mao": len(teste["y"]),
              "treino_com_espelhadas": len(y_treino_aug)},
    "entrada": "resumo de 128 números do gesture_embedder (dentro do .task)",
    "modelos": {
        "Random Forest": {"n_estimators": 100, "random_state": SEED},
        "Support Vector Machine": {"padronizacao": "StandardScaler", "kernel": "rbf",
                                   "C": 1.0, "gamma": "scale", "probability": True},
        "Spiking Neural Network": {"epocas": EPOCAS, "taxa_aprendizado": LR, "lote": LOTE,
                                   "otimizador": "Adam",
                                   "perda": "mse_count_loss(correct_rate=0.8, incorrect_rate=0.1)",
                                   "spikes_saida_por_amostra": round(media_spikes, 2),
                                   "historico": historico_snn},
    },
    "metricas_teste": {nome: dict(zip(["acuracia", "precisao", "recall", "f1"],
                                      [round(float(v), 4) for v in m])) for nome, m in tabela},
    "acuracia_teste_espelhado": {k: round(float(v), 4) for k, v in acc_esp.items()},
    "versoes": {"python": platform.python_version(),
                **{m: versao(m) for m in ["numpy", "sklearn", "torch", "snntorch", "mediapipe"]}},
}
REGISTRO.parent.mkdir(parents=True, exist_ok=True)
REGISTRO.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Registro do treino salvo em {REGISTRO}")
