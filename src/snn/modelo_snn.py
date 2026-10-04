"""
Spiking Neural Network (SNN) que classifica a letra a partir dos 21 landmarks
da mão extraídos pelo MediaPipe.

Usado por:
  - src/evaluation/testar_snn.py: treina, avalia e salva models/snn_libras.pt
  - src/desktop/detectar_libras.py: carrega o modelo salvo e classifica cada
    frame da webcam (opção --modelo snn)

Dependências: pip install torch snntorch
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CAMINHO_SNN = ROOT_DIR / "models" / "snn_libras.pt"

NUM_PASSOS = 25     # quantos instantes de tempo a rede "observa" cada mão
OCULTOS = 128       # neurônios em cada camada escondida
BETA = 0.9          # fator de decaimento da membrana dos neurônios LIF


def normalizar(X):
    """Punho (landmark 0) na origem e escala pelo ponto mais distante dele.

    Recebe um array [N, 63] (x, y, z dos 21 pontos) e devolve o mesmo formato.
    """
    pontos = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3)
    pontos = pontos - pontos[:, :1, :]
    escala = np.linalg.norm(pontos, axis=2).max(axis=1).reshape(-1, 1, 1)
    return (pontos / np.maximum(escala, 1e-6)).reshape(-1, 63).astype(np.float32)


def espelhar(X_normalizado):
    """Inverte o eixo x: a mesma letra feita com a outra mão (ou num frame espelhado)."""
    pontos = X_normalizado.reshape(-1, 21, 3).copy()
    pontos[:, :, 0] *= -1
    return pontos.reshape(-1, 63)


class SNN(nn.Module):
    def __init__(self, entradas=63, ocultos=OCULTOS, saidas=21):
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


def salvar_snn(modelo, classes, caminho=CAMINHO_SNN):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pesos": modelo.state_dict(), "classes": [str(c) for c in classes]}, caminho)


class ClassificadorSNN:
    """Carrega models/snn_libras.pt e classifica uma mão por vez (tempo real)."""

    def __init__(self, caminho=CAMINHO_SNN):
        dados = torch.load(caminho, map_location="cpu")
        self.classes = dados["classes"]
        self.modelo = SNN(saidas=len(self.classes))
        self.modelo.load_state_dict(dados["pesos"])
        self.modelo.eval()

    @torch.no_grad()
    def prever(self, hand_landmarks):
        """Recebe os 21 landmarks do MediaPipe e devolve (letra, confiança).

        A confiança é a fração dos NUM_PASSOS em que o neurônio vencedor disparou
        (a rede foi treinada para disparar em ~80% dos passos na letra certa).
        """
        pontos = [c for lm in hand_landmarks for c in (lm.x, lm.y, lm.z)]
        x = torch.from_numpy(normalizar(pontos))
        contagem = self.modelo(x).sum(0)[0]
        indice = int(contagem.argmax())
        return self.classes[indice], float(contagem[indice]) / NUM_PASSOS
