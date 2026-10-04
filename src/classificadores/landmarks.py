"""Pré-processamento dos 21 landmarks da mão, comum a todos os classificadores."""
import numpy as np


def para_array(hand_landmarks):
    """Converte os 21 landmarks do MediaPipe em [x0, y0, z0, ..., x20, y20, z20]."""
    return [c for lm in hand_landmarks for c in (lm.x, lm.y, lm.z)]


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
