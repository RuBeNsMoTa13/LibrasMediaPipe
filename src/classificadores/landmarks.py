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


# Cadeias de pontos de cada dedo, do punho até a ponta (numeração do MediaPipe)
DEDOS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
         (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]


def _angulo(a, b):
    """Ângulo entre os vetores a e b ([N, 3] cada), em radianos dividido por pi (0 a 1)."""
    cos = (a * b).sum(axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-6)
    return np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi


def com_angulos(X_normalizado):
    """Acrescenta 19 ângulos às 63 coordenadas normalizadas: [N, 63] -> [N, 82].

    - 15 dobras: em cada junta de cada dedo, o ângulo entre o osso de antes e o
      de depois (dedo esticado ~0, dobrado perto de 0,5).
    - 4 aberturas: ângulo entre dedos vizinhos, da base à ponta. É o que separa,
      por exemplo, o U (indicador e médio juntos) do V (separados).

    Ângulos não mudam com o tamanho da mão, com a rotação nem com o espelhamento,
    então ajudam o modelo a reconhecer mãos diferentes da do dataset. As
    coordenadas continuam lá porque guardam para onde a mão aponta.
    """
    X = np.asarray(X_normalizado, dtype=np.float32).reshape(-1, 63)
    p = X.reshape(-1, 21, 3)
    angulos = []
    for dedo in DEDOS:
        for i in range(1, 4):
            antes = p[:, dedo[i]] - p[:, dedo[i - 1]]
            depois = p[:, dedo[i + 1]] - p[:, dedo[i]]
            angulos.append(_angulo(antes, depois))
    direcoes = [p[:, dedo[4]] - p[:, dedo[1]] for dedo in DEDOS]
    for d1, d2 in zip(direcoes, direcoes[1:]):
        angulos.append(_angulo(d1, d2))
    return np.concatenate([X, np.stack(angulos, axis=1)], axis=1).astype(np.float32)


def preparar(hand_landmarks, num_features):
    """Landmarks de um frame -> vetor no formato que o modelo salvo espera.

    Modelos treinados antes dos ângulos recebem só as 63 coordenadas.
    """
    x = normalizar(para_array(hand_landmarks))
    return com_angulos(x) if num_features > 63 else x
