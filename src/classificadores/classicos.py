"""
Random Forest e SVM treinados sobre os landmarks normalizados.

Os modelos são treinados e salvos por src/evaluation/testar_snn.py e
carregados por src/desktop/detectar_libras.py (opção --modelo rf/svm ou tecla M).
Dependência: scikit-learn (não precisa de torch).
"""
import pickle
from pathlib import Path

import numpy as np

from classificadores.landmarks import normalizar, para_array

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CAMINHO_RF = ROOT_DIR / "models" / "rf_libras.pkl"
CAMINHO_SVM = ROOT_DIR / "models" / "svm_libras.pkl"


def salvar_classico(modelo, classes, caminho):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump({"modelo": modelo, "classes": [str(c) for c in classes]}, f)


class ClassificadorClassico:
    """Classifica uma mão por vez com um modelo do scikit-learn salvo em models/."""

    def __init__(self, caminho):
        with open(caminho, "rb") as f:
            dados = pickle.load(f)
        self.modelo = dados["modelo"]
        self.classes = dados["classes"]

    def prever(self, hand_landmarks):
        """Recebe os 21 landmarks do MediaPipe e devolve (letra, confiança)."""
        x = normalizar(para_array(hand_landmarks))
        probabilidades = self.modelo.predict_proba(x)[0]
        indice = int(np.argmax(probabilidades))
        return self.classes[int(self.modelo.classes_[indice])], float(probabilidades[indice])
