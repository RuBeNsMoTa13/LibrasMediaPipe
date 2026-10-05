"""
Random Forest e SVM treinados sobre o resumo de 128 números da mão
(src/classificadores/resumo.py), o mesmo que o .task usa.

Os modelos são treinados e salvos por src/evaluation/testar_snn.py e
carregados por src/desktop/detectar_libras.py (opção --modelo rf/svm ou tecla M).
Dependência: scikit-learn (não precisa de torch).
"""
import pickle
from pathlib import Path

import numpy as np

from classificadores.resumo import carregar_rede, resumir_resultado

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
        carregar_rede()  # falha já ao abrir o app se ai-edge-litert faltar
        with open(caminho, "rb") as f:
            dados = pickle.load(f)
        self.modelo = dados["modelo"]
        self.classes = dados["classes"]

    def prever(self, resultado):
        """Recebe o resultado do MediaPipe de um frame e devolve (letra, confiança)."""
        x = resumir_resultado(resultado)
        probabilidades = self.modelo.predict_proba(x)[0]
        indice = int(np.argmax(probabilidades))
        return self.classes[int(self.modelo.classes_[indice])], float(probabilidades[indice])
