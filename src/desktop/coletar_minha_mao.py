"""
Grava os landmarks da SUA mão pela webcam para treinar RF, SVM e SNN com ela.

Como usar:
  python src/desktop/coletar_minha_mao.py
  - Faça o sinal de uma letra e SEGURE a tecla dessa letra (ex.: U).
    Cada frame com a mão detectada vira uma amostra (~100 por letra é bom).
  - Mexa um pouco a mão enquanto grava (aproxime, afaste, incline de leve),
    assim o modelo aprende as variações do seu jeito de fazer a letra.
  - ESC salva e sai.

As amostras são acrescentadas em data/minha_mao/landmarks.npz (rodar de novo
soma às anteriores). Depois rode: python src/evaluation/testar_snn.py
O frame é espelhado como em detectar_libras.py, então os pontos ficam iguais
aos que o app vê ao vivo.
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(ROOT_DIR / "models" / "gesture_recognizer.task")
SAIDA = ROOT_DIR / "data" / "minha_mao" / "landmarks.npz"
LETRAS = set("ABCDEFGILMNOPQRSTUVWY")  # as 21 letras do dataset

options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

X, y = [], []
if SAIDA.exists():
    d = np.load(SAIDA)
    X, y = list(d["X"]), list(d["y"])
contagem = {c: y.count(c) for c in LETRAS}

cap = cv2.VideoCapture(0)
inicio = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = recognizer.recognize_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((time.time() - inicio) * 1000))

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:  # ESC
        break
    letra = chr(tecla).upper() if tecla < 128 else ""

    if res.hand_landmarks:
        h, w = frame.shape[:2]
        for lm in res.hand_landmarks[0]:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        if letra in LETRAS:
            X.append([c for lm in res.hand_landmarks[0] for c in (lm.x, lm.y, lm.z)])
            y.append(letra)
            contagem[letra] += 1
            cv2.putText(frame, f"GRAVANDO {letra}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    feitas = " ".join(f"{c}:{contagem[c]}" for c in sorted(LETRAS) if contagem[c])
    cv2.putText(frame, "Segure a tecla da letra para gravar | ESC salva e sai", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, feitas[:90], (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow("Coletar minha mao", frame)

cap.release()
cv2.destroyAllWindows()
recognizer.close()

if X:
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAIDA, X=np.array(X, dtype=np.float32), y=np.array(y))
    print(f"{len(X)} amostras salvas em {SAIDA}")
    print("Agora rode: python src/evaluation/testar_snn.py")
