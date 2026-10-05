"""
"Resumo" de 128 números de uma mão, calculado pela mesma rede que o .task usa.

O gesture_recognizer.task não classifica as coordenadas cruas: dentro dele há
uma rede do Google (gesture_embedder.tflite) que transforma os 21 landmarks da
imagem, os 21 landmarks 3D em metros e a mão (direita ou esquerda) num vetor de
128 números, e só esse vetor passa pelo classificador treinado no Colab.

Aqui a mesma rede é lida de dentro do .task e roda sozinha, para que Random
Forest, SVM e SNN classifiquem a partir da mesma base que o .task. Verificado:
passando o resumo pelo classificador que vem no .task, o resultado é idêntico
ao do GestureRecognizer.

Dependência: pip install ai-edge-litert
"""
import io
import os
import zipfile
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELO = ROOT_DIR / "models" / "gesture_recognizer.task"
CACHE = ROOT_DIR / "results" / "tables" / "resumo_libras.npz"
PASTA_TREINO = ROOT_DIR / "data" / "libras" / "train"
PASTA_TESTE = ROOT_DIR / "data" / "libras" / "test"
TAMANHO = 128

_rede = None


def carregar_rede():
    """Abre o .task (um zip com outro zip dentro) e carrega gesture_embedder.tflite."""
    global _rede
    if _rede is None:
        from ai_edge_litert.interpreter import Interpreter

        with zipfile.ZipFile(MODELO) as externo:
            interno = zipfile.ZipFile(io.BytesIO(externo.read("hand_gesture_recognizer.task")))
            conteudo = interno.read("gesture_embedder.tflite")
        _rede = Interpreter(model_content=conteudo)
        _rede.allocate_tensors()
    return _rede


def prob_mao_direita(categoria):
    """Entrada 'handedness' da rede: probabilidade de ser a mão direita."""
    return categoria.score if categoria.category_name == "Right" else 1.0 - categoria.score


def resumir(pontos, pontos_mundo, direita):
    """[N, 21, 3] landmarks da imagem, [N, 21, 3] landmarks 3D e [N] prob. de mão
    direita -> [N, 128]. A própria rede normaliza posição e tamanho da mão."""
    rede = carregar_rede()
    entradas = {d["name"]: d["index"] for d in rede.get_input_details()}
    saida = rede.get_output_details()[0]["index"]
    resumos = np.empty((len(pontos), TAMANHO), dtype=np.float32)
    for i in range(len(pontos)):
        rede.set_tensor(entradas["hand"], np.asarray(pontos[i], np.float32)[None])
        rede.set_tensor(entradas["handedness"], np.array([[direita[i]]], np.float32))
        rede.set_tensor(entradas["world_hand"], np.asarray(pontos_mundo[i], np.float32)[None])
        rede.invoke()
        resumos[i] = rede.get_tensor(saida)[0]
    return resumos


def resumir_resultado(resultado, mao=0):
    """Resumo [1, 128] de uma mão de um GestureRecognizerResult (frame da webcam)."""
    pontos = [[[p.x, p.y, p.z] for p in resultado.hand_landmarks[mao]]]
    mundo = [[[p.x, p.y, p.z] for p in resultado.hand_world_landmarks[mao]]]
    return resumir(pontos, mundo, [prob_mao_direita(resultado.handedness[mao][0])])


def espelhar(pontos, pontos_mundo, direita):
    """A mesma letra feita com a outra mão (ou num frame espelhado)."""
    p = np.array(pontos, np.float32)
    m = np.array(pontos_mundo, np.float32)
    p[:, :, 0] = 1.0 - p[:, :, 0]  # coordenadas da imagem vão de 0 a 1
    m[:, :, 0] *= -1               # coordenadas 3D são centradas na mão
    return p, m, 1.0 - np.asarray(direita, np.float32)


# --- extração das fotos do dataset (com cache) ---
def _extrair_pasta(recognizer, pasta):
    import cv2
    import mediapipe as mp

    dados = {"pontos": [], "mundo": [], "direita": [], "y": [], "pred_task": []}
    total = {}
    for letra in sorted(os.listdir(pasta)):
        if not (pasta / letra).is_dir():
            continue
        arquivos = sorted(os.listdir(pasta / letra))
        print(f" -> Processando letra '{letra}' ({len(arquivos)} imagens)...")
        total[letra.upper()] = 0
        for nome in arquivos:
            img = cv2.imread(str(pasta / letra / nome))
            if img is None:
                continue
            total[letra.upper()] += 1
            res = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.hand_landmarks:
                continue
            dados["pontos"].append([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
            dados["mundo"].append([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
            dados["direita"].append(prob_mao_direita(res.handedness[0][0]))
            dados["y"].append(letra.upper())
            dados["pred_task"].append(res.gestures[0][0].category_name.upper()
                                      if res.gestures else "NENHUM")
    letras = sorted(total)
    dados["letras"], dados["total"] = letras, [total[c] for c in letras]
    return {k: np.array(v) for k, v in dados.items()}


def carregar_dados():
    """Landmarks de treino e teste (do cache ou extraídos com o .task) e seus resumos.

    Devolve dois dicionários (treino, teste) com: pontos, mundo, direita, y,
    pred_task (letra que o .task deu), letras e total (fotos por letra) e resumo.
    """
    if CACHE.exists():
        print(f"Usando landmarks em cache: {CACHE}")
        d = np.load(CACHE)
        treino = {k[7:]: d[k] for k in d.files if k.startswith("treino_")}
        teste = {k[6:]: d[k] for k in d.files if k.startswith("teste_")}
    else:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODELO)),
            running_mode=vision.RunningMode.IMAGE))
        print("\nExtraindo landmarks do TREINO...")
        treino = _extrair_pasta(recognizer, PASTA_TREINO)
        print("\nExtraindo landmarks do TESTE...")
        teste = _extrair_pasta(recognizer, PASTA_TESTE)
        recognizer.close()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{f"treino_{k}": v for k, v in treino.items()},
                            **{f"teste_{k}": v for k, v in teste.items()})
        print(f"Landmarks salvos em {CACHE}")

    for parte in (treino, teste):
        parte["resumo"] = resumir(parte["pontos"], parte["mundo"], parte["direita"])
    return treino, teste
