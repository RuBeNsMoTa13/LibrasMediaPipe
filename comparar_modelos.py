import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Ignorar avisos do terminal para manter a tela limpa
warnings.filterwarnings("ignore")

# --- 1. CONFIGURAÇÃO ---
# Caminhos para as suas pastas de treino e teste
PASTA_TREINO = os.path.join("libras", "train")
PASTA_TESTE = os.path.join("libras", "test")
MODELO = "gesture_recognizer.task" 

# Inicializar o MediaPipe (usando o seu próprio modelo treinado)
base_options = python.BaseOptions(model_asset_path=MODELO)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE 
)
recognizer = vision.GestureRecognizer.create_from_options(options)

def extrair_landmarks_da_pasta(caminho_base):
    """Função para entrar nas pastas A, B, C... e extrair os pontos tridimensionais das mãos"""
    X = []
    y = []
    
    # Verifica se a pasta existe
    if not os.path.exists(caminho_base):
        print(f"Atenção: A pasta '{caminho_base}' não foi encontrada.")
        return X, y
        
    pastas_letras = os.listdir(caminho_base)
    
    for letra in pastas_letras:
        caminho_pasta_letra = os.path.join(caminho_base, letra)
        if not os.path.isdir(caminho_pasta_letra):
            continue
            
        arquivos = os.listdir(caminho_pasta_letra)
        print(f" -> Processando letra '{letra}' ({len(arquivos)} imagens)...")
        
        for nome_arquivo in arquivos:
            caminho_imagem = os.path.join(caminho_pasta_letra, nome_arquivo)
            imagem_cv2 = cv2.imread(caminho_imagem)
            
            if imagem_cv2 is None:
                continue
                
            imagem_rgb = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
            
            # MediaPipe processa a imagem para achar a mão
            resultado = recognizer.recognize(mp_image)
            
            if resultado.hand_landmarks:
                pontos_da_mao = []
                for landmark in resultado.hand_landmarks[0]:
                    pontos_da_mao.extend([landmark.x, landmark.y, landmark.z])
                
                X.append(pontos_da_mao)
                y.append(letra.upper())
                
    return X, y

# --- 2. EXTRAÇÃO SEPARADA (TREINO E TESTE) ---
print("\n" + "="*50)
print("INICIANDO EXTRAÇÃO DE DADOS: TRAIN (Treinamento)")
print("="*50)
X_treino, y_treino = extrair_landmarks_da_pasta(PASTA_TREINO)
print(f"Mãos detectadas no Treino: {len(X_treino)}")

print("\n" + "="*50)
print("INICIANDO EXTRAÇÃO DE DADOS: TEST (Teste)")
print("="*50)
X_teste, y_teste = extrair_landmarks_da_pasta(PASTA_TESTE)
print(f"Mãos detectadas no Teste: {len(X_teste)}")

if len(X_treino) == 0 or len(X_teste) == 0:
    print("\nERRO: Faltam imagens! Verifique se as pastas libras/train e libras/test existem e têm imagens dentro.")
    exit()

# --- 3. CONVERSÃO PARA NUMPY ARRAY ---
# Isto resolve o aviso do Pylance no VSCode e otimiza o uso no scikit-learn
X_treino = np.array(X_treino)
y_treino = np.array(y_treino)
X_teste = np.array(X_teste)
y_teste = np.array(y_teste)

# --- 4. TREINAMENTO E AVALIAÇÃO: RANDOM FOREST ---
print("\nTreinando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_treino, y_treino)
y_pred_rf = rf.predict(X_teste)

acc_rf = accuracy_score(y_teste, y_pred_rf)
prec_rf = precision_score(y_teste, y_pred_rf, average='weighted')
rec_rf = recall_score(y_teste, y_pred_rf, average='weighted')
f1_rf = f1_score(y_teste, y_pred_rf, average='weighted')

# --- 5. TREINAMENTO E AVALIAÇÃO: SVM ---
print("Treinando Support Vector Machine (SVM)...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_treino, y_treino)
y_pred_svm = svm.predict(X_teste)

acc_svm = accuracy_score(y_teste, y_pred_svm)
prec_svm = precision_score(y_teste, y_pred_svm, average='weighted')
rec_svm = recall_score(y_teste, y_pred_svm, average='weighted')
f1_svm = f1_score(y_teste, y_pred_svm, average='weighted')

# --- 6. IMPRIMINDO OS RESULTADOS PARA O LATEX ---
print("\n" + "="*65)
print("VALORES PARA PREENCHER A SUA TABELA NO LATEX")
print("="*65)
print(f"Random Forest           & {acc_rf:.2f} & {prec_rf:.2f} & {rec_rf:.2f} & {f1_rf:.2f} \\\\")
print(f"Support Vector Machine  & {acc_svm:.2f} & {prec_svm:.2f} & {rec_svm:.2f} & {f1_svm:.2f} \\\\")
print("="*65)