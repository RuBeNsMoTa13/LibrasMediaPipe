import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURAÇÕES ---
# Coloque aqui o caminho da pasta onde estão suas fotos de teste separadas por pastas
PASTA_TESTE = "teste_libras" 
MODELO = "gesture_recognizer.task"

# --- 2. PREPARAR O MEDIAPIPE ---
# Diferente do seu app.py, aqui usamos RunningMode.IMAGE porque são fotos paradas, não webcam
base_options = python.BaseOptions(model_asset_path=MODELO)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE 
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Listas para guardar as respostas
y_verdadeiro = [] # O que a foto realmente é (nome da pasta)
y_previsto = []   # O que o MediaPipe achou que era

print("Iniciando a avaliação das imagens...")

# --- 3. LER AS IMAGENS E FAZER AS PREVISÕES ---
# Vai entrar em cada pasta (A, B, C...) dentro da pasta de teste
for letra_real in os.listdir(PASTA_TESTE):
    caminho_pasta_letra = os.path.join(PASTA_TESTE, letra_real)
    
    if not os.path.isdir(caminho_pasta_letra):
        continue
        
    for nome_arquivo in os.listdir(caminho_pasta_letra):
        caminho_imagem = os.path.join(caminho_pasta_letra, nome_arquivo)
        
        # Carrega a imagem com OpenCV e converte para o formato do MediaPipe
        imagem_cv2 = cv2.imread(caminho_imagem)
        if imagem_cv2 is None:
            continue
            
        imagem_rgb = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        
        # Faz a previsão
        resultado = recognizer.recognize(mp_image)
        
        # Registra a letra real (nome da pasta)
        y_verdadeiro.append(letra_real.upper())
        
        # Registra o que o modelo previu
        if resultado.gestures:
            # Pega o gesto com maior pontuação
            letra_prevista = resultado.gestures[0][0].category_name.upper()
            y_previsto.append(letra_prevista)
        else:
            # Se ele não reconheceu nenhuma mão na foto
            y_previsto.append("Nenhum")

print("\nAvaliação concluída! Gerando métricas...\n")

# --- 4. GERAR O RELATÓRIO (PRECISÃO, RECALL, F1-SCORE) ---
print("--- RELATÓRIO DE DESEMPENHO (9.2) ---")
print(classification_report(y_verdadeiro, y_previsto))

# --- 5. GERAR E SALVAR A MATRIZ DE CONFUSÃO (9.3) ---
classes = sorted(list(set(y_verdadeiro + y_previsto)))
matriz = confusion_matrix(y_verdadeiro, y_previsto, labels=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Matriz de Confusão - Reconhecimento de LIBRAS', fontsize=14, fontweight='bold')
plt.ylabel('Letra Correta (Realidade)', fontsize=12)
plt.xlabel('Letra Prevista (Modelo)', fontsize=12)

plt.tight_layout()
plt.savefig('matriz_de_confusao.png', dpi=300)
print("\nImagem 'matriz_de_confusao.png' salva com sucesso na sua pasta!")
plt.show()