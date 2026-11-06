"""
Curvas de acurácia e perda do treino do gesture_recognizer.task (Figuras 1 e 2 do TCC).

O .task é treinado no Google Colab com o MediaPipe Model Maker
(notebooks/treinar_no_colab.ipynb). O notebook anota o treino em
results/treinos/task.json, com a perda e a acurácia de cada época; este script
só lê esse registro e desenha os gráficos, não treina nada.

Gráficos salvos em results/figures/: task_curva_acuracia.png e task_curva_perda.png
Uso: python src/avaliacao/curvas_treino_task.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REGISTRO = ROOT_DIR / 'results' / 'treinos' / 'task.json'
OUTPUT_DIR = ROOT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

historico = json.loads(REGISTRO.read_text(encoding='utf-8'))['historico']
epocas = [h['epoca'] for h in historico]
loss_treino = [h['perda_treino'] for h in historico]
loss_val = [h['perda_validacao'] for h in historico]
acc_treino = [h['acuracia_treino'] for h in historico]
acc_val = [h['acuracia_validacao'] for h in historico]

# Com muitas épocas, marcar cada ponto e cada número no eixo poluiria o gráfico
marcador = dict(marker='o') if len(epocas) <= 30 else {}
marcador_val = dict(marker='s') if len(epocas) <= 30 else {}

# Configuração geral do visual
plt.style.use('seaborn-v0_8-darkgrid')

# ---------------------------------------------------------
# GRÁFICO 1: ACURÁCIA (Evolução dos Acertos)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epocas, acc_treino, linestyle='-', color='blue', label='Acurácia (Treinamento)', **marcador)
plt.plot(epocas, acc_val, linestyle='-', color='orange', label='Acurácia (Validação)', **marcador_val)
plt.title('Evolução da Acurácia por Época', fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
if len(epocas) <= 30:
    plt.xticks(epocas)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
caminho_acc = str(OUTPUT_DIR / 'task_curva_acuracia.png')
plt.savefig(caminho_acc, dpi=300)
print(f"Gráfico de acurácia salvo como '{caminho_acc}'")

# ---------------------------------------------------------
# GRÁFICO 2: PERDA / LOSS (Evolução do Erro)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epocas, loss_treino, linestyle='-', color='red', label='Perda (Treinamento)', **marcador)
plt.plot(epocas, loss_val, linestyle='-', color='green', label='Perda (Validação)', **marcador_val)
plt.title('Evolução da Perda (Loss) por Época', fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Perda', fontsize=12)
if len(epocas) <= 30:
    plt.xticks(epocas)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
caminho_perda = str(OUTPUT_DIR / 'task_curva_perda.png')
plt.savefig(caminho_perda, dpi=300)
print(f"Gráfico de perda salvo como '{caminho_perda}'")

plt.show()
