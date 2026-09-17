import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seus dados extraídos da tabela
epocas = list(range(1, 11))

# Métricas de Perda (Loss)
loss_treino = [1.3314, 0.7486, 0.6605, 0.6084, 0.5731, 0.5459, 0.5371, 0.5240, 0.5149, 0.5085]
loss_val = [0.2241, 0.2224, 0.2201, 0.2352, 0.2198, 0.2384, 0.2432, 0.2300, 0.2059, 0.1948]

# Métricas de Acurácia
acc_treino = [0.5239, 0.7054, 0.7335, 0.7539, 0.7715, 0.7816, 0.7836, 0.7907, 0.7951, 0.7907]
acc_val = [0.8304, 0.8700, 0.8546, 0.8348, 0.8480, 0.8348, 0.8568, 0.8634, 0.8568, 0.8634]

# Configuração geral do visual
plt.style.use('seaborn-v0_8-darkgrid')

# ---------------------------------------------------------
# GRÁFICO 1: ACURÁCIA (Evolução dos Acertos)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epocas, acc_treino, marker='o', linestyle='-', color='blue', label='Acurácia (Treinamento)')
plt.plot(epocas, acc_val, marker='s', linestyle='-', color='orange', label='Acurácia (Validação)')
plt.title('Evolução da Acurácia por Época', fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.xticks(epocas)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
caminho_acc = str(OUTPUT_DIR / 'grafico_acuracia.png')
plt.savefig(caminho_acc, dpi=300)
print(f"Gráfico de acurácia salvo como '{caminho_acc}'")

# ---------------------------------------------------------
# GRÁFICO 2: PERDA / LOSS (Evolução do Erro)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epocas, loss_treino, marker='o', linestyle='-', color='red', label='Perda (Treinamento)')
plt.plot(epocas, loss_val, marker='s', linestyle='-', color='green', label='Perda (Validação)')
plt.title('Evolução da Perda (Loss) por Época', fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Perda', fontsize=12)
plt.xticks(epocas)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
caminho_perda = str(OUTPUT_DIR / 'grafico_perda.png')
plt.savefig(caminho_perda, dpi=300)
print(f"Gráfico de perda salvo como '{caminho_perda}'")

plt.show()