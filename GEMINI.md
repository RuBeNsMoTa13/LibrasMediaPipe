# GEMINI.md - Diretrizes e Arquitetura do Projeto LibrasMediaPipe

Este documento consolida a arquitetura, as convenções e os modos de operação do projeto de TCC **LibrasMediaPipe** (Reconhecimento de LIBRAS em Tempo Real utilizando MediaPipe e Modelos de Machine Learning).

---

## 1. Visão Geral e Objetivo

* **Objetivo Acadêmico:** Tradução automatizada de sinais do alfabeto manual da Língua Brasileira de Sinais (LIBRAS) em tempo real, comparando abordagens de Visão Computacional moderna (MediaPipe Tasks) com algoritmos clássicos de Machine Learning (Random Forest e SVM).
* **Entrada de Dados:** Frames de vídeo / imagens capturando a mão do usuário.
* **Extração de Atributos:** 21 landmarks tridimensionais (63 coordenadas: $x, y, z$) extraídos pelo Google MediaPipe.
* **Saída:** Identificação da letra com score de confiança, buffer acumulador de texto (soletração) e sintetização de voz (TTS).

---

## 2. Modos de Operação (Dualidade de Ambientes)

O projeto possui dois ambientes de execução distintos com arquiteturas e propósitos próprios:

```
                      +------------------------------------------+
                      |            LibrasMediaPipe               |
                      +------------------------------------------+
                                    /              \
                                   /                \
        [Ambiente Local Desktop]  /                  \  [Ambiente Cloud Web]
       +-------------------------+                    +-------------------------+
       |   detectar_libras.py    |                    |         app.py          |
       |  - OpenCV VideoCapture  |                    |  - Gradio WebRTC/Stream |
       |  - Latência Zero        |                    |  - Docker Container     |
       |  - TTS Nativo (pyttsx3) |                    |  - Hugging Face Spaces  |
       |  - 30+ FPS direto no SO |                    |  - Sujeito a RTT de rede|
       +-------------------------+                    +-------------------------+
```

1. **Ambiente Local (Desktop - `src/desktop/detectar_libras.py`):**
   * Acesso direto ao hardware de câmera via OpenCV (`cv2.VideoCapture(0)`).
   * Sem overhead de rede: processamento imediato a 30-60 FPS.
   * Síntese de voz local (`pyttsx3`) com saída de áudio nativa no sistema operacional.
   * Ideal para demonstrações presenciais da banca e testes de alta fidelidade.

2. **Ambiente Remoto (Cloud - `src/web/app.py` / Hugging Face Spaces):**
   * Interface Web acessível por navegador via Gradio.
   * Empacotamento em container Docker (`Dockerfile`).
   * Desafio de latência de rede (RTT de envio de frame e retorno da imagem anotada).
   * Requer otimizações severas de decimação e frame-skipping para mitigar engasgos (*lag*).

> Para uma análise técnica aprofundada dos gargalos e soluções entre esses dois ambientes, consulte [docs/ambientes/local-vs-huggingface.md](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md).

---

## 3. Estrutura do Repositório

```
LibrasMediaPipe/
├── GEMINI.md                           # Fonte primária da verdade arquitetural
├── README.md                           # Configuração do Space no Hugging Face
├── Dockerfile                          # Build Docker apontando para src/web/app.py
├── requirements.txt                    # Dependências Python do projeto
│
├── data/                               # Datasets estruturados
│   └── libras/                         # Dataset de treino e teste (todos os modelos)
│       ├── train/ (A..Y)
│       └── test/ (A..Y)
│
├── models/                             # Modelos e pesos de Machine Learning
│   └── gesture_recognizer.task         # Modelo compilado MediaPipe Tasks
│
├── results/                            # Saídas gráficas e relatórios gerados
│   ├── figures/                        # Gráficos e matriz de confusão
│   │   ├── grafico_acuracia.png
│   │   ├── grafico_perda.png
│   │   ├── matriz_de_confusao.png      # Matriz do .task
│   │   ├── matriz_confusao_{rf,svm,snn}.png
│   │   ├── snn_curva_{perda,acuracia}.png
│   │   ├── comparacao_modelos.png / f1_por_letra.png
│   │   └── distribuicao_amostras.png / deteccao_maos_por_letra.png
│   └── tables/                         # Caches de landmarks (.npz)
│
├── src/                                # Código-fonte do projeto
│   ├── desktop/                        # Aplicação local desktop
│   │   └── detectar_libras.py          # OpenCV + buffer de soletração + TTS
│   ├── web/                            # Aplicação web (Gradio)
│   │   └── app.py                      # Servidor Web para Hugging Face Spaces
│   └── evaluation/                     # Benchmarks e avaliação científica
│       ├── comparar_modelos.py         # Benchmark: Random Forest vs SVM (LaTeX)
│       ├── gerar_metricas.py           # Relatório de classificação e matriz
│       ├── graficos.py                 # Curvas de aprendizado (acurácia e loss)
│       ├── testar_snn.py               # Treina e salva RF, SVM e SNN
│       └── graficos_modelos.py         # Gráficos de comparação dos 4 modelos
│
├── docs/                               # Documentação técnica e auditoria
│   ├── README.md                       # Índice navegável de documentação
│   ├── todo.md                         # Backlog oficial rastreável do TCC
│   └── ambientes/
│       └── local-vs-huggingface.md     # Detalhamento dos ambientes Local vs Spaces
└── .agents/                            # Regras e governança do workspace
```

---

## 4. Comandos de Execução

### 4.1. Execução Local (Recomendado para Demonstração e Testes)
```powershell
python src/desktop/detectar_libras.py
```
* Pressione `q` para encerrar a janela da câmera.

### 4.2. Execução do Servidor Web (Gradio / Hugging Face Local)
```powershell
python src/web/app.py
```
* Acesse no navegador: `http://localhost:7860`.

### 4.3. Benchmark e Métricas para a Monografia
```powershell
# Compara Random Forest vs SVM e gera tabela LaTeX:
python src/evaluation/comparar_modelos.py

# Avalia o modelo MediaPipe e gera matriz_de_confusao.png:
python src/evaluation/gerar_metricas.py

# Plota as curvas de acurácia e perda do treinamento:
python src/evaluation/graficos.py

# Treina RF, SVM e SNN e salva em models/:
python src/evaluation/testar_snn.py

# Gera matrizes de RF/SVM/SNN, curvas da SNN, comparação e F1 por letra:
python src/evaluation/graficos_modelos.py
```


---

## 5. Convenções do Projeto

* **Commits:** Mensagens devem seguir o padrão Conventional Commits (ex: `feat:`, `fix:`, `docs:`, `perf:`). Commits nunca são executados de forma autônoma pelo assistente.
* **Documentação:** Qualquer nova documentação técnica deve ser criada sob a pasta `docs/` e indexada no `docs/README.md`.
