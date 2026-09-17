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

1. **Ambiente Local (Desktop - `detectar_libras.py`):**
   * Acesso direto ao hardware de câmera via OpenCV (`cv2.VideoCapture(0)`).
   * Sem overhead de rede: processamento imediato a 30-60 FPS.
   * Síntese de voz local (`pyttsx3`) com saída de áudio nativa no sistema operacional.
   * Ideal para demonstrações presenciais da banca e testes de alta fidelidade.

2. **Ambiente Remoto (Cloud - `app.py` / Hugging Face Spaces):**
   * Interface Web acessível por navegador via Gradio.
   * Empacotamento em container Docker (`Dockerfile`).
   * Desafio de latência de rede (RTT de envio de frame e retorno da imagem anotada).
   * Requer otimizações severas de decimação e frame-skipping para mitigar engasgos (*lag*).

> Para uma análise técnica aprofundada dos gargalos e soluções entre esses dois ambientes, consulte [docs/ambientes/local-vs-huggingface.md](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md).

---

## 3. Estrutura do Repositório

```
LibrasMediaPipe/
├── GEMINI.md                    # Fonte primária da verdade arquitetural
├── README.md                    # Configuração do Space no Hugging Face
├── Dockerfile                   # Configuração de build e runtime Linux/Docker
├── requirements.txt             # Dependências Python do projeto
├── gesture_recognizer.task      # Modelo compilado MediaPipe Tasks
│
├── app.py                       # Servidor Web Gradio para Hugging Face Spaces
├── detectar_libras.py           # Aplicação desktop OpenCV com buffer e TTS
├── comparar_modelos.py          # Benchmark: Random Forest vs SVM com 63 landmarks
├── gerar_metricas.py            # Avaliação de imagens, relatório e matriz de confusão
├── graficos.py                  # Gráficos de curvas de aprendizado (acurácia e loss)
│
├── libras/                      # Dataset para treino e teste de ML clássico
│   ├── train/                   # Subpastas A..Y com amostras de treino
│   └── test/                    # Subpastas A..Y com amostras de teste
├── teste_libras/                # Amostras para teste do gesture_recognizer.task
│
├── docs/                        # Documentação técnica e auditoria
│   ├── README.md                # Índice navegável de documentação
│   ├── todo.md                  # Backlog oficial rastreável do TCC
│   └── ambientes/
│       └── local-vs-huggingface.md # Detalhamento dos ambientes Local vs Spaces
└── .agents/                     # Regras e workflows locais
```

---

## 4. Comandos de Execução

### 4.1. Execução Local (Recomendado para Demonstração e Testes)
```powershell
python detectar_libras.py
```
* Pressione `q` para encerrar a janela da câmera.

### 4.2. Execução do Servidor Web (Gradio / Hugging Face Local)
```powershell
python app.py
```
* Acesse no navegador: `http://localhost:7860`.

### 4.3. Benchmark e Métricas para a Monografia
```powershell
# Compara Random Forest vs SVM e gera tabela LaTeX:
python comparar_modelos.py

# Avalia o modelo MediaPipe e gera matriz_de_confusao.png:
python gerar_metricas.py

# Plota as curvas de acurácia e perda do treinamento:
python graficos.py
```

---

## 5. Convenções do Projeto

* **Commits:** Mensagens devem seguir o padrão Conventional Commits (ex: `feat:`, `fix:`, `docs:`, `perf:`). Commits nunca são executados de forma autônoma pelo assistente.
* **Documentação:** Qualquer nova documentação técnica deve ser criada sob a pasta `docs/` e indexada no `docs/README.md`.
