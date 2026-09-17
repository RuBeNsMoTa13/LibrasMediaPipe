---
title: LibrasMediaPipe
emoji: 🐠
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# LibrasMediaPipe 🤟🤖

> **Trabalho de Conclusão de Curso (TCC) em Ciência de Dados e Inteligência Artificial**  
> Reconhecimento e tradução em tempo real do alfabeto manual da Língua Brasileira de Sinais (LIBRAS) utilizando visão computacional, esqueletização por *landmarks* tridimensionais (Google MediaPipe) e modelos de *Machine Learning*.

---

## 📌 Sumário
* [Visão Geral](#-visão-geral)
* [Dualidade de Ambientes (Local vs. Cloud)](#-dualidade-de-ambientes-local-vs-cloud)
* [Pipeline de Inteligência Artificial](#-pipeline-de-inteligência-artificial)
* [Estrutura do Repositório](#-estrutura-do-repositório)
* [Instalação e Pré-requisitos](#-instalação-e-pré-requisitos)
* [Como Executar](#-como-executar)
* [Resultados e Benchmarks](#-resultados-e-benchmarks)
* [Documentação Técnica Completa](#-documentação-técnica-completa)
* [Licença e Autoria](#-licença-e-autoria)

---

## 📖 Visão Geral

O **LibrasMediaPipe** é um projeto de acessibilidade e inteligência artificial aplicada projetado para interpretar gestos do alfabeto estático de LIBRAS (21 classes: `A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y`).

Em vez de processar matrizes pesadas de pixels brutos com redes neurais convolucionais profundas (que demandam alto poder computacional e GPUs dedicadas), a solução emprega a **esqueletização das mãos via Google MediaPipe Tasks API**, extraindo **21 pontos anatômicos tridimensionais (63 coordenadas $x, y, z$)**. Essa redução drástica da dimensionalidade viabiliza inferências ultrarrápidas em tempo real, inclusive em computadores modestos e dispositivos de borda (*Edge Computing*).

---

## 🔄 Dualidade de Ambientes (Local vs. Cloud)

O projeto foi projetado com uma arquitetura dual para atender a dois propósitos distintos:

```
                      +------------------------------------------+
                      |            LibrasMediaPipe               |
                      +------------------------------------------+
                                    /              \
                                   /                \
        [Ambiente Local Desktop]  /                  \  [Ambiente Cloud Web]
       +-------------------------+                    +-------------------------+
       | src/desktop/detectar_...|                    |     src/web/app.py      |
       |  - OpenCV VideoCapture  |                    |  - Gradio Web Interface |
       |  - Latência Zero (Local)|                    |  - Docker Container     |
       |  - TTS Nativo (pyttsx3) |                    |  - Hugging Face Spaces  |
       |  - Buffer de Soletração |                    |  - Acesso via Navegador |
       |  - 30+ FPS direto no SO |                    |  - Sem Instalação Local |
       +-------------------------+                    +-------------------------+
```

1. **Aplicação Desktop Local (`src/desktop/detectar_libras.py`):**
   * Acesso nativo à webcam local via OpenCV sem overhead de rede (30 a 60 FPS fluidos).
   * **Buffer acumulador de soletração:** Forma palavras com controle de repetição (*debounce* de 0.7s), suporte a comandos de espaço e deleção.
   * **Text-to-Speech (TTS):** Síntese de voz assíncrona com `pyttsx3` conectada ao driver de áudio do sistema operacional, pronunciando a palavra soletrada após inatividade.
   * Ideal para demonstrações presenciais da banca avaliadora e alta performance.

2. **Aplicação Cloud Web (`src/web/app.py`):**
   * Interface acessível por navegador web desenvolvida com Gradio e empacotada em container Docker.
   * Otimizações de rede: *downscaling* agressivo de entrada (`DOWNSCALE_WIDTH = 192`), *frame-skipping* (`PROCESS_EVERY_N = 3`) e *cache* de predições.
   * Hospedado no Hugging Face Spaces para demonstração pública instantânea sem necessidade de instalar dependências.

> Para entender detalhadamente o impacto de latência de rede (RTT) e estratégias de mitigação no Spaces, leia o documento [docs/ambientes/local-vs-huggingface.md](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md).

---

## 🧠 Pipeline de Inteligência Artificial

1. **Captura & Pré-processamento:** Leitura do frame, conversão BGR $\rightarrow$ RGB e espelhamento horizontal.
2. **Extração de Atributos:** O MediaPipe Hand Landmarker identifica 21 coordenadas tridimensionais da mão ($[x_0, y_0, z_0, ..., x_{20}, y_{20}, z_{20}]$).
3. **Classificação:**
   * **MediaPipe Gesture Recognizer:** Modelo pré-compilado (`gesture_recognizer.task`) treinado especificamente nas 21 letras da LIBRAS.
   * **Modelos Clássicos Supervisionados:** Extração tabular das 63 coordenadas para benchmark científico com **Random Forest** (100 árvores) e **Support Vector Machine (SVM com kernel RBF)** via Scikit-Learn.
4. **Pós-processamento e Acessibilidade:** Filtragem por limiar de confiança ($\ge 75\%$), buffer de caracteres e sintetização de fala.

---

## 📁 Estrutura do Repositório

```
LibrasMediaPipe/
├── GEMINI.md                           # Fonte primária da verdade arquitetural
├── README.md                           # Documentação central do projeto
├── Dockerfile                          # Configuração do container Docker (Hugging Face)
├── requirements.txt                    # Dependências Python do projeto
├── .gitignore                          # Exclusões de arquivos de compilação e cache
│
├── data/                               # Datasets estruturados
│   ├── libras/                         # Treino e teste para ML clássico (A..Y)
│   │   ├── train/
│   │   └── test/
│   └── teste_libras/ (A..Y)            # Imagens para validação do MediaPipe Tasks
│
├── models/                             # Modelos e pesos compilados
│   └── gesture_recognizer.task         # Modelo do MediaPipe Tasks
│
├── results/                            # Saídas científicas geradas
│   ├── figures/                        # Gráficos e matriz de confusão
│   │   ├── grafico_acuracia.png
│   │   ├── grafico_perda.png
│   │   └── matriz_de_confusao.png
│   └── tables/                         # Relatórios tabulares
│
├── src/                                # Código-fonte da aplicação
│   ├── desktop/                        # Aplicação local nativa
│   │   └── detectar_libras.py          # OpenCV + buffer de digitação + TTS
│   ├── web/                            # Aplicação web em nuvem
│   │   └── app.py                      # Servidor Gradio para Hugging Face Spaces
│   └── evaluation/                     # Scripts de avaliação e benchmark
│       ├── comparar_modelos.py         # Benchmark: Random Forest vs SVM (LaTeX)
│       ├── gerar_metricas.py           # Relatório de classificação e matriz
│       └── graficos.py                 # Curvas de perda e acurácia por época
│
├── docs/                               # Governança e auditoria técnica do TCC
│   ├── README.md                       # Índice de documentação técnica
│   ├── todo.md                         # Backlog oficial do projeto
│   └── ambientes/                      # Estudo aprofundado: Local vs. Cloud
└── .agents/                            # Configurações e regras locais do workspace
```

---

## 💻 Instalação e Pré-requisitos

### Pré-requisitos:
* Python 3.10, 3.11 ou 3.12 instalado.
* Webcam conectada (para a aplicação em tempo real).

### Passo a Passo:

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/RuBeNsMoTa13/LibrasMediaPipe.git
   cd LibrasMediaPipe
   ```

2. **Crie e ative um ambiente virtual:**
   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Como Executar

### 1. Aplicação Desktop Local (Recomendado para Apresentações)
Executa a visão computacional localmente com síntese de voz (TTS) e buffer de escrita:
```powershell
python src/desktop/detectar_libras.py
```
* **Controles:** Posicione a mão em frente à câmera com boa iluminação. Pressione `q` para sair.

### 2. Aplicação Web Gradio (Local ou Cloud)
Inicia o servidor web local idêntico ao ambiente do Hugging Face:
```powershell
python src/web/app.py
```
* Abra no navegador: [http://localhost:7860](http://localhost:7860).

### 3. Benchmarks e Métricas para a Monografia

* **Comparação entre Modelos Clássicos (Random Forest vs. SVM):**
  Extrai os 21 landmarks dos datasets de treino e teste e gera os valores formatados para tabela LaTeX:
  ```powershell
  python src/evaluation/comparar_modelos.py
  ```

* **Matriz de Confusão e Relatório de Classificação:**
  Avalia o modelo MediaPipe Tasks contra as amostras de teste e salva a figura em `results/figures/matriz_de_confusao.png`:
  ```powershell
  python src/evaluation/gerar_metricas.py
  ```

* **Curvas de Aprendizado (Acurácia e Loss):**
  Plota e salva as curvas de evolução por época em `results/figures/`:
  ```powershell
  python src/evaluation/graficos.py
  ```

### 4. Execução via Docker
Para testar a imagem exatamente como o container é inicializado no Hugging Face:
```bash
docker build -t libras-mediapipe .
docker run -p 7860:7860 libras-mediapipe
```

---

## 📊 Resultados e Benchmarks

O pipeline científico do projeto permite comparar diretamente diferentes abordagens de aprendizado de máquina para o reconhecimento de gestos:

| Modelo | Acurácia | Precisão Ponderada | Recall Ponderado | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Tasks (`gesture_recognizer.task`)** | Alta (tempo real) | Robusto | Robusto | Alta generalização |
| **Random Forest (100 estimators)** | Avaliado via script | Avaliado via script | Avaliado via script | Avaliado via script |
| **Support Vector Machine (SVM - RBF)** | Avaliado via script | Avaliado via script | Avaliado via script | Avaliado via script |

As matrizes de confusão e gráficos gerados encontram-se salvos no diretório [`results/figures/`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/results/figures).

---

## 📚 Documentação Técnica Completa

Para aprofundar-se na metodologia, decisões de arquitetura e backlog do TCC, consulte a documentação dedicada na pasta `docs/`:

* 📖 **[Índice de Documentação Técnica](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/README.md)**
* ⚖️ **[Estudo Técnico: Ambiente Local vs. Hugging Face Spaces](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md)**
* 📋 **[Backlog Oficial e Tarefas do TCC](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/todo.md)**
* 🏛️ **[Diretrizes Arquiteturais (GEMINI.md)](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/GEMINI.md)**

---

## 👨‍💻 Autoria e Informações Acadêmicas

* **Curso:** Bacharelado em Ciência de Dados e Inteligência Artificial
* **Tema:** Reconhecimento do Alfabeto Manual de LIBRAS em Tempo Real com MediaPipe e Aprendizado de Máquina
* **Autor:** Rubens Mota
