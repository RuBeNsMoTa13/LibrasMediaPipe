---
title: LibrasMediaPipe
emoji: 🐠
colorFrom: yellow
colorTo: purple
sdk: static
app_file: web/index.html
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
       | src/desktop/detectar_...|                    |     web/index.html      |
       |  - OpenCV VideoCapture  |                    |  - MediaPipe JS (WASM)  |
       |  - Latência Zero (Local)|                    |  - Roda no navegador    |
       |  - TTS Nativo (pyttsx3) |                    |  - HF Space estático    |
       |  - Buffer de Soletração |                    |  - Acesso via Navegador |
       |  - 30+ FPS direto no SO |                    |  - Sem Instalação Local |
       +-------------------------+                    +-------------------------+
```

1. **Aplicação Desktop Local (`src/desktop/detectar_libras.py`):**
   * Acesso nativo à webcam local via OpenCV sem overhead de rede (30 a 60 FPS fluidos).
   * **Buffer acumulador de soletração:** Forma palavras com controle de repetição (*debounce* de 0.7s), suporte a comandos de espaço e deleção.
   * **Text-to-Speech (TTS):** Síntese de voz assíncrona com `pyttsx3` conectada ao driver de áudio do sistema operacional, pronunciando a palavra soletrada após inatividade.
   * Ideal para demonstrações presenciais da banca avaliadora e alta performance.

2. **Aplicação Web no Navegador (`web/index.html`):**
   * Executa o mesmo `gesture_recognizer.task` direto no navegador com `@mediapipe/tasks-vision` (WebAssembly + WebGL), sem servidor de inferência: o vídeo nunca sai da máquina do usuário.
   * Mesmos recursos da versão desktop: buffer de soletração (0,7 s entre letras), limiar de confiança ajustável e voz em pt-BR pela Web Speech API.
   * Publicada como Space estático no Hugging Face: <https://rubensmota13-librasmediapipe.static.hf.space>.
   * Substitui a primeira versão em Gradio/Docker (`src/web/app.py`), que lagava por enviar cada frame ao servidor e de volta (cerca de 1,5 s de atraso).

> Para entender por que a versão em Gradio lagava e por que a versão no navegador funciona, leia [docs/ambientes/versao-web-navegador.md](docs/ambientes/versao-web-navegador.md). A análise original de latência está em [docs/ambientes/local-vs-huggingface.md](docs/ambientes/local-vs-huggingface.md).

---

## 🧠 Pipeline de Inteligência Artificial

1. **Captura & Pré-processamento:** Leitura do frame, conversão BGR $\rightarrow$ RGB e espelhamento horizontal.
2. **Extração de Atributos:** O MediaPipe Hand Landmarker identifica 21 coordenadas tridimensionais da mão ($[x_0, y_0, z_0, ..., x_{20}, y_{20}, z_{20}]$).
3. **Classificação:**
   * **MediaPipe Gesture Recognizer:** Modelo pré-compilado (`gesture_recognizer.task`) treinado especificamente nas 21 letras da LIBRAS.
   * **Random Forest, SVM e Spiking Neural Network (SNN):** classificam o mesmo "resumo" de 128 números da mão que o `.task` usa por dentro (a rede `gesture_embedder.tflite`, lida de dentro do `.task` por `src/classificadores/resumo.py`). Random Forest (100 árvores) e SVM (kernel RBF) via Scikit-Learn; SNN com neurônios LIF via snnTorch.
4. **Pós-processamento e Acessibilidade:** Filtragem por limiar de confiança ($\ge 75\%$), buffer de caracteres e sintetização de fala.

---

## 📁 Estrutura do Repositório

```
LibrasMediaPipe/
├── GEMINI.md                           # Fonte primária da verdade arquitetural
├── README.md                           # Documentação central do projeto
├── Dockerfile                          # Container Docker da versão Gradio (legado)
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
│   ├── gesture_recognizer.task         # Modelo do MediaPipe Tasks
│   ├── rf_libras.pkl / svm_libras.pkl  # Random Forest e SVM (testar_snn.py)
│   └── snn_libras.pt                   # Spiking Neural Network (testar_snn.py)
│
├── results/                            # Saídas científicas geradas
│   ├── figures/                        # Gráficos (ver "Resultados e Benchmarks")
│   │   ├── grafico_acuracia.png        # Treino do .task (Colab): acurácia por época
│   │   ├── grafico_perda.png           # Treino do .task (Colab): perda por época
│   │   ├── matriz_de_confusao.png      # Matriz do .task
│   │   ├── matriz_confusao_rf.png      # Matriz do Random Forest
│   │   ├── matriz_confusao_svm.png     # Matriz do SVM
│   │   ├── matriz_confusao_snn.png     # Matriz da SNN
│   │   ├── snn_curva_perda.png         # Perda da SNN por época
│   │   ├── snn_curva_acuracia.png      # Acurácia da SNN (treino e teste) por época
│   │   ├── comparacao_modelos.png      # Acurácia, precisão, recall e F1 dos 4 modelos
│   │   ├── f1_por_letra.png            # F1 de cada letra em cada modelo
│   │   ├── distribuicao_amostras.png   # Imagens por letra no treino e no teste
│   │   └── deteccao_maos_por_letra.png # % de fotos em que o MediaPipe achou a mão
│   └── tables/                         # Caches de landmarks (.npz)
│
├── web/                                # Aplicação web no navegador (Space estático)
│   └── index.html                      # MediaPipe JS + webcam + buffer + voz
│
├── src/                                # Código-fonte da aplicação
│   ├── classificadores/                # RF, SVM e SNN usados no app
│   │   ├── resumo.py                   # Resumo de 128 números da mão (rede do .task)
│   │   ├── classicos.py                # Random Forest e SVM
│   │   └── snn.py                      # Spiking Neural Network
│   ├── desktop/                        # Aplicação local nativa
│   │   └── detectar_libras.py          # OpenCV + buffer de digitação + TTS
│   ├── web/                            # Primeira versão web (legado)
│   │   └── app.py                      # Servidor Gradio (antigo Space Docker)
│   └── evaluation/                     # Scripts de avaliação e benchmark
│       ├── comparar_modelos.py         # Benchmark: Random Forest vs SVM (LaTeX)
│       ├── gerar_metricas.py           # Relatório de classificação e matriz
│       ├── graficos.py                 # Curvas de perda e acurácia por época
│       ├── testar_snn.py               # Treina e salva RF, SVM e SNN
│       └── graficos_modelos.py         # Gráficos de comparação dos 4 modelos
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
* **Trocar de classificador na webcam:** a tecla `M` passa por MediaPipe `.task` → SNN → Random Forest → SVM, e a barra superior mostra o modelo ativo. Nos três últimos, o MediaPipe só extrai os landmarks e o modelo salvo em `models/` (`snn_libras.pt`, `rf_libras.pkl`, `svm_libras.pkl`) classifica a letra em cada frame. A SNN precisa de `pip install torch snntorch`; Random Forest e SVM precisam de `pip install scikit-learn`. Para já abrir num modelo, use `--modelo task|snn|rf|svm`:
  ```powershell
  python src/desktop/detectar_libras.py --modelo snn
  ```

### 2. Aplicação Web no Navegador (Local ou Cloud)
Online: <https://rubensmota13-librasmediapipe.static.hf.space>

Para rodar localmente, sirva a raiz do repositório (a câmera só abre em HTTPS ou `localhost`):
```powershell
python -m http.server
```
* Abra no navegador: [http://localhost:8000/web/](http://localhost:8000/web/).
* A versão antiga em Gradio ainda pode ser executada com `python src/web/app.py` (porta 7860).

### 3. Benchmarks e Métricas para a Monografia

* **Comparação entre Modelos Clássicos (Random Forest vs. SVM):**
  Extrai os 21 landmarks dos datasets de treino e teste e gera os valores formatados para tabela LaTeX:
  ```powershell
  python src/evaluation/comparar_modelos.py
  ```

* **Spiking Neural Network (SNN) vs. Random Forest vs. SVM:**
  Treina os três modelos sobre o resumo de 128 números da mão (o mesmo que o `.task` usa), imprime as métricas e salva os modelos em `models/` para o app da webcam. Precisa de `pip install torch snntorch scikit-learn ai-edge-litert`; os landmarks ficam em cache em `results/tables/resumo_libras.npz`:
  ```powershell
  python src/evaluation/testar_snn.py
  ```

* **Gráficos de comparação dos quatro modelos:**
  Gera em `results/figures/` as matrizes de confusão de RF, SVM e SNN, as curvas da SNN, a comparação de métricas, o F1 por letra, a distribuição das amostras e a detecção de mãos por letra (lista completa em [Resultados e Benchmarks](#-resultados-e-benchmarks)):
  ```powershell
  python src/evaluation/graficos_modelos.py
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

### 4. Execução via Docker (versão Gradio, legado)
Para testar a imagem da primeira versão web, como o container era inicializado no Hugging Face:
```bash
docker build -t libras-mediapipe .
docker run -p 7860:7860 libras-mediapipe
```

---

## 📊 Resultados e Benchmarks

O pipeline científico do projeto permite comparar diretamente diferentes abordagens de aprendizado de máquina para o reconhecimento de gestos:

Todos os modelos são avaliados nas mesmas fotos de `data/libras/test` em que o MediaPipe encontrou uma mão (valores gerados por `graficos_modelos.py`):

| Modelo | Acurácia | Precisão Ponderada | Recall Ponderado | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Tasks (`gesture_recognizer.task`)** | 0,891 | 0,931 | 0,891 | 0,863 |
| **Random Forest (100 árvores)** | 0,994 | 0,994 | 0,994 | 0,994 |
| **Support Vector Machine (SVM - RBF)** | 0,982 | 0,986 | 0,982 | 0,983 |
| **Spiking Neural Network (SNN - LIF)** | 0,990 | 0,991 | 0,990 | 0,990 |

> **Atenção ao ler esses números:** as fotos de treino e de teste do dataset vêm dos mesmos vídeos e, aparentemente, da mesma pessoa, então o teste é muito parecido com o treino. Por isso RF, SVM e SNN ficam perto de 100% aqui, mas erram mais ao vivo com uma mão nova (por exemplo, U confundido com R ou V). Essas métricas medem o desempenho no dataset, não com um usuário novo.

### Gráficos gerados

Todos ficam em [`results/figures/`](results/figures).

| Gráfico | O que mostra | Script |
| :--- | :--- | :--- |
| `grafico_acuracia.png` / `grafico_perda.png` | Acurácia e perda por época do treino do `.task` (Colab) | `graficos.py` |
| `matriz_de_confusao.png` | Matriz de confusão do `.task` | `gerar_metricas.py` |
| `matriz_confusao_rf.png` | Matriz de confusão do Random Forest | `graficos_modelos.py` |
| `matriz_confusao_svm.png` | Matriz de confusão do SVM | `graficos_modelos.py` |
| `matriz_confusao_snn.png` | Matriz de confusão da SNN | `graficos_modelos.py` |
| `snn_curva_perda.png` | Perda da SNN a cada época | `graficos_modelos.py` |
| `snn_curva_acuracia.png` | Acurácia da SNN no treino e no teste a cada época | `graficos_modelos.py` |
| `comparacao_modelos.png` | Acurácia, precisão, recall e F1 dos quatro modelos lado a lado | `graficos_modelos.py` |
| `f1_por_letra.png` | F1 de cada letra em cada modelo (mapa de calor) | `graficos_modelos.py` |
| `distribuicao_amostras.png` | Quantidade de imagens por letra no treino e no teste | `graficos_modelos.py` |
| `deteccao_maos_por_letra.png` | % das fotos de teste em que o MediaPipe encontrou a mão | `graficos_modelos.py` |

---

## 📚 Documentação Técnica Completa

Para aprofundar-se na metodologia, decisões de arquitetura e backlog do TCC, consulte a documentação dedicada na pasta `docs/`:

* 📖 **[Índice de Documentação Técnica](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/README.md)**
* 🌐 **[Versão Web no Navegador: por que lagava e por que agora funciona](docs/ambientes/versao-web-navegador.md)**
* ⚖️ **[Estudo Técnico: Ambiente Local vs. Hugging Face Spaces](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md)**
* 📋 **[Backlog Oficial e Tarefas do TCC](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/todo.md)**
* 🏛️ **[Diretrizes Arquiteturais (GEMINI.md)](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/GEMINI.md)**

---

## 👨‍💻 Autoria e Informações Acadêmicas

* **Curso:** Bacharelado em Ciência de Dados e Inteligência Artificial
* **Tema:** Reconhecimento do Alfabeto Manual de LIBRAS em Tempo Real com MediaPipe e Aprendizado de Máquina
* **Autor:** Rubens Mota
