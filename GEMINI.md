# GEMINI.md - Diretrizes e Arquitetura do Projeto LibrasMediaPipe

Este documento consolida a arquitetura, as convenções e os modos de operação do projeto de TCC **LibrasMediaPipe** (reconhecimento do alfabeto manual de LIBRAS em tempo real com o MediaPipe e modelos de Machine Learning). Ele orienta o assistente Gemini: descreve o que existe hoje no repositório, onde cada coisa fica e quais cuidados tomar ao falar dos resultados.

---

## 1. Visão Geral e Objetivo

* **Objetivo acadêmico:** reconhecer em tempo real as letras do alfabeto manual da Língua Brasileira de Sinais (LIBRAS) e comparar quatro classificadores sobre a mesma base. O primeiro é o modelo pronto do MediaPipe (`models/gesture_recognizer.task`), treinado no Google Colab com o MediaPipe Model Maker. Os outros três são treinados neste repositório: Random Forest, SVM e uma rede neural de impulsos (SNN, do inglês *Spiking Neural Network*). Os três classificam o mesmo "resumo" de 128 números da mão que o `.task` usa por dentro, para que a comparação seja justa.
* **Letras:** 21 letras, as que existem no dataset: A B C D E F G I L M N O P Q R S T U V W Y.
* **Entrada de dados:** imagens da webcam (nas aplicações) e fotos 64x64 do dataset `data/libras` (no treino e na avaliação).
* **Extração de atributos:** o MediaPipe encontra a mão e devolve 21 pontos (landmarks) na imagem, os mesmos 21 pontos em 3D, em metros, e se a mão é direita ou esquerda. A rede `gesture_embedder`, que vem dentro do `.task`, transforma essas informações no resumo de 128 números (`src/classificadores/resumo.py`).
* **Saída:** a letra com a sua confiança, um limiar de confiança ajustável (padrão de 50%), um buffer que junta as letras em palavras (soletração, com intervalo mínimo de 0,7 s entre letras) e a leitura do texto em voz alta.

Caminho de cada imagem da câmera:

```
frame da webcam (espelhado, como um espelho)
  -> MediaPipe encontra a mão: 21 landmarks na imagem + 21 landmarks 3D + mão direita/esquerda
  -> classificação, de uma de duas formas:
       (a) o próprio .task (gesture_embedder + cabeça treinada no Colab,
           que é a parte final do modelo e escolhe a letra), ou
       (b) gesture_embedder -> resumo de 128 números -> Random Forest, SVM ou SNN
  -> limiar de confiança -> buffer de soletração -> voz
```

---

## 2. Modos de Operação (Dualidade de Ambientes)

O projeto possui dois ambientes de execução, com propósitos diferentes:

```
                      +------------------------------------------+
                      |            LibrasMediaPipe               |
                      +------------------------------------------+
                                    /              \
                                   /                \
        [Ambiente Local Desktop]  /                  \  [Ambiente Web no Navegador]
       +---------------------------+              +-----------------------------+
       |   detectar_libras.py      |              |       web/index.html        |
       |  - Python + OpenCV        |              |  - MediaPipe JS (WASM)      |
       |  - .task, SNN, RF ou SVM  |              |  - somente o .task          |
       |  - tecla M troca o modelo |              |  - roda no navegador        |
       |  - voz nativa (pyttsx3)   |              |  - voz: Web Speech API      |
       +---------------------------+              |  - Space estático do HF     |
                                                  +-----------------------------+
```

1. **Ambiente Local (Desktop - `src/desktop/detectar_libras.py`):**
   * Lê a câmera diretamente com o OpenCV (`cv2.VideoCapture(0)`) e espelha cada frame (`cv2.flip`).
   * Permite escolher o classificador ao abrir (`--modelo task|snn|rf|svm`) e trocá-lo durante a execução com a tecla `M`. SNN, Random Forest e SVM só aparecem se os arquivos deles existirem em `models/`; eles são gerados por `src/treino/treinar_modelos.py`.
   * Fala o texto com o `pyttsx3`, quando ele está instalado.
   * É o ambiente indicado para a demonstração na banca e para comparar os quatro classificadores ao vivo.

2. **Ambiente Web (navegador - `web/index.html`):**
   * Um único arquivo HTML com JavaScript, sem servidor de aplicação. Ele usa a biblioteca JavaScript do MediaPipe (`@mediapipe/tasks-vision`), que roda no próprio navegador com WebAssembly. O vídeo nunca sai do computador de quem acessa.
   * Usa somente o `.task`. Random Forest, SVM e SNN existem apenas na versão em Python.
   * Tem os mesmos recursos de uso da versão local: limiar ajustável (50% por padrão), 0,7 s entre letras e voz em português pela Web Speech API do navegador.
   * Está publicado como Space estático do Hugging Face: <https://rubensmota13-librasmediapipe.static.hf.space>. O cabeçalho YAML do `README.md` configura esse Space (`sdk: static`, `app_file: web/index.html`) e não deve ser alterado.
   * No GitHub, o `.task` fica no Git LFS (aparece como um ponteiro pequeno). O Space precisa receber o arquivo real.
   * Histórico: a versão web anterior era uma interface Gradio (primeiro com o SDK Gradio do Hugging Face e, a partir de 26/05/2026, num container Docker), que mandava cada frame para o servidor e por isso atrasava. Ela foi removida do repositório.

> Para entender por que a versão web saiu do servidor e passou a rodar no navegador, consulte [docs/ambientes/versao-web-navegador.md](docs/ambientes/versao-web-navegador.md).

---

## 3. Estrutura do Repositório

```
LibrasMediaPipe/
├── GEMINI.md                           # Este guia para o assistente Gemini
├── README.md                           # Apresentação do projeto; o cabeçalho YAML configura o Space (não alterar)
├── requirements.txt                    # Dependências Python do projeto
├── .gitattributes                      # Git LFS para o .task (os .pkl e .pt ficam no git normal)
├── .gitignore
│
├── data/
│   └── libras/                         # Fotos 64x64, já recebidas divididas em treino e teste
│       ├── train/ (A..Y)               # 3.468 fotos
│       └── test/ (A..Y)                # 1.153 fotos
│
├── models/
│   ├── gesture_recognizer.task         # Modelo do MediaPipe Model Maker, treinado no Colab
│   ├── rf_libras.pkl                   # Random Forest
│   ├── svm_libras.pkl                  # SVM
│   └── snn_libras.pt                   # SNN
│
├── results/
│   ├── figures/                        # 12 gráficos (lista na seção 4.4)
│   └── tables/                         # Cache resumo_libras.npz (ignorado pelo git)
│
├── web/
│   └── index.html                      # App no navegador (MediaPipe JS), publicado no Space estático
│
├── src/
│   ├── classificadores/
│   │   ├── resumo.py                   # Lê o gesture_embedder de dentro do .task e gera o resumo de 128 números
│   │   ├── classicos.py                # Carrega o Random Forest e o SVM salvos, para o app
│   │   └── snn.py                      # Arquitetura e carregamento da SNN
│   ├── desktop/
│   │   └── detectar_libras.py          # App da webcam: OpenCV, soletração, voz e troca de modelo
│   ├── treino/
│   │   └── treinar_modelos.py          # Treina RF, SVM e SNN, salva em models/ e imprime a tabela LaTeX
│   └── avaliacao/
│       ├── graficos_modelos.py         # Gráficos de comparação dos quatro modelos
│       ├── matriz_confusao_task.py     # Matriz de confusão do .task (Figura 3 do TCC)
│       └── curvas_treino_task.py       # Curvas do treino do .task no Colab (Figuras 1 e 2 do TCC)
│
├── docs/
│   ├── README.md                       # Índice da documentação
│   ├── todo.md                         # Backlog do TCC
│   ├── ambientes/
│   │   └── versao-web-navegador.md     # Por que a versão web passou a rodar no navegador
│   └── avaliacao/
│       └── metodologia-e-limitacoes.md # Como as métricas foram medidas e o que elas não mostram
│
└── .agents/                            # Regras do workspace
```

Na limpeza de outubro de 2026, a pasta `src/evaluation` foi dividida em `src/treino` e `src/avaliacao`, e foram removidos: a versão Gradio (`src/web/app.py` e `Dockerfile`), o antigo `comparar_modelos.py` (Random Forest e SVM sobre as coordenadas cruas da tela, que gerou a Tabela 1 do TCC), o `src/classificadores/landmarks.py` e o documento `docs/ambientes/local-vs-huggingface.md`. Não procure nem cite esses arquivos como se ainda existissem.

O `landmarks.py` fazia a normalização no pulso: colocava o pulso na origem e dividia os pontos pela distância até o ponto mais afastado dele. Essa normalização foi usada pelas primeiras versões de Random Forest, SVM e SNN, no app e no treino, entre 4 e 5 de outubro de 2026. Depois ela foi trocada pelo resumo de 128 números e, na limpeza, nenhum código a usava mais. Ela nunca foi usada no `comparar_modelos.py`, o script que gerou a Tabela 1 do TCC.

---

## 4. Comandos de Execução

### 4.1. Execução Local (Recomendado para Demonstração e Testes)
```powershell
python src/desktop/detectar_libras.py
python src/desktop/detectar_libras.py --modelo snn   # ou task, rf, svm
```
* `q` (minúsculo) encerra, `H` abre a ajuda com todos os atalhos, `M` troca o classificador e `+` / `-` mudam o limiar de confiança.

### 4.2. Execução da Versão Web no Computador
Na raiz do repositório:
```powershell
python -m http.server
```
* Abra no navegador: `http://localhost:8000/web/`. A câmera só abre em HTTPS ou em `localhost`.

### 4.3. Treino e Métricas para a Monografia
```powershell
# Treina RF, SVM e SNN sobre o resumo de 128 números, salva em models/
# e imprime a tabela LaTeX dos quatro modelos (inclui a linha do .task):
python src/treino/treinar_modelos.py

# Gera os gráficos de comparação dos quatro modelos e as matrizes de RF, SVM e SNN:
python src/avaliacao/graficos_modelos.py

# Matriz de confusão do .task em todas as fotos de teste (Figura 3):
python src/avaliacao/matriz_confusao_task.py

# Desenha as curvas de acurácia e perda do treino do .task no Colab (Figuras 1 e 2):
python src/avaliacao/curvas_treino_task.py
```

### 4.4. Gráficos em `results/figures/` e o script que gera cada um
| Gráfico | Script |
| :--- | :--- |
| `task_curva_acuracia.png`, `task_curva_perda.png` | `src/avaliacao/curvas_treino_task.py` (números copiados do log do treino no Colab) |
| `matriz_confusao_task.png` | `src/avaliacao/matriz_confusao_task.py` |
| `matriz_confusao_rf.png`, `matriz_confusao_svm.png`, `matriz_confusao_snn.png`, `snn_curva_perda.png`, `snn_curva_acuracia.png`, `comparacao_modelos.png`, `f1_por_letra.png`, `distribuicao_amostras.png`, `deteccao_maos_por_letra.png` | `src/avaliacao/graficos_modelos.py` |

---

## 5. Cuidados com as Métricas

As métricas do projeto foram medidas nas fotos de `data/libras/test`, que já veio separada do treino e tem as mesmas mãos do treino (pelo menos 4 a 5 mãos diferentes, contadas pela cor da pele e pelos acessórios, e não uma pessoa só). Por isso elas mostram o acerto com mãos já vistas, e não com um usuário novo. Esse problema se chama vazamento: o teste repete as mãos do treino. Ao falar dos resultados, leve em conta também:

* **Teste por grupo de gravação.** A auditoria de outubro de 2026 separou as fotos em 6 grupos de gravações parecidas e treinou o SVM deixando um grupo de fora de cada vez. Em 2 dos 6 grupos, o acerto do SVM no grupo deixado de fora caiu para 77% e 88%. Nos outros 4 grupos, ele ficou entre 93% e 99%. Não diga que qualquer grupo deixado de fora derruba o acerto.
* **Uso ao vivo.** Com uma mão nova na frente da webcam, os modelos ainda confundem letras parecidas; o caso relatado foi U com R. O resumo de 128 números não resolveu o vazamento.
* **O 0,891 do `.task`.** Esse acerto não é um resultado "mais realista": o `.task` acerta cerca de 0,89 até nas fotos de treino, porque foi pouco treinado. O log do Colab confirma só as 10 épocas (passadas completas pelas fotos de treino). Os lotes de 2 fotos são o padrão do Model Maker e, por isso, a configuração provável, mas não confirmada.
* **Texto do TCC de 18/09.** Ele cita normalização no pulso, divisão 70/15/15 e a base V-LIBRASIL. Nenhum código do repositório fez a divisão 70/15/15 nem usou a V-LIBRASIL. A normalização no pulso só existiu nas primeiras versões de RF, SVM e SNN (seção 3) e nunca foi usada no script que gerou a Tabela 1, que usava as coordenadas cruas da tela.

Os detalhes e os próximos passos estão em [docs/avaliacao/metodologia-e-limitacoes.md](docs/avaliacao/metodologia-e-limitacoes.md).

---

## 6. Convenções do Projeto

* **Commits:** Mensagens devem seguir o padrão Conventional Commits (ex: `feat:`, `fix:`, `docs:`, `perf:`). Commits nunca são executados de forma autônoma pelo assistente.
* **Documentação:** Qualquer nova documentação técnica deve ser criada sob a pasta `docs/` e indexada no `docs/README.md`.
