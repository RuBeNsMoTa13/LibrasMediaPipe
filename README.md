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
> Reconhecimento em tempo real das letras do alfabeto manual da Língua Brasileira de Sinais (LIBRAS) pela webcam, usando o Google MediaPipe para encontrar a mão e quatro classificadores para dizer qual letra ela está fazendo.

---

## 📌 Sumário
* [Visão Geral](#-visão-geral)
* [Ambientes de Execução](#-ambientes-de-execução)
* [Pipeline de Inteligência Artificial](#-pipeline-de-inteligência-artificial)
* [Estrutura do Repositório](#-estrutura-do-repositório)
* [Instalação e Pré-requisitos](#-instalação-e-pré-requisitos)
* [Como Executar](#-como-executar)
* [Resultados e Benchmarks](#-resultados-e-benchmarks)
* [Documentação Técnica Completa](#-documentação-técnica-completa)
* [Autoria e Informações Acadêmicas](#-autoria-e-informações-acadêmicas)

---

## 📖 Visão Geral

O **LibrasMediaPipe** é um projeto de acessibilidade que reconhece, pela webcam, as letras estáticas do alfabeto manual de LIBRAS. São 21 letras: `A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y`. Ficam de fora H, J, K, X e Z, que em LIBRAS são feitas com movimento e não cabem numa foto parada.

A ideia central é não analisar a foto inteira, pixel por pixel. Primeiro o **Google MediaPipe** encontra a mão na imagem e marca 21 pontos nela (as pontas e as juntas dos dedos e o pulso). Esses pontos são chamados de *landmarks*. Depois, uma rede pequena transforma esses pontos em um "resumo" de 128 números que descreve o formato da mão, e um classificador decide qual letra é. Trabalhar com pontos e resumos, em vez de fotos inteiras, deixa o reconhecimento leve o bastante para rodar em tempo real num computador comum, sem placa de vídeo dedicada.

As letras reconhecidas vão sendo juntadas numa palavra (soletração) e o programa pode falar essa palavra em voz alta.

---

## 🔄 Ambientes de Execução

O projeto tem duas versões do aplicativo, com os mesmos recursos de soletração e voz:

```
                      +------------------------------------------+
                      |            LibrasMediaPipe               |
                      +------------------------------------------+
                                    /              \
                                   /                \
          [Desktop, no computador] /                  \ [Web, no navegador]
       +---------------------------+                  +---------------------------+
       | src/desktop/detectar_...  |                  |     web/index.html        |
       |  - Python + OpenCV        |                  |  - MediaPipe JavaScript   |
       |  - 4 classificadores      |                  |  - Só o .task             |
       |    (.task, SNN, RF, SVM)  |                  |  - Roda no navegador      |
       |  - Voz com pyttsx3        |                  |  - Voz com Web Speech API |
       |  - Precisa instalar       |                  |  - Sem instalar nada      |
       +---------------------------+                  +---------------------------+
```

1. **Aplicativo desktop (`src/desktop/detectar_libras.py`):**
   * Abre a webcam do computador com o OpenCV e mostra a letra reconhecida sobre o vídeo.
   * **Troca de classificador ao vivo:** a tecla `M` alterna entre o MediaPipe `.task`, a SNN, o Random Forest e o SVM, e a barra superior da janela mostra qual classificador está em uso. Também é possível escolher na hora de abrir, com `--modelo task|snn|rf|svm`.
   * **Soletração:** as letras reconhecidas são juntadas numa palavra. A mesma letra não é repetida enquanto a mão continua fazendo o mesmo sinal; para repetir uma letra, tire a mão da câmera e mostre a letra de novo. Entre uma letra registrada e a seguinte, é preciso passar pelo menos 0,7 s. Há teclas para inserir espaço e apagar.
   * **Voz:** a palavra soletrada é falada pelo `pyttsx3`, que usa a voz instalada no sistema operacional. No modo manual a fala acontece ao teclar `Enter`; no modo automático (tecla `V`), depois de 5 s sem letras novas.
   * **Limiar de confiança:** o programa só aceita uma letra quando o classificador tem pelo menos 50% de certeza. Esse valor sobe e desce de 5 em 5% com as teclas `+` e `-`.

2. **Aplicativo web no navegador (`web/index.html`):**
   * Roda o mesmo `gesture_recognizer.task` direto no navegador, com a biblioteca `@mediapipe/tasks-vision` (WebAssembly e WebGL). Não existe servidor que faça o reconhecimento: o vídeo nunca sai do computador do usuário.
   * Tem a mesma soletração (0,7 s entre letras), o mesmo limiar ajustável (padrão 50%) e voz em português pela Web Speech API do navegador. Usa apenas o classificador `.task`.
   * Está publicado como Space estático no Hugging Face: <https://rubensmota13-librasmediapipe.static.hf.space>.

> A versão web anterior, feita em Gradio (publicada primeiro com o SDK Gradio do Hugging Face e, a partir de 26/05/2026, num contêiner Docker), enviava cada quadro do vídeo para um servidor e por isso tinha cerca de 1,5 s de atraso; ela foi removida e substituída pela versão no navegador, explicada em [docs/ambientes/versao-web-navegador.md](docs/ambientes/versao-web-navegador.md).

---

## 🧠 Pipeline de Inteligência Artificial

O caminho de cada quadro da webcam, do vídeo até a voz, é este:

```
  Quadro da webcam (espelhado, como num espelho)
        |
        v
  MediaPipe encontra a mão
  (21 pontos na imagem + 21 pontos 3D em metros + mão direita ou esquerda)
        |
        v
  Rede gesture_embedder (vem dentro do .task)
  transforma os pontos num resumo de 128 números
        |
        +--> (a) cabeça do próprio .task, treinada no Colab ----+
        |                                                       |
        +--> (b) Random Forest, SVM ou SNN, treinados aqui -----+
                                                                |
                                                                v
                                          Letra + grau de certeza (confiança)
                                                                |
                                                                v
                              Limiar de confiança -> soletração -> voz
```

No desenho, a **cabeça** do `.task` é a parte final desse modelo: ela recebe o resumo de 128 números e escolhe a letra. Os outros três classificadores fazem o mesmo papel, mas foram treinados neste repositório.

### 1. Captura e detecção da mão
O quadro da webcam é espelhado na horizontal (`cv2.flip`), para a imagem se comportar como um espelho, e convertido de BGR para RGB, o formato de cores que o MediaPipe espera. O MediaPipe então devolve três informações sobre a mão:
* **21 pontos na imagem:** a posição de cada ponto na tela.
* **21 pontos 3D em metros:** a mesma mão, medida em metros a partir do centro dela, o que não depende de onde ela está na tela.
* **Lado da mão:** se é a mão direita ou a esquerda.

### 2. O resumo de 128 números
Dentro do arquivo `models/gesture_recognizer.task` existe uma rede do Google chamada `gesture_embedder`. Ela recebe as três informações acima e devolve uma lista de 128 números que descreve o formato da mão. Dá para pensar nesse resumo como uma "impressão digital" do gesto: a mesma letra feita por mãos diferentes tende a gerar resumos parecidos. A própria rede cuida da posição e do tamanho da mão, por isso o projeto, hoje, não faz nenhuma normalização manual dos pontos (como centralizar no pulso).

O arquivo [`src/classificadores/resumo.py`](src/classificadores/resumo.py) tira essa rede de dentro do `.task` e a roda sozinha. Assim, os quatro classificadores partem exatamente da mesma informação. Foi conferido que, passando esse resumo pela cabeça que vem no `.task`, o resultado é idêntico ao do próprio MediaPipe.

Entre 4 e 5 de outubro de 2026, as versões de Random Forest, SVM e SNN usadas no app recebiam os pontos da mão centralizados no pulso, tanto no treino quanto no app. Depois essa normalização foi trocada pelo resumo de 128 números. (O Random Forest e o SVM da Tabela 1 do TCC, de junho, eram outros: usavam as coordenadas cruas da tela, sem normalização.) A troca para o resumo não resolveu o principal limite da avaliação: ao vivo, com uma mão que não está no dataset, os modelos ainda confundem letras parecidas, como U e R (veja [Resultados e Benchmarks](#-resultados-e-benchmarks)).

### 3. Os quatro classificadores
Todos recebem o resumo de 128 números e devolvem uma das 21 letras com um grau de certeza.

| Classificador | O que é | Configuração usada |
| :--- | :--- | :--- |
| **MediaPipe `.task`** | Classificador do MediaPipe, treinado no Google Colab com o MediaPipe Model Maker. | Cabeça pequena, sem camadas escondidas: uma etapa que padroniza os 128 números (BatchNorm), uma que zera os valores negativos (ReLU) e uma camada densa que dá uma nota a cada uma das 22 saídas (as 21 letras e "none", que significa "nenhum gesto"). Treinado no Google Colab com [`notebooks/treinar_no_colab.ipynb`](notebooks/treinar_no_colab.ipynb), com o dataset inteiro do Kaggle (46.262 fotos), usando a pasta `train` para treinar e a `test` só para testar. As configurações e a perda e a acurácia de cada época ficam em [`results/treinos/task.json`](results/treinos/task.json). O notebook do treino original (10 épocas, lotes de 2, `train` e `test` misturados) está em [`notebooks/historico/`](notebooks/historico/Libras_gesture_recognizer.ipynb). |
| **Random Forest (RF)** | Conjunto de árvores de decisão que votam na letra. | 100 árvores; o resto é o padrão do scikit-learn. |
| **Support Vector Machine (SVM)** | Separa as letras traçando fronteiras entre os grupos de resumos. | Padronização (`StandardScaler`) + kernel RBF, `C=1`, `gamma='scale'`, `probability=True`. |
| **Spiking Neural Network (SNN)** | Rede neural inspirada no cérebro, cujos neurônios trocam "pulsos" (*spikes*) ao longo do tempo. | snnTorch. Entrada com os 128 números do resumo, duas camadas escondidas de 128 neurônios LIF e uma camada de saída com 21 neurônios LIF (um por letra), com decaimento 0,9; 25 passos de tempo; otimizador Adam (taxa 0,002); 50 épocas; lotes de 64; perda `mse_count_loss`, que pede que o neurônio da letra certa dispare em 80% dos passos e os outros em 10%. |

RF, SVM e SNN são treinados por [`src/treino/treinar_modelos.py`](src/treino/treinar_modelos.py) com cada mão do treino duplicada em versão espelhada, para reconhecer tanto a mão direita quanto a esquerda. O script anota o treino em [`results/treinos/classificadores.json`](results/treinos/classificadores.json): data, configurações, perda da SNN a cada época, métricas e versões das bibliotecas.

### 4. Depois da classificação
* **Limiar de confiança:** letras com certeza abaixo do limiar (padrão 50%) são ignoradas.
* **Soletração:** a mesma letra não entra duas vezes seguidas enquanto a mão continua no mesmo sinal; para repetir, é preciso tirar a mão da câmera. Entre uma letra registrada e a seguinte, é preciso passar pelo menos 0,7 s.
* **Voz:** a palavra é falada ao teclar `Enter` ou, no modo automático, depois de 5 s sem letras novas.

---

## 📁 Estrutura do Repositório

```
LibrasMediaPipe/
├── README.md                           # Este arquivo (o cabeçalho no topo configura o Space)
├── GEMINI.md                           # Diretrizes do projeto para o assistente Gemini
├── requirements.txt                    # Todas as dependências Python do projeto
├── .gitattributes                      # Coloca o .task no Git LFS
├── .gitignore                          # Arquivos que o git ignora (caches, ambientes)
├── .agents/rules/workspace-rules.md    # Regras do workspace para assistentes de IA
│
├── data/libras/                        # Dataset de fotos 64x64, uma pasta por letra
│   ├── train/A..Y/                     # 3.468 fotos de treino (amostra de ~10% do Kaggle)
│   └── test/A..Y/                      # 1.153 fotos de teste (o treino no Colab usa as 46.262 do Kaggle)
│
├── models/                             # Modelos prontos para uso
│   ├── gesture_recognizer.task         # MediaPipe .task (treinado no Colab, fica no Git LFS)
│   ├── rf_libras.pkl                   # Random Forest (gerado por treinar_modelos.py)
│   ├── svm_libras.pkl                  # SVM (gerado por treinar_modelos.py)
│   └── snn_libras.pt                   # SNN (gerado por treinar_modelos.py)
│
├── notebooks/
│   ├── treinar_no_colab.ipynb          # Treina os 4 modelos no Google Colab e anota tudo
│   └── historico/                      # Notebook do treino original do .task
│
├── results/
│   ├── figures/                        # Gráficos (lista em "Resultados e Benchmarks")
│   ├── treinos/                        # Registro de cada treino (task.json, classificadores.json)
│   └── tables/                         # Cache resumo_libras.npz (criado ao rodar, ignorado pelo git)
│
├── web/
│   └── index.html                      # App no navegador (MediaPipe JS, soletração e voz)
│
├── src/
│   ├── classificadores/                # Peças usadas pelo app e pelos scripts
│   │   ├── resumo.py                   # Tira a rede do .task, gera o resumo de 128 números e o cache
│   │   ├── classicos.py                # Carrega Random Forest e SVM salvos
│   │   └── snn.py                      # Arquitetura da SNN, salvar e carregar
│   ├── desktop/
│   │   └── detectar_libras.py          # App da webcam (OpenCV, soletração, voz, tecla M)
│   ├── treino/
│   │   └── treinar_modelos.py          # Treina RF, SVM e SNN e salva em models/
│   └── avaliacao/
│       ├── graficos_modelos.py         # Métricas e gráficos de comparação dos 4 modelos
│       ├── matriz_confusao_task.py     # Matriz de confusão do .task (Figura 3 do TCC)
│       └── curvas_treino_task.py       # Curvas do treino do .task, lidas de results/treinos/task.json
│
└── docs/                               # Documentação técnica do TCC
    ├── README.md                       # Índice da documentação
    ├── todo.md                         # Lista de tarefas (backlog)
    ├── ambientes/
    │   └── versao-web-navegador.md     # Como a versão web foi refeita no navegador
    └── avaliacao/
        └── metodologia-e-limitacoes.md # Como os modelos foram avaliados e o que os números significam
```

---

## 💻 Instalação e Pré-requisitos

### Pré-requisitos:
* Python 3.10, 3.11 ou 3.12 instalado.
* [Git LFS](https://git-lfs.com) instalado. O modelo `models/gesture_recognizer.task` fica guardado no Git LFS; sem ele, o git baixa só um "ponteiro" de poucos bytes no lugar do modelo e nada funciona.
* Webcam conectada (para os aplicativos em tempo real).

### Passo a Passo:

1. **Clone o repositório e baixe o modelo do Git LFS:**
   ```bash
   git lfs install
   git clone https://github.com/RuBeNsMoTa13/LibrasMediaPipe.git
   cd LibrasMediaPipe
   git lfs pull
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
   O `requirements.txt` tem tudo o que o projeto usa: o app desktop (`mediapipe`, `opencv-python`, `numpy`, `pillow`, `pyttsx3`), o resumo de 128 números (`ai-edge-litert`), os classificadores (`scikit-learn`, `torch`, `snntorch`) e os gráficos (`matplotlib`, `seaborn`). O `torch` é um download grande.

---

## 🚀 Como Executar

Todos os comandos são rodados a partir da pasta raiz do repositório.

### 1. Aplicativo desktop (recomendado para apresentações)
```powershell
python src/desktop/detectar_libras.py
```
Abre a webcam, mostra a letra reconhecida e a palavra soletrada, e fala a palavra. Começa com o MediaPipe `.task`. Para já abrir com outro classificador:
```powershell
python src/desktop/detectar_libras.py --modelo snn
```
As opções são `task`, `snn`, `rf` e `svm`. SNN, Random Forest e SVM usam os arquivos de `models/`; se algum deles não existir, o app avisa e usa o `.task`.

| Tecla | O que faz |
| :--- | :--- |
| `M` | Troca o classificador (.task, SNN, RF, SVM) |
| `Enter` | Fala a palavra acumulada e limpa o texto |
| `V` | Alterna a voz entre manual (`Enter`) e automática (após 5 s sem letras) |
| `+` / `=` e `-` / `_` | Aumenta ou diminui o limiar de confiança em 5% |
| `Espaço` | Insere um espaço entre palavras |
| `D` / `Backspace` | Apaga o último caractere |
| `C` | Limpa o texto acumulado |
| `H` | Abre e fecha o manual de atalhos |
| `q` (minúsculo) | Fecha o aplicativo (com Caps Lock ligado ou com Shift, a tecla não funciona) |

### 2. Aplicativo web no navegador
Online, sem instalar nada: <https://rubensmota13-librasmediapipe.static.hf.space>

Para rodar no próprio computador, sirva a pasta raiz do repositório (o navegador só libera a câmera em HTTPS ou em `localhost`):
```powershell
python -m http.server
```
Depois abra [http://localhost:8000/web/](http://localhost:8000/web/). A página carrega o `models/gesture_recognizer.task` local, por isso o `git lfs pull` precisa ter sido feito. Os atalhos de teclado são os mesmos do desktop, exceto o `M` (no navegador só existe o `.task`) e o `q`, que no navegador só desliga a câmera.

### 3. Treino dos classificadores
```powershell
python src/treino/treinar_modelos.py
```
* Na primeira vez, passa todas as fotos de `data/libras/train` e `data/libras/test` pelo `.task`, guarda os pontos de cada mão em `results/tables/resumo_libras.npz` e reaproveita esse arquivo nas vezes seguintes (apague-o para extrair de novo). Fotos em que o MediaPipe não acha a mão ficam de fora.
* Treina a SNN, o Random Forest e o SVM sobre o resumo de 128 números.
* Imprime acurácia, precisão, recall e F1 dos quatro modelos numa tabela separada por tabulação, pronta para colar no Word. A linha do `.task` é medida nas mesmas fotos com mão que os outros três.
* Salva (e substitui) `models/snn_libras.pt`, `models/rf_libras.pkl` e `models/svm_libras.pkl`, usados pelo app desktop.

### 4. Avaliação e gráficos

* **Comparação dos quatro modelos:**
  ```powershell
  python src/avaliacao/graficos_modelos.py
  ```
  Treina de novo RF, SVM e SNN com a mesma receita do treino (sem mexer nos arquivos de `models/`), imprime a tabela de métricas e salva nove gráficos em `results/figures/`: as matrizes de confusão de RF, SVM e SNN, as duas curvas da SNN, a comparação de métricas, o F1 por letra, a distribuição das fotos e a detecção de mãos por letra.

* **Matriz de confusão do `.task` (Figura 3 do TCC):**
  ```powershell
  python src/avaliacao/matriz_confusao_task.py
  ```
  Passa as 1.153 fotos de teste pelo `.task`. Quando o MediaPipe não acha a mão, a resposta conta como erro, com o nome "Nenhum". Imprime o relatório de classificação por letra e salva `results/figures/matriz_confusao_task.png`.

* **Curvas do treino do `.task` (Figuras 1 e 2 do TCC):**
  ```powershell
  python src/avaliacao/curvas_treino_task.py
  ```
  Não treina nada: desenha a acurácia e a perda das 10 épocas do treino feito no Colab, a partir de números copiados do log desse treino. Salva `results/figures/task_curva_acuracia.png` e `results/figures/task_curva_perda.png`.

---

## 📊 Resultados e Benchmarks

Os quatro modelos são avaliados nas mesmas fotos: as **1.135** fotos de `data/libras/test` (de um total de 1.153) em que o MediaPipe encontrou a mão. Os valores são impressos por `treinar_modelos.py` e por `graficos_modelos.py`. Precisão, recall e F1 são calculados letra por letra e depois resumidos numa média ponderada pelo número de fotos de cada letra. Por isso o F1 geral não é a média direta da precisão e do recall gerais e pode ficar abaixo dos dois, como acontece com o `.task` (F1 de 0,863, com precisão de 0,931 e recall de 0,891).

| Modelo | Acurácia | Precisão Ponderada | Recall Ponderado | F1 Ponderado |
| :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Tasks (`gesture_recognizer.task`)** | 0,891 | 0,931 | 0,891 | 0,863 |
| **Random Forest (100 árvores)** | 0,994 | 0,994 | 0,994 | 0,994 |
| **Support Vector Machine (SVM - RBF)** | 0,982 | 0,986 | 0,982 | 0,983 |
| **Spiking Neural Network (SNN - LIF)** | 0,990 | 0,991 | 0,990 | 0,990 |

Contando também as 18 fotos sem mão como erro (todas as 1.153 fotos, em `matriz_confusao_task.py`), o `.task` fica com acurácia de 0,877 e F1 de 0,856.

> **Atenção ao ler esses números:** eles medem o desempenho com as **mesmas mãos** que aparecem no treino, não com um usuário novo.
> * O dataset `data/libras` já chegou dividido em treino e teste; nenhum script do projeto faz essa divisão, e não existe a informação de qual pessoa ou vídeo gerou cada foto.
> * Há pelo menos 4 a 5 mãos diferentes nas fotos, e as mesmas mãos aparecem no treino e no teste. As fotos de teste são um pouco mais diferentes das de treino do que seriam se tivessem sido sorteadas do mesmo conjunto, mas um método que só procura a mão de treino com os pontos (landmarks) mais parecidos (vizinho mais próximo, 1-NN) ainda acerta 99,2% do teste.
> * Para estimar o acerto com uma pessoa nova, as fotos com mão foram separadas em 6 grupos de gravações parecidas (mesma cor de pele, fundo e iluminação), e o SVM foi treinado sem cada grupo e testado nele. Em 2 dos 6 grupos o acerto caiu para 77% e 88%, com confusões como N e M, P e N, C e O. Nos outros 4 grupos ele ficou entre 93% e 99%. Ou seja, uma mão ou um ambiente bem diferente do treino pode derrubar bastante o acerto, mas isso não acontece com qualquer grupo.
> * Ao vivo, com uma mão que não está no dataset, os modelos ainda confundem letras parecidas; o caso relatado é U e R. No teste, essa confusão quase não aparece para RF, SVM e SNN (um único caso, no SVM), o que é mais um sinal de que o teste é fácil demais para eles. Trocar os pontos da mão pelo resumo de 128 números não resolveu esse problema.
> * O 0,891 do `.task` **não** quer dizer que ele seja "mais realista". Ele acerta cerca de 0,89 até nas fotos de treino, ou seja, está subtreinado: não aprendeu bem nem os exemplos que viu. A causa provável é a configuração padrão do Model Maker (lotes de 2 fotos e 10 épocas); o log do Colab confirma só as 10 épocas, e o lote de 2 é uma dedução. A mesma cabeça, treinada com lotes maiores e mais épocas, chega a 0,99.
>
> A explicação completa, os próximos testes recomendados (como um teste com uma mão que nunca foi usada no treino) e as diferenças entre o texto do TCC e o que o código faz estão em [docs/avaliacao/metodologia-e-limitacoes.md](docs/avaliacao/metodologia-e-limitacoes.md).

### Gráficos gerados

Todos ficam em [`results/figures/`](results/figures).

| Gráfico | O que mostra | Script |
| :--- | :--- | :--- |
| `task_curva_acuracia.png` / `task_curva_perda.png` | Acurácia e perda por época do treino do `.task` no Colab (Figuras 1 e 2 do TCC) | `curvas_treino_task.py` |
| `matriz_confusao_task.png` | Matriz de confusão do `.task` em todas as fotos de teste, com "Nenhum" para foto sem mão (Figura 3 do TCC) | `matriz_confusao_task.py` |
| `matriz_confusao_rf.png` | Matriz de confusão do Random Forest | `graficos_modelos.py` |
| `matriz_confusao_svm.png` | Matriz de confusão do SVM | `graficos_modelos.py` |
| `matriz_confusao_snn.png` | Matriz de confusão da SNN | `graficos_modelos.py` |
| `snn_curva_perda.png` | Perda da SNN a cada época | `graficos_modelos.py` |
| `snn_curva_acuracia.png` | Acurácia da SNN no treino e no teste a cada época | `graficos_modelos.py` |
| `comparacao_modelos.png` | Acurácia, precisão, recall e F1 dos quatro modelos lado a lado | `graficos_modelos.py` |
| `f1_por_letra.png` | F1 de cada letra em cada modelo (mapa de calor) | `graficos_modelos.py` |
| `distribuicao_amostras.png` | Quantidade de fotos por letra no treino e no teste | `graficos_modelos.py` |
| `deteccao_maos_por_letra.png` | Porcentagem das fotos de teste em que o MediaPipe encontrou a mão | `graficos_modelos.py` |
| `tsne_letras.png` | Resumos de 128 números de todas as fotos com mão projetados em 2D pelo t-SNE; à direita, só as letras parecidas (M, N, T, F, U, R, V) | `grafico_tsne.py` |

---

## 📚 Documentação Técnica Completa

Para entender a metodologia, as decisões de arquitetura e as tarefas pendentes do TCC, consulte a pasta `docs/`:

* 📖 **[Índice da documentação técnica](docs/README.md)**
* 🔬 **[Metodologia e limitações da avaliação](docs/avaliacao/metodologia-e-limitacoes.md)**
* 🌐 **[Versão web no navegador: por que a versão antiga lagava e por que a nova funciona](docs/ambientes/versao-web-navegador.md)**
* 📋 **[Lista de tarefas do TCC (backlog)](docs/todo.md)**
* 🏛️ **[Diretrizes do projeto (GEMINI.md)](GEMINI.md)**

---

## 👨‍💻 Autoria e Informações Acadêmicas

* **Curso:** Bacharelado em Ciência de Dados e Inteligência Artificial
* **Tema:** Reconhecimento do Alfabeto Manual de LIBRAS em Tempo Real com MediaPipe e Aprendizado de Máquina
* **Autor:** Rubens Mota
