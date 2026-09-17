<!--
Data/Hora: 2026-09-17 14:58 (UTC-3)
Branch: main
Commit: 31a99a8
Status: Atualizado
-->

# Análise de Ambientes: Execução Local vs. Hugging Face Spaces

Este documento técnico formaliza a separação arquitetural do projeto **LibrasMediaPipe**, comparando a execução local nativa (*desktop*) com a execução em nuvem no *Hugging Face Spaces*. Além disso, detalha as causas fundamentais de latência (*lag*) observadas no ambiente em nuvem e propõe soluções práticas de engenharia de software para otimização.

---

## 1. Contexto e Motivação

O projeto foi concebido originalmente como uma aplicação **local desktop** de visão computacional e acessibilidade, projetada para capturar diretamente o fluxo de vídeo da webcam do usuário, processar os landmarks a alta taxa de quadros (30+ FPS) e emitir voz sintetizada através do hardware de áudio do computador.

Posteriormente, por recomendação da orientação acadêmica, o projeto foi expandido para o **Hugging Face Spaces** com o objetivo de facilitar a demonstração pública sem exigir que avaliadores ou usuários finais instalem Python, dependências do OpenCV e drivers de vídeo em suas próprias máquinas.

Entretanto, a transição de uma arquitetura *standalone* local para uma arquitetura cliente-servidor distribuída na nuvem introduz desafios significativos de latência, consumo de banda e limitações de hardware.

---

## 2. Comparativo Estrutural dos Ambientes

| Critério | Ambiente Local Desktop (`src/desktop/detectar_libras.py`) | Ambiente Cloud (`src/web/app.py` no Hugging Face Spaces) |
| :--- | :--- | :--- |
| **Ponto de Entrada** | Script Python nativo executado no terminal. | Container Docker com interface Web via Gradio. |

| **Captura de Vídeo** | Acesso direto ao dispositivo de captura (`cv2.VideoCapture(0)`). | API do navegador (`navigator.mediaDevices.getUserMedia`). |
| **Pipeline de Imagem** | Memória RAM local $\rightarrow$ Processamento $\rightarrow$ Tela (OpenCV GUI). | Navegador $\rightarrow$ Rede $\rightarrow$ Servidor Nuvem $\rightarrow$ Rede $\rightarrow$ Navegador. |
| **Latência por Frame** | Quase nula (15ms a 33ms por quadro $\approx$ 30 a 60 FPS). | Média a Alta (150ms a 600ms dependendo da conexão e fila do Gradio). |
| **Gargalo Principal** | Capacidade de processamento da CPU/GPU local. | Largura de banda, *Round-Trip Time* (RTT) de rede e cota do servidor. |
| **Áudio / TTS** | Síntese de voz direta no sistema operacional (`pyttsx3`). | Inoperante no servidor (ausência de placa de som); requer Web Speech API no navegador. |
| **Hardware Disponível** | Todo o hardware disponível na máquina do usuário. | Limitação de 2 vCPUs compartilhadas e 16 GB de RAM (Free Tier do Spaces). |
| **Facilidade de Acesso** | Exige ambiente Python, pip e clone do repositório. | Acesso instantâneo via link web em qualquer navegador. |

---

## 3. Por que o Projeto Apresenta "Lag" no Hugging Face?

O comportamento de engasgo ou lentidão (*lag*) no Hugging Face Spaces não decorre de falhas no algoritmo do MediaPipe, mas sim da **física e da arquitetura do streaming de imagens via web**:

### 3.1. O Ciclo do Round-Trip Time (RTT) de Rede
No ambiente local, cada frame é lido da memória de vídeo local a uma velocidade de barramento de gigabytes por segundo. No Hugging Face Spaces:
1. O navegador captura um frame da webcam (ex: 640x480 pixels).
2. O navegador codifica a imagem (em JPEG ou WebP) e a envia via HTTP POST ou WebSocket para o servidor do Hugging Face (geralmente hospedado nos Estados Unidos ou Europa).
3. O servidor recebe o pacote, descompacta a imagem para a memória e entrega para o script Python.
4. O MediaPipe processa a inferência.
5. O OpenCV desenha o esqueleto e anotações no frame.
6. O servidor **re-codifica** a imagem resultante e envia todo o fluxo de bytes de volta pela internet.
7. O navegador recebe o frame anotado, descompacta e renderiza na tela.

> Esse trajeto transatlântico de ida e volta (RTT) adiciona entre **150ms e 400ms de latência pura de rede** a cada frame transmitido, tornando impossível manter uma experiência fluida de 30 FPS apenas reenviando imagens estáticas repetidamente.

### 3.2. Custo de Compressão e Descompressão Contínua
Codificar e decodificar matrizes de imagem de 640x480 pixels 15 a 30 vezes por segundo consome ciclos massivos de CPU. Nos servidores gratuitos do Hugging Face (que compartilham 2 vCPUs básicas), esse overhead compete diretamente com o tempo de inferência do MediaPipe.

### 3.3. Peculiaridade do Áudio (TTS) em Ambientes Headless
Na máquina local, bibliotecas como `pyttsx3` comunicam-se diretamente com o driver de áudio do sistema operacional (Windows SAPI5 ou Linux ALSA/PulseAudio). 

Em um container Docker na nuvem:
* Não existe placa de som física nem caixas de som conectadas ao container.
* Chamar `pyttsx3` no servidor tentaria tocar o som nos servidores da nuvem, e não no fone de ouvido do usuário conectado no navegador.
* **Solução:** O TTS na nuvem precisa ser disparado no lado do cliente (*client-side*) utilizando a **Web Speech API** nativa do próprio navegador web via JavaScript.

---

## 4. Otimizações Já Implementadas no `src/web/app.py`

Para atenuar esses gargalos no ambiente web, as seguintes técnicas de engenharia foram inseridas em [`src/web/app.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/src/web/app.py):


1. **Downscaling Agressivo para Inferência (`DOWNSCALE_WIDTH = 192`):**
   * A imagem de entrada é reduzida para apenas 192 pixels de largura antes de ser enviada ao `GestureRecognizer`. Como os landmarks da mão são geométricos, essa redução diminui o tempo de cálculo da rede neural sem perder a precisão dos pontos-chave.
2. **Frame Throttling / Pulo de Quadros (`PROCESS_EVERY_N = 3`):**
   * O MediaPipe processa apenas 1 a cada 3 frames recebidos. Nos outros 2 frames, a aplicação utiliza os resultados armazenados em *cache* (`cached_label` e `cached_landmarks`), reduzindo a carga de processamento na CPU em 66%.
3. **Cache de Landmarks:**
   * Evita redesenhar ou recalcular poses em momentos em que o usuário mantém a mão estável.

---

## 5. Roteiro de Melhorias Propostas para o Hugging Face

Para eliminar o *lag* restante e tornar a aplicação no Hugging Face tão responsiva quanto a local, recomenda-se adotar as seguintes estratégias:

### Proposta A: Retornar Apenas Metadados (Eliminar o Reenvio de Imagens)
* **Problema Atual:** O servidor recebe o vídeo do cliente, desenha nele e envia de volta um novo vídeo. Isso dobra o consumo de banda.
* **Solução:** O servidor do Gradio recebe o frame, processa e devolve **apenas um JSON** contendo:
  ```json
  {"letra": "A", "confianca": 0.89, "landmarks": [[x1, y1], [x2, y2], ...]}
  ```
* O navegador do cliente desenha os pontos sobre o seu próprio vídeo usando um elemento `<canvas>` em JavaScript. Isso elimina 50% do tráfego de rede e reduz a latência à metade.

### Proposta B: Inferência 100% Client-Side com MediaPipe Web (Solução Definitiva)
* O MediaPipe disponibiliza a biblioteca oficial em JavaScript (`@mediapipe/tasks-vision`) que pode ser executada diretamente via WebAssembly / WebGL no navegador do usuário.
* **Vantagem:** O vídeo da webcam **nunca sai da máquina do usuário**. O modelo roda diretamente na GPU do navegador do cliente a 60 FPS, sem gastar processamento do Hugging Face e sem qualquer atraso de rede.
* O Hugging Face Spaces funciona nesse caso apenas como hospedeiro estático dos arquivos da aplicação.

### Proposta C: Adoção de WebRTC Real (FastRTC / Gradio WebRTC)
* Em vez de usar polling de imagens HTTP/WebSocket (`streaming=True` tradicional do Gradio), utilizar o suporte nativo a **WebRTC**, que opera sobre protocolo UDP, negociando dinamicamente a taxa de quadros e eliminando filas de buffers acumuladas.

### Proposta D: Síntese de Voz (TTS) via Web Speech API
* Injetar um trecho curto de JavaScript na interface Gradio que escute a mudança do texto reconhecido e acione `window.speechSynthesis.speak(new SpeechSynthesisUtterance(texto))`, permitindo que o usuário escute a soletração diretamente em seus alto-falantes locais.

---

## 6. Como Apresentar essa Dualidade na Monografia do TCC

A separação entre a versão local e a versão em nuvem não deve ser vista como um "problema", mas sim como um **dos pontos mais ricos do trabalho acadêmico**:

1. **Discussão sobre Edge Computing vs. Cloud Computing:**
   * Demonstrar com dados empíricos (FPS e latência média) como a computação na borda (*Edge Computing*, o script local) é superior para tarefas de visão computacional interativa e acessibilidade em tempo real.
2. **Trade-offs de Arquitetura de Software:**
   * Contrastar a facilidade de distribuição (*acesso universal via URL sem instalação*) com o custo de latência de rede (*RTT* e compressão de streaming).
3. **Maturidade de Engenharia:**
   * Mostrar à banca que você compreende a diferença entre treinar um modelo de Machine Learning e colocá-lo em produção em ambientes distribuídos heterogêneos.
