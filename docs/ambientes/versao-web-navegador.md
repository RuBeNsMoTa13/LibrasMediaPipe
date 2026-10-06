<!--
Data/Hora: 2026-10-06 19:30 (UTC-3)
Branch: claude/auditoria-pipeline-3l8ukb
Commit: c209803
Status: Atualizado
-->

# Versão Web no Navegador: por que lagava e por que agora funciona

A versão web do reconhecimento do alfabeto manual de Libras passou a rodar inteiramente no navegador de quem acessa, e por isso deixou de travar. Antes, cada imagem da webcam era enviada ao servidor do Hugging Face, processada pelo MediaPipe em Python e devolvida pela rede. Agora o mesmo modelo treinado (`models/gesture_recognizer.task`) é executado localmente com a biblioteca JavaScript do MediaPipe, e o vídeo nunca sai do computador do usuário.

O modelo não mudou entre as duas versões. O que mudou foi **onde a inferência acontece**: no servidor (arquitetura cliente-servidor) ou no próprio dispositivo (inferência no cliente, *client-side*). Essa mudança põe em prática a proposta de inferência no cliente levantada na análise de latência da versão antiga, cujo resumo está na seção 1.

* **Aplicação publicada:** <https://rubensmota13-librasmediapipe.static.hf.space>
* **Código:** [`web/index.html`](../../web/index.html)

---

## 1. Versão antiga: Gradio no servidor

A primeira versão web era um Space Docker com uma interface Gradio, escrita em `src/web/app.py`. Esse arquivo foi removido do repositório e continua disponível no histórico do git. O navegador capturava a webcam e enviava os frames ao servidor, onde o MediaPipe em Python reconhecia o gesto, o OpenCV desenhava os pontos da mão e a imagem anotada voltava para a tela. O reconhecimento chegava a ficar cerca de 1,5 s atrasado em relação ao gesto, por quatro motivos que se somam:

1. **Poucos frames enviados.** O evento `input_video.stream(...)` usa o intervalo padrão do Gradio (`stream_every` de 0,5 s), ou seja, no máximo 2 frames por segundo chegam ao servidor.
2. **Descarte de frames.** O código usa `PROCESS_EVERY_N = 3`, uma otimização pensada para a câmera local a 30 FPS. Com apenas 2 FPS de entrada, o modelo passa a rodar uma vez a cada 1,5 s.
3. **Ida e volta pela rede.** Cada frame é codificado no navegador, enviado ao Hugging Face, decodificado, anotado, recodificado e devolvido. A latência da internet entra duas vezes em cada frame.
4. **Processamento dobrado e compartilhado.** Com `WEBRTC_COLOR_FALLBACK = True`, o reconhecedor roda duas vezes por frame (ordem de cores RGB e BGR). Isso acontece nas 2 vCPUs compartilhadas do plano gratuito, com uma fila única para todos os usuários, então os frames se acumulam quando há mais de um acesso.

O arquivo do modelo não era a causa. Se o `.task` tivesse chegado corrompido ao Space, o reconhecedor nem seria inicializado e o app nunca mostraria uma letra; ele reconhecia, só que com atraso.

---

## 2. Versão nova: MediaPipe no navegador

A nova versão é um único arquivo, `web/index.html`, em HTML e JavaScript puros, sem framework e sem servidor de aplicação. Ela usa:

* **`@mediapipe/tasks-vision` 1.0.1**, a versão JavaScript oficial do MediaPipe Tasks, carregada da CDN jsDelivr. É a mesma API `GestureRecognizer` usada em Python.
* **WebAssembly**, que executa no navegador o motor do MediaPipe escrito em C++.
* **WebGL (GPU)** para acelerar a inferência; se a GPU não estiver disponível, o código tenta automaticamente a CPU (XNNPACK).
* **O mesmo `gesture_recognizer.task`** treinado no MediaPipe Model Maker, sem conversão.
* **APIs do navegador:** `getUserMedia` para a webcam, `canvas` para desenhar o vídeo e os pontos da mão, e Web Speech API para falar o texto em pt-BR.

O laço principal usa `requestAnimationFrame` e só processa quando a câmera entrega um frame novo. Cada frame é espelhado no `canvas`, como o `cv2.flip` da versão desktop, e passado a `recognizeForVideo`. A interface mantém os recursos da versão local: limiar de confiança ajustável (50% por padrão), intervalo mínimo de 0,7 s entre letras para montar palavras, botões de espaço, apagar e limpar, e síntese de voz.

```mermaid
flowchart LR
    subgraph Antes["Antes: Space Docker com Gradio"]
        A1[Webcam no navegador] -->|frame enviado| A2((Internet)) --> A3[Servidor HF<br/>MediaPipe Python] --> A4((Internet)) -->|imagem devolvida| A5[Tela]
    end
    subgraph Depois["Depois: Space estático, MediaPipe no navegador"]
        HF[Hugging Face<br/>Space estático] -. download único do .task 8,4 MB .-> B2
        B1[Webcam<br/>getUserMedia] --> B2[MediaPipe JS<br/>WASM + WebGL] --> B3[Tela<br/>letra, palavra e voz]
    end
```

---

## 3. Por que a versão nova funciona

A versão nova elimina a rede do caminho de cada frame, e com isso as quatro causas do atraso desaparecem ao mesmo tempo. A taxa de quadros passa a depender só da câmera e do hardware de quem acessa.

| Aspecto | Versão antiga (Gradio) | Versão nova (navegador) |
| :--- | :--- | :--- |
| Onde o modelo roda | Servidor do Hugging Face (Python) | Dispositivo do usuário (WebAssembly) |
| Frames analisados | 2 por segundo enviados, 1 reconhecido a cada 1,5 s | Todo frame novo da câmera |
| Rede por frame | Imagem inteira vai e volta | Nenhuma; só o download inicial do modelo |
| Inferências por frame | 2 (RGB e BGR) | 1 |
| Hardware | 2 vCPUs gratuitas compartilhadas | GPU via WebGL, ou CPU |
| Vários usuários | Fila única, atraso cresce | Cada um usa a própria máquina |
| Privacidade | Vídeo sai do computador | Vídeo nunca sai do computador |

Os testes foram feitos no Chromium, em um servidor de nuvem sem GPU, em 4 de outubro de 2026:

* **Acurácia:** o modelo rodando no navegador acertou 267 de 315 imagens do conjunto `data/libras/test` (85%), o que confirma que o `.task` funciona sem conversão.
* **Velocidade:** com uma câmera simulada, a inferência levou 64 ms por frame, cerca de 15 FPS usando apenas a CPU. Em um computador com placa de vídeo o desempenho tende a ser maior, mas isso não foi medido.

Mesmo no pior caso medido, são cerca de 15 reconhecimentos por segundo contra um a cada 1,5 s na versão antiga.

---

## 4. Publicação no Hugging Face

O mesmo Space foi mantido, mas trocado de Docker para estático. No cabeçalho do `README.md`, `sdk: docker` virou `sdk: static`, a linha `app_port: 7860` saiu e entrou `app_file: web/index.html`. Um Space estático apenas serve arquivos por HTTPS, sem container e sem CPU de servidor, o que também libera a câmera no navegador.

Houve um ajuste necessário para o modelo. O Space estático publica só a pasta `web/`, então o caminho relativo `../models/gesture_recognizer.task` não existia no site. A solução foi escolher o endereço do modelo pelo domínio:

```js
const MODEL_URL = location.hostname.endsWith(".hf.space")
  ? "https://huggingface.co/spaces/RubensMota13/LibrasMediaPipe/resolve/main/models/gesture_recognizer.task"
  : "../models/gesture_recognizer.task";
```

No Space, o navegador baixa o `.task` (8,4 MB) direto do repositório do Space; no computador local, usa a pasta `models/`. Depois da troca, a página chegou ao estado "Modelo pronto" e o arquivo servido tinha os 8,4 MB do modelo real.

Dois cuidados ficaram registrados:

* No GitHub, o `.task` está no Git LFS e aparece como um ponteiro de 132 bytes. O Space precisa do arquivo real, enviado pela API do Hugging Face a partir do computador local.
* O Space ainda guarda os arquivos da versão Docker (`app.py`, `Dockerfile`, `apt.txt`). Eles não afetam o site estático. No repositório do GitHub, o `src/web/app.py` e o `Dockerfile` foram removidos e continuam no histórico do git como registro da primeira abordagem.

### Como testar localmente

A câmera só abre em HTTPS ou em `localhost`. Na raiz do repositório:

```powershell
python -m http.server
```

Depois abra [http://localhost:8000/web/](http://localhost:8000/web/).

---

## 5. Conclusão

O desempenho de uma aplicação de visão computacional em tempo real depende tanto da arquitetura de execução quanto do modelo. O mesmo `gesture_recognizer.task` que atrasava cerca de 1,5 s no servidor passou a responder a cerca de 15 FPS quando executado no navegador, sem nenhuma alteração no treino. Mover a inferência para o cliente eliminou a latência de rede, a fila compartilhada e a limitação de CPU do plano gratuito, e ainda garantiu que as imagens do usuário não sejam enviadas a nenhum servidor.

O código em Python segue sendo a base do trabalho para o treino do modelo, o cálculo das métricas e a versão desktop. A versão web reaproveita o modelo resultante, o que mostra que o artefato treinado é portável entre plataformas.
