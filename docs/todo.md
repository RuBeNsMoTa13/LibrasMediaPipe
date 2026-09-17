<!--
Data/Hora: 2026-09-17 14:58 (UTC-3)
Branch: main
Commit: 31a99a8
Status: Atualizado
-->

# Backlog Oficial de Tarefas (TODO) - LibrasMediaPipe

Este documento registra o status de desenvolvimento do projeto de TCC em Ciência de Dados e IA.

---

## 1. Visão Computacional e Modelagem de Machine Learning

- [x] Extração de 21 landmarks tridimensionais das mãos com Google MediaPipe.
- [x] Treinamento do modelo `gesture_recognizer.task` com 21 classes do alfabeto estático de LIBRAS.
- [x] Script de benchmark comparativo com Random Forest e Support Vector Machine (SVM) em [`comparar_modelos.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/comparar_modelos.py).
- [x] Geração de métricas de avaliação (`classification_report`) e exportação da matriz de confusão em [`gerar_metricas.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/gerar_metricas.py).
- [x] Visualização gráfica das curvas de perda e acurácia por época em [`graficos.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/graficos.py).
- [ ] Implementar normalização espacial dos landmarks (translação relativa ao pulso $x_0, y_0, z_0$ e divisão pela escala) para aumentar invariância espacial no SVM/Random Forest.
- [ ] Documentar a justificativa teórica para a ausência de letras dinâmicas (H, J, K, X, Z) e sugerir abordagens temporais (ex: LSTM/GRU) como trabalhos futuros.

---

## 2. Aplicação Desktop Local (`detectar_libras.py`)

- [x] Captura de vídeo em tempo real com OpenCV.
- [x] Filtro de limiar de confiança mínima (`CONFIDENCE_THRESHOLD = 0.75`).
- [x] Buffer acumulador de caracteres (montagem de palavras) com debounce temporal (`LETTER_DELAY_SECONDS = 0.7`).
- [x] Tratamento de tokens especiais: espaço e deleção.
- [x] Síntese de voz assíncrona com `pyttsx3` disparada após tempo de inatividade (`LEGEND_CLEAR_SECONDS = 3`).

---

## 3. Aplicação Web e Otimizações para Hugging Face Spaces (`app.py`)

- [x] Implementação de interface interativa com Gradio em [`app.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/app.py).
- [x] Containerização com Docker (`Dockerfile`) para execução no Hugging Face Spaces.
- [x] Otimizações básicas de latência: downscale de entrada (`DOWNSCALE_WIDTH = 192`) e throttling/frame-skipping (`PROCESS_EVERY_N = 3`).
- [ ] **Melhoria de Performance no Spaces (Combate ao Lag):**
  - [ ] Reduzir a carga de transmissão de vídeo: enviar apenas predições de texto/landmarks em vez de retransmitir o frame inteiro do servidor de volta ao cliente.
  - [ ] Avaliar adoção de inferência *client-side* (via MediaPipe JavaScript ou WebAssembly) ou integração com WebRTC de baixa latência (`fastrtc` / Gradio WebRTC nativo).
  - [ ] Substituir o TTS de servidor por TTS *client-side* (Web Speech API via JavaScript no navegador), já que servidores em nuvem não possuem dispositivo de saída de áudio nativo.
  - [ ] Permitir ajuste dinâmico da resolução e taxa de amostragem pelo usuário no painel do Gradio.

---

## 4. Documentação e Monografia do TCC

- [x] Criação do [`GEMINI.md`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/GEMINI.md) na raiz do projeto.
- [x] Criação do índice de documentação em [`docs/README.md`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/README.md).
- [x] Documento aprofundado comparando o ambiente local com o Hugging Face Spaces em [`docs/ambientes/local-vs-huggingface.md`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md).
- [ ] Elaboração do capítulo de Metodologia e Resultados comparativos (LaTeX) com base nos dados gerados por `comparar_modelos.py`.
- [ ] Adicionar seção de análise de limitações de hardware e latência de rede na monografia.
