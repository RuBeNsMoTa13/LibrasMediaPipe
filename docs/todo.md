<!--
Data/Hora: 2026-10-06 19:30 (UTC-3)
Branch: claude/auditoria-pipeline-3l8ukb
Commit: c209803
Status: Atualizado
-->

# Backlog Oficial de Tarefas (TODO) - LibrasMediaPipe

Este documento registra o andamento do projeto de TCC em Ciência de Dados e IA. Cada item marcado com `[x]` já foi feito e cada item com `[ ]` ainda está pendente. Os caminhos citados são relativos à raiz do repositório, e os links abrem o arquivo correspondente.

---

## 1. Visão Computacional e Modelagem de Machine Learning

- [x] Extração dos 21 pontos da mão (landmarks) com o Google MediaPipe. Para cada mão, o MediaPipe entrega a posição dos 21 pontos na imagem, os mesmos 21 pontos em 3D (em metros) e se a mão é direita ou esquerda.
- [x] Treinamento do modelo `models/gesture_recognizer.task` com as 21 letras estáticas do alfabeto manual de LIBRAS (A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y), mais a classe "none" (nenhuma letra). O treino foi feito no Google Colab com o MediaPipe Model Maker, e a configuração usada no Colab não está no repositório (ver a seção 5).
- [x] Script de comparação com Random Forest e Support Vector Machine (SVM) em `src/evaluation/comparar_modelos.py`, que gerou a Tabela 1 do TCC. **Substituído** por [`src/treino/treinar_modelos.py`](../src/treino/treinar_modelos.py) e removido do repositório. O script antigo usava as 63 coordenadas cruas da tela e um SVM sem padronização dos dados (sem `StandardScaler`).
- [x] Treino de Random Forest, SVM e de uma Spiking Neural Network (SNN) em [`src/treino/treinar_modelos.py`](../src/treino/treinar_modelos.py). Os três modelos classificam o mesmo resumo de 128 números da mão, treinam também com cada mão espelhada e são salvos em `models/` (`rf_libras.pkl`, `svm_libras.pkl` e `snn_libras.pt`). O script imprime a tabela em LaTeX com as quatro linhas: os três modelos e o próprio `.task`, medido nas mesmas fotos de teste em que o MediaPipe encontrou a mão.
- [x] Normalização dos pontos da mão: passou a ser feita pela rede `gesture_embedder`, que vem dentro do `.task` e transforma os pontos da mão em um resumo de 128 números ([`src/classificadores/resumo.py`](../src/classificadores/resumo.py)). Random Forest, SVM e SNN usam esse resumo. A antiga normalização escrita à mão (centralizar no pulso e dividir pela escala, em `src/classificadores/landmarks.py`) nunca foi usada por nenhum modelo e foi removida.
- [x] Matriz de confusão e relatório de classificação (`classification_report`) do `.task` em [`src/avaliacao/matriz_confusao_task.py`](../src/avaliacao/matriz_confusao_task.py) (antigo `src/evaluation/gerar_metricas.py`). Usa todas as 1.153 fotos de `data/libras/test` e conta como erro ("Nenhum") as fotos em que o MediaPipe não encontra a mão. Gera a Figura 3 do TCC (`results/figures/matriz_confusao_task.png`).
- [x] Curvas de perda e acurácia por época do treino do `.task` em [`src/avaliacao/curvas_treino_task.py`](../src/avaliacao/curvas_treino_task.py) (antigo `src/evaluation/graficos.py`). O script apenas desenha os números copiados do log do treino no Colab. Gera as Figuras 1 e 2 do TCC (`task_curva_acuracia.png` e `task_curva_perda.png`).
- [x] Gráficos de comparação dos quatro modelos em [`src/avaliacao/graficos_modelos.py`](../src/avaliacao/graficos_modelos.py): matrizes de confusão de RF, SVM e SNN, curvas da SNN, métricas lado a lado, F1 por letra, quantidade de fotos por letra e taxa de detecção da mão por letra.
- [ ] Documentar a justificativa teórica para a ausência de letras dinâmicas (H, J, K, X, Z) e sugerir abordagens temporais (ex.: LSTM/GRU) como trabalhos futuros.

---

## 2. Aplicação Desktop Local (`src/desktop/detectar_libras.py`)

Código: [`src/desktop/detectar_libras.py`](../src/desktop/detectar_libras.py). Para rodar: `python src/desktop/detectar_libras.py [--modelo task|snn|rf|svm]`.

- [x] Captura de vídeo em tempo real com OpenCV, com a imagem espelhada (`cv2.flip`) para funcionar como um espelho.
- [x] Escolha do classificador: `--modelo task|snn|rf|svm` na linha de comando (padrão `task`) e tecla `M` para trocar durante o uso. Só entram na troca os modelos que estão salvos em `models/`.
- [x] Filtro de limiar de confiança mínima com ajuste em tempo real pelo teclado (`CONFIDENCE_THRESHOLD = 0.50`, ajustável com `+` e `-` em passos de 5%, entre 20% e 95%).
- [x] Buffer acumulador de caracteres (montagem de palavras) com intervalo mínimo entre letras (`LETTER_DELAY_SECONDS = 0.7`). Ao tirar a mão da câmera, ou ao fazer a posição neutra "none" (classe que só o `.task` tem), a última letra vista é esquecida, o que permite repetir a mesma letra.
- [x] Tratamento de tokens especiais e atalhos manuais de edição: espaço (`Espaço`), apagar a última letra (`Backspace` / `D`) e limpar a legenda (`C`).
- [x] Feedback visual colorido no overlay: verde para sinal com confiança igual ou acima do limiar, laranja para sinal abaixo do limiar e cinza quando não há gesto.
- [x] Síntese de voz assíncrona com `pyttsx3`, com fala manual pela tecla `Enter` e alternância pela tecla `V` para o modo automático, que fala e limpa a legenda depois de 5 segundos sem nova letra (`LEGEND_CLEAR_SECONDS = 5`).
- [x] Pop-up do manual de atalhos aberto pela tecla `H` (e fechado com `H` ou `ESC`) na própria janela de vídeo. Enquanto ele está aberto, as letras não entram na legenda, mas a câmera e o reconhecimento continuam rodando.
- [x] Interface desktop com janela redimensionável (`cv2.WINDOW_NORMAL`), tentativa de captura em HD (1280x720) e HUD em bandejas (barra superior e inferior) sem sobreposição de textos.
- [x] Tipografia TrueType (`Segoe UI` / `Arial`) com suporte a caracteres acentuados da língua portuguesa e paleta de cores de alto contraste para visibilidade a distância. As fontes são procuradas na pasta de fontes do Windows; em outro sistema, o app usa a fonte padrão do OpenCV, que não mostra acentos.

---

## 3. Aplicação Web no Navegador (`web/index.html`)

Código: [`web/index.html`](../web/index.html). Publicada no Space estático do Hugging Face: <https://rubensmota13-librasmediapipe.static.hf.space>. Para testar no computador, rode `python -m http.server` na raiz do repositório e abra <http://localhost:8000/web/>. A história completa da troca está em [`ambientes/versao-web-navegador.md`](ambientes/versao-web-navegador.md).

- [x] Versão web refeita como um único arquivo HTML e JavaScript, usando a biblioteca JavaScript do MediaPipe (`@mediapipe/tasks-vision` 1.0.1) e o mesmo `gesture_recognizer.task`, sem conversão.
- [x] Inferência no próprio navegador (*client-side*): o modelo roda no computador de quem acessa, com a placa de vídeo (GPU, via WebGL) ou, se ela não estiver disponível, com o processador (CPU).
- [x] Nenhum frame da webcam é enviado a servidor: o vídeo nunca sai do computador do usuário. Só o arquivo do modelo é baixado, uma vez, ao abrir a página.
- [x] Voz pela Web Speech API do navegador, em português do Brasil, no lugar do `pyttsx3` (que roda só no computador local).
- [x] Mesmos recursos de soletração do desktop: limiar de confiança ajustável (padrão 50%, de 20% a 95%), intervalo mínimo de 0,7 s entre letras, botões de espaço, apagar e limpar, fala manual (`Enter`), modo de voz automático (`V`) e manual de atalhos (`H`).
- [x] Space do Hugging Face trocado de Docker para estático: o cabeçalho do `README.md` usa `sdk: static` e `app_file: web/index.html`.
- [x] **Versão Gradio removida.** A primeira versão web (`src/web/app.py`, com interface Gradio rodando no servidor) e o `Dockerfile` foram apagados do repositório, e `gradio` e `huggingface-hub` saíram do `requirements.txt`. Os itens antigos de combate ao lag dessa versão (enviar menos dados pela rede, avaliar WebRTC, levar a voz para o navegador e ajustar a resolução no painel do Gradio) foram resolvidos pela mudança para o navegador ou deixaram de fazer sentido.
- Observação: a versão web usa apenas o `.task`. Random Forest, SVM e SNN existem só no app desktop.

---

## 4. Documentação e Monografia do TCC

- [x] Criação do [`GEMINI.md`](../GEMINI.md) na raiz do projeto.
- [x] Reorganização arquitetural do repositório em camadas funcionais (`src/`, `models/`, `data/`, `results/`).
- [x] Criação do índice de documentação em [`docs/README.md`](README.md).
- [x] Documento comparando o ambiente local com o Hugging Face Spaces (`docs/ambientes/local-vs-huggingface.md`), escrito para diagnosticar o lag da versão Gradio. Foi removido junto com essa versão; o resumo do diagnóstico ficou em [`ambientes/versao-web-navegador.md`](ambientes/versao-web-navegador.md).
- [x] Documento da versão web no navegador em [`ambientes/versao-web-navegador.md`](ambientes/versao-web-navegador.md).
- [x] **Limpeza e reorganização do repositório (6 de outubro de 2026):**
  - Apagados os arquivos que nada usava: a versão Gradio (`src/web/app.py` e `Dockerfile`), o script antigo `src/evaluation/comparar_modelos.py`, a normalização sem uso `src/classificadores/landmarks.py`, o cache `__pycache__/app.cpython-312.pyc`, o `data/libras/.gitattributes` e o documento `docs/ambientes/local-vs-huggingface.md`.
  - A pasta `src/evaluation` foi dividida em [`src/treino/`](../src/treino/) (treino dos modelos) e [`src/avaliacao/`](../src/avaliacao/) (gráficos e métricas), com nomes que dizem o que cada script faz: `testar_snn.py` virou `src/treino/treinar_modelos.py`, `gerar_metricas.py` virou `src/avaliacao/matriz_confusao_task.py`, `graficos.py` virou `src/avaliacao/curvas_treino_task.py` e `graficos_modelos.py` foi para `src/avaliacao/` com o mesmo nome.
  - Gráficos do `.task` renomeados em `results/figures/`: `grafico_acuracia.png` virou `task_curva_acuracia.png`, `grafico_perda.png` virou `task_curva_perda.png` e `matriz_de_confusao.png` virou `matriz_confusao_task.png`.
- [x] Documento de metodologia e limitações da avaliação em [`avaliacao/metodologia-e-limitacoes.md`](avaliacao/metodologia-e-limitacoes.md).
- [x] Capítulo de Metodologia e Resultados comparativos (LaTeX) escrito com os números de `comparar_modelos.py` (Tabela 1 do TCC, versão de 18/09). Esse script foi **substituído** por [`src/treino/treinar_modelos.py`](../src/treino/treinar_modelos.py); a atualização do texto e da tabela está na seção 5.
- [ ] Adicionar seção de análise de limitações de hardware e latência de rede na monografia. O material de base está em [`ambientes/versao-web-navegador.md`](ambientes/versao-web-navegador.md) e [`avaliacao/metodologia-e-limitacoes.md`](avaliacao/metodologia-e-limitacoes.md).

---

## 5. Avaliação sem vazamento

"Vazamento" aqui quer dizer que o teste não é tão novo para o modelo quanto parece. A base `data/libras` já veio dividida em `train/` e `test/` (commit `31a99a8`), nenhum script do projeto faz essa divisão e as fotos não dizem de qual pessoa ou vídeo vieram. Há pelo menos 4 a 5 mãos diferentes, e as mesmas mãos aparecem no treino e no teste. Um classificador simples que só copia a letra da mão de treino com os pontos mais parecidos (vizinho mais próximo, 1-NN) acerta 99,2% do teste. Por isso as acurácias atuais (Random Forest 0,994, SNN 0,990, SVM 0,982 e `.task` 0,891, nas 1.135 fotos de teste em que o MediaPipe achou a mão) medem o desempenho com as mesmas pessoas, não com um usuário novo. Quando um grupo de gravações parecidas (mesma cor de pele, fundo e iluminação) fica de fora do treino, o SVM cai para 77% a 88% nesse grupo, com confusões como N e M, P e N, C e O. Os detalhes estão em [`avaliacao/metodologia-e-limitacoes.md`](avaliacao/metodologia-e-limitacoes.md).

- [x] Os quatro modelos são medidos nas mesmas fotos de teste (as 1.135 de 1.153 em que o MediaPipe encontrou a mão), em [`src/treino/treinar_modelos.py`](../src/treino/treinar_modelos.py) e [`src/avaliacao/graficos_modelos.py`](../src/avaliacao/graficos_modelos.py).
- [ ] **Teste com mão nova:** Rubens grava cerca de 30 fotos por letra da própria mão, com fundos e iluminação variados, numa pasta `data/usuario_novo/` que nunca entra no treino nem na escolha de ajustes dos modelos. O TCC passa a mostrar duas colunas de resultado: "teste do dataset (mesmas pessoas)" e "usuário novo".
- [ ] **Validação por grupo:** como não há rótulo de pessoa, separar as fotos em grupos de gravações parecidas (ou usar os metadados do dataset original, se existirem) e treinar deixando um grupo de fora de cada vez (`GroupKFold`), num script novo em `src/avaliacao/` que mostre a média e o pior grupo.
- [ ] **Validação separada do teste:** criar um conjunto de validação próprio e escolher por ele, e nunca pelo teste, a função de perda, o número de épocas e o parâmetro `C` do SVM. Hoje a perda da SNN foi escolhida olhando a acurácia no teste, e `graficos_modelos.py` mede o teste a cada época.
- [ ] **Retreinar o `.task` no Model Maker, guardando o notebook no repositório:** o `.task` acerta cerca de 0,89 até nas fotos de treino, porque está subtreinado (padrão do Model Maker: lotes de 2 fotos e 10 épocas). A mesma cabeça de classificação treinada por mais tempo chega a 0,99. Retreinar com lotes maiores e mais épocas e salvar o notebook com os hiperparâmetros (por exemplo, em `notebooks/model_maker.ipynb`), para que o treino possa ser repetido.
- [ ] **Alinhar o texto do TCC ao código:** a versão de 18/09 diz que os pontos foram normalizados no pulso, que houve divisão estratificada 70/15/15 e que foi usada a base V-LIBRASIL, mas o código nunca fez isso. A Tabela 1 saiu de coordenadas cruas da tela e de um SVM sem padronização. Trocar a tabela pelos números de `treinar_modelos.py`, descrever o resumo de 128 números e dizer que a base veio dividida, com as mesmas mãos no treino e no teste.
