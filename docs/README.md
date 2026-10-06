<!--
Data/Hora: 2026-10-06 19:30 (UTC-3)
Branch: claude/auditoria-pipeline-3l8ukb
Commit: c209803
Status: Atualizado
-->

# Documentação Técnica - LibrasMediaPipe (TCC)

Este é o índice da documentação técnica do projeto **LibrasMediaPipe**, o Trabalho de Conclusão de Curso em Ciência de Dados e Inteligência Artificial que reconhece pela webcam as letras do alfabeto manual de LIBRAS. Aqui estão os documentos que explicam as decisões do projeto, como os modelos foram avaliados e o que ainda falta fazer. Todos os links levam a arquivos deste repositório.

---

## Índice da Documentação

1. **[Backlog e planejamento do TCC](todo.md)**
   * Lista do que já foi feito e do que ainda falta, dividida em modelos de Machine Learning, aplicativo desktop, aplicativo web, escrita da monografia e avaliação sem vazamento.
   * Também registra o que foi substituído ou removido do repositório e por quê.

2. **[Versão web no navegador](ambientes/versao-web-navegador.md)**
   * Explica por que a primeira versão web, em Gradio rodando no servidor do Hugging Face, chegava a ficar cerca de 1,5 s atrasada, e como a versão atual roda o MediaPipe direto no navegador, publicada como Space estático.
   * Traz os testes feitos com a versão nova: 85% de acerto em 315 imagens de teste e cerca de 15 quadros por segundo usando só a CPU.

3. **[Metodologia e limitações da avaliação](avaliacao/metodologia-e-limitacoes.md)**
   * Explica como os quatro classificadores foram treinados e medidos, de onde veio a divisão entre fotos de treino e de teste e o que os números significam.
   * Mostra por que as acurácias atuais medem o desempenho com as mesmas pessoas que aparecem no treino, e não com um usuário novo, e quais testes faltam para medir isso.

4. **[Diretrizes do projeto (GEMINI.md)](../GEMINI.md)**
   * Resumo da arquitetura, da estrutura de pastas, dos comandos e das convenções do projeto, escrito para orientar o assistente Gemini.
   * Inclui a lista de gráficos em `results/figures/` com o script que gera cada um e os cuidados ao falar das métricas.

---

## Mapa Rápido dos Componentes de Código

Os caminhos abaixo são relativos à raiz do repositório.

**Aplicativos**

* **App web no navegador:** [`web/index.html`](../web/index.html). Roda o `gesture_recognizer.task` no próprio navegador com o MediaPipe em JavaScript. É a página publicada no Space estático do Hugging Face.
* **App desktop com soletração e voz:** [`src/desktop/detectar_libras.py`](../src/desktop/detectar_libras.py). Abre a webcam com o OpenCV, junta as letras em palavras e fala o texto. A tecla `M` troca o classificador entre `.task`, SNN, Random Forest e SVM.

**Classificadores** (pasta [`src/classificadores/`](../src/classificadores/))

* [`resumo.py`](../src/classificadores/resumo.py): lê a rede `gesture_embedder` de dentro do `.task` e transforma os pontos de uma mão em um resumo de 128 números. Também extrai esse resumo de todas as fotos do dataset e o guarda em cache.
* [`classicos.py`](../src/classificadores/classicos.py): carrega o Random Forest e o SVM salvos em `models/` para o app desktop.
* [`snn.py`](../src/classificadores/snn.py): define a arquitetura da SNN (rede neural de impulsos) e carrega o modelo salvo em `models/`.

**Treino**

* [`src/treino/treinar_modelos.py`](../src/treino/treinar_modelos.py): treina o Random Forest, o SVM e a SNN sobre o resumo de 128 números, salva os três em `models/` e imprime a tabela em LaTeX com as métricas dos quatro modelos, incluindo a linha do `.task` medida nas mesmas fotos de teste.

**Avaliação e gráficos** (pasta [`src/avaliacao/`](../src/avaliacao/))

* [`graficos_modelos.py`](../src/avaliacao/graficos_modelos.py): gera os gráficos de comparação dos quatro modelos, as matrizes de confusão de RF, SVM e SNN, as curvas de treino da SNN, o F1 por letra, a distribuição das fotos e a taxa de detecção da mão.
* [`matriz_confusao_task.py`](../src/avaliacao/matriz_confusao_task.py): gera a matriz de confusão do `.task` em todas as fotos de teste, contando como erro as fotos em que o MediaPipe não encontrou a mão (Figura 3 do TCC).
* [`curvas_treino_task.py`](../src/avaliacao/curvas_treino_task.py): desenha as curvas de acurácia e de perda do treino do `.task` no Google Colab, a partir de números copiados do registro desse treino (Figuras 1 e 2 do TCC).

A lista completa de gráficos, com o script que gera cada um, está no [README principal](../README.md) e no [GEMINI.md](../GEMINI.md).

---

## Convenções desta pasta

* Todo documento técnico novo fica em `docs/` e deve ser incluído neste índice.
* Todo arquivo em `docs/` começa com um cabeçalho de rastreabilidade com a data e hora (UTC-3), a branch, o commit de referência e o status do documento.
* Arquivos removidos do repositório, como a antiga versão web em Gradio e os scripts antigos de `src/evaluation/`, continuam disponíveis no histórico do git.
