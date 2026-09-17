<!--
Data/Hora: 2026-09-17 14:58 (UTC-3)
Branch: main
Commit: 31a99a8
Status: Atualizado
-->

# Documentação Técnica - LibrasMediaPipe (TCC)

Bem-vindo ao índice central da documentação técnica do projeto de TCC **LibrasMediaPipe**. Esta documentação visa fornecer rastreabilidade, clareza arquitetural e fundamentação teórica/prática para o trabalho de conclusão de curso em Ciência de Dados e Inteligência Artificial.

---

## Índice da Documentação

1. **[Backlog e Planejamento do TCC (todo.md)](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/todo.md)**
   * Controle de tarefas pendentes e concluídas em visão computacional, experimentos de ML, otimizações no Hugging Face e escrita da monografia.

2. **[Ambientes de Execução: Local vs. Hugging Face Spaces](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/docs/ambientes/local-vs-huggingface.md)**
   * Análise comparativa detalhada entre o ambiente de desktop local (`detectar_libras.py`) e o ambiente em nuvem (`app.py`).
   * Diagnóstico do gargalo de desempenho (*lag* e latência na nuvem), peculiaridades de áudio (TTS) e estratégias de otimização no Gradio.

3. **[Diretrizes do Projeto (GEMINI.md)](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/GEMINI.md)**
   * Arquitetura global, pipeline de extração de landmarks e convenções de código.

---

## Mapa Rápido dos Componentes de Código

* **Interface Web (Cloud):** [`app.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/app.py)
* **Interface Desktop com Buffer e TTS:** [`detectar_libras.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/detectar_libras.py)
* **Experimentos e Classificadores:** [`comparar_modelos.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/comparar_modelos.py)
* **Matriz de Confusão e Relatório:** [`gerar_metricas.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/gerar_metricas.py)
* **Curvas de Aprendizado (Loss / Acurácia):** [`graficos.py`](file:///c:/Users/Rubens/Desktop/projetinhos/LibrasMediaPipe/graficos.py)
