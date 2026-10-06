# Regras Locais do Workspace - LibrasMediaPipe (TCC)

## 1. Escopo e Diretrizes do Projeto
* Este repositório contém o código-fonte, experimentos de aprendizado de máquina e documentação do Trabalho de Conclusão de Curso (TCC) em Ciência de Dados e Inteligência Artificial.
* O projeto conta com dois ambientes fundamentais:
  1. **Ambiente Local Desktop (`src/desktop/detectar_libras.py`):** Aplicativo em Python com OpenCV, focado em baixa latência e voz (TTS) com a voz instalada no sistema operacional. É o único que permite trocar o classificador ao vivo entre o `.task`, a SNN, o Random Forest e o SVM.
  2. **Ambiente Web no Navegador (`web/index.html`):** Página em HTML e JavaScript que roda o `gesture_recognizer.task` no próprio navegador com o MediaPipe JavaScript, sem servidor de inferência. Está publicada como Space estático no Hugging Face (<https://rubensmota13-librasmediapipe.static.hf.space>) e serve para demonstrar o projeto sem instalar nada.

## 2. Padrões de Código e Versionamento
* Não realizar commits automáticos. Todas as alterações devem permanecer na working tree para revisão do usuário.
* Toda documentação técnica nova deve ser roteada para a pasta `docs/` e indexada no `docs/README.md`.
* Documentos na pasta `docs/` devem manter o cabeçalho padronizado de rastreabilidade (Data/Hora em UTC-3, Branch ativa, Commit de referência e Status).
