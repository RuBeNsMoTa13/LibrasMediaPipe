# Regras Locais do Workspace - LibrasMediaPipe (TCC)

## 1. Escopo e Diretrizes do Projeto
* Este repositório contém o código-fonte, experimentos de aprendizado de máquina e documentação do Trabalho de Conclusão de Curso (TCC) em Ciência de Dados e Inteligência Artificial.
* O projeto conta com dois ambientes fundamentais:
  1. **Ambiente Local Desktop (`detectar_libras.py`):** Focado em máxima performance (30+ FPS), baixa latência e TTS nativo no SO.
  2. **Ambiente Remoto Web (`app.py` / Hugging Face Spaces):** Focado em acessibilidade e facilidade de demonstração sem instalação local.

## 2. Padrões de Código e Versionamento
* Não realizar commits automáticos. Todas as alterações devem permanecer na working tree para revisão do usuário.
* Toda documentação técnica nova deve ser roteada para a pasta `docs/` e indexada no `docs/README.md`.
* Documentos na pasta `docs/` devem manter o cabeçalho padronizado de rastreabilidade (Data/Hora em UTC-3, Branch ativa, Commit de referência e Status).
