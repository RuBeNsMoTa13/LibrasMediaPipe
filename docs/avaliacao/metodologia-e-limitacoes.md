<!--
Data/Hora: 2026-11-27 17:30 (UTC-3)
Branch: claude/project-thread-o1zhx7
Status: Atualizado (base completa do Kaggle)
-->

# Metodologia da avaliação e limitações dos resultados

Este documento explica, para quem não é da área e para a banca do TCC, de onde vêm os dados, como eles chegam a cada modelo e o que as métricas podem e não podem afirmar. Todos os números são da base completa do Kaggle (46.262 fotos), usada desde 27/11/2026 para os quatro classificadores, e são os mesmos do TCC.

## Resumo

1. **Os quatro modelos acertam entre 98,9% e 99,7% das fotos de teste, mas o teste tem as mesmas mãos do treino.** Esses números medem "as mesmas mãos, em outro momento da gravação", e não "uma pessoa nova usando o app".
2. **Na validação por grupo de gravações parecidas, deixar um grupo fora do treino derrubou o acerto do SVM nesse grupo para 82% em 1 dos 6 grupos e para 95,5% em outro** (seção 5.2). Nos outros 4 grupos, ele ficou entre 97,9% e 98,7%. É a estimativa mais próxima do uso ao vivo que esta base permite.
3. **Um classificador que só copia a letra da foto de treino mais parecida (1-NN) acerta 99,8% do teste**, o que confirma que o teste é parecido demais com o treino.

## Termos usados

* **Landmarks:** os 21 pontos que o MediaPipe marca na mão (pulso, juntas e pontas dos dedos).
* **Resumo de 128 números:** lista de 128 números que descreve o formato da mão, calculada por uma rede neural do Google (`gesture_embedder`) a partir dos landmarks. Na literatura, chama-se *embedding*.
* **Treino, validação e teste:** o modelo aprende com o treino; a validação serve para fazer escolhas durante o desenvolvimento; o teste mede o resultado final com fotos que o modelo nunca viu.
* **Vazamento:** quando algo do teste já "apareceu" no treino de alguma forma, o que deixa a nota do teste mais alta do que seria com dados realmente novos.
* **Hiperparâmetros:** escolhas feitas antes do treino, como o número de épocas, o tamanho do lote ou o número de árvores.
* **Época:** uma passada completa do modelo por todas as fotos de treino.
* **Lote:** quantas fotos o modelo vê de uma vez antes de cada ajuste dos seus pesos.
* **Função de perda:** a conta que mede o quanto o modelo está errando durante o treino. O treino tenta deixar esse número cada vez menor.
* **Semente:** número que fixa os sorteios internos do treino, para que o resultado possa ser repetido.
* **KMeans:** algoritmo que separa itens em grupos parecidos sem saber de antemão quais são esses grupos.

---

## 1. De onde vêm os dados e como estão divididos

As fotos vêm do dataset do Kaggle `williansoliveira/libras`, que já vem dividido nas pastas `train/<letra>/` e `test/<letra>/`. São 21 letras estáticas: A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W e Y. Os scripts leem o dataset pela variável `LIBRAS_DADOS` ou baixam com o `kagglehub` (`pasta_dados()` em [`resumo.py`](../../src/classificadores/resumo.py)).

| Pasta | Fotos | Por letra | Fotos em que o MediaPipe achou a mão |
|---|---|---|---|
| `train` | 34.714 | 1.614 a 1.686 | 34.284 (98,8%) |
| `test` | 11.548 | 450 a 583 | 11.402 (98,7%) |

**Como a divisão foi feita:**

* **Nenhum script do projeto divide os dados.** A divisão é a do próprio dataset, com cerca de 75% para treino e 25% para teste.
* Os nomes dos arquivos são só números, **sem identificação de pessoa, vídeo ou sessão**.

**Quantas pessoas aparecem.** Sem rótulo de pessoa não dá para contar com certeza, mas a inspeção visual mostra **pelo menos 4 a 5 mãos diferentes** (por exemplo, pele clara com anel, pele clara com pulseira, pele mais escura, mão com relógio e mão com manga escura). **As mesmas mãos, com os mesmos acessórios e o mesmo fundo, aparecem no treino e no teste.**

**O teste é um sorteio de fotos?** Não exatamente, mas fica muito perto do treino: o "vizinho mais próximo" (1-NN), que só copia a letra da foto de treino mais parecida, acerta **99,8%** do teste usando o resumo de 128 números (e 99,97% usando as coordenadas normalizadas). **A divisão parece ser por trecho de gravação, e não por pessoa.**

---

## 2. Como cada modelo recebe os dados

### 2.1 O caminho de uma imagem até a letra

1. **Imagem.** No treino, uma foto do dataset. No app desktop, cada frame da webcam é espelhado antes de tudo ([`detectar_libras.py`, linha 293](../../src/desktop/detectar_libras.py#L293)); a versão web faz o mesmo.
2. **O MediaPipe acha a mão** e devolve, da primeira mão encontrada: os 21 landmarks na imagem (de 0 a 1), os mesmos 21 pontos em 3D (em metros, centrados na mão) e a probabilidade de ser a mão direita. **Fotos sem mão encontrada ficam fora do treino e do teste** (`_extrair_pasta` em [`resumo.py`](../../src/classificadores/resumo.py)).
3. **Resumo de 128 números.** Os landmarks passam pela rede `gesture_embedder.tflite`, lida de dentro do próprio `.task` (`carregar_rede` e `resumir` em [`resumo.py`](../../src/classificadores/resumo.py)). **É essa rede do Google que ajusta posição e tamanho da mão.** O projeto não normaliza os landmarks por conta própria em nenhum classificador.
4. **Classificação**, de dois jeitos: **(a)** o próprio `.task`, que passa o resumo pela pequena camada treinada no Colab (seção 3); ou **(b)** Random Forest, SVM ou SNN, que recebem exatamente o mesmo resumo. Os quatro modelos partem, portanto, da mesma base.
5. **No app,** a letra só é aceita se a confiança passar do limiar (50% por padrão, ajustável com `+` e `-`), depois de ficar 1 s estável (com uma barra de progresso acima da mão), com no mínimo 0,7 s entre letras, e a palavra pode ser falada em voz alta. A versão web usa só o `.task`, também com limiar padrão de 50%.

O app ao vivo usa a mesma função de resumo do treino, então treino e uso real recebem os dados do mesmo jeito. Como o app espelha o frame e a pessoa pode usar a outra mão, cada mão de treino entra duas vezes, como está e espelhada ([`treinar_modelos.py`, linhas 121 a 125](../../src/treino/treinar_modelos.py#L121-L125)): o treino passa de 34.284 para 68.568 linhas. O teste não é espelhado.

### 2.2 Hiperparâmetros e normalizadores

| Modelo | Normalizador | Configuração | Onde |
|---|---|---|---|
| Random Forest | Nenhum (árvores de decisão não precisam) | 100 árvores, semente 42; o resto no padrão do scikit-learn | [`treinar_modelos.py`, linha 149](../../src/treino/treinar_modelos.py#L149) |
| SVM | `StandardScaler`, que deixa cada número com média 0 e desvio 1, calculado só no treino | Kernel RBF, `C=1` e `gamma='scale'` (padrões do scikit-learn), `probability=True` para o app mostrar a confiança | [`treinar_modelos.py`, linha 152](../../src/treino/treinar_modelos.py#L152) |
| SNN (snnTorch) | Padronização à mão com média e desvio do treino, guardados em `snn_libras.pt` | Entrada com os 128 números do resumo, duas camadas escondidas de 128 neurônios LIF e uma camada de saída com 21 neurônios LIF, uma por letra (decaimento 0,9); 25 passos de tempo; Adam com taxa 0,002; 50 épocas; lotes de 64; perda `mse_count_loss` (80% de disparos na letra certa, 10% nas outras) | [`treinar_modelos.py`, linhas 56 a 71](../../src/treino/treinar_modelos.py#L56-L71); [`snn.py`](../../src/classificadores/snn.py) |
| MediaPipe `.task` | Feito dentro do `.task` | Ver seção 3 | [`notebooks/retreinar_task_py310.ipynb`](../../notebooks/retreinar_task_py310.ipynb) |

Na SNN (rede neural de impulsos), cada neurônio "dispara" ou não a cada passo de tempo, como um neurônio biológico. O neurônio LIF (do inglês *leaky integrate-and-fire*) acumula o sinal que recebe, perde aos poucos parte do que acumulou (o decaimento) e dispara quando passa de um limite. A letra escolhida é a do neurônio de saída que disparou mais vezes nos 25 passos.

**Os normalizadores são ajustados só com o treino, então não há vazamento no pré-processamento.** O vazamento está na divisão dos dados (seção 5).

---

## 3. Como o `.task` foi treinado

O `.task` foi treinado em 26/11/2026 no Google Colab com o MediaPipe Model Maker 0.2.1.4 (Python 3.10), pelo notebook [`notebooks/retreinar_task_py310.ipynb`](../../notebooks/retreinar_task_py310.ipynb). O histórico e as métricas estão em [`results/treinos/task.json`](../../results/treinos/task.json).

* O `.task` é um arquivo compactado com o detector de mão, a rede `gesture_embedder.tflite` (fixa, do Google), um classificador de 8 gestos padrão do Google (não usado) e o `custom_gesture_classifier.tflite`, **a única parte treinada no Colab**.
* Essa parte é pequena: `BatchNormalization`, `ReLU`, *dropout* de 0,05 e uma camada `Dense` com 22 saídas (as 21 letras mais `none`, que o Model Maker sempre cria). É o que o Model Maker monta com `layer_widths=[]`, ou seja, **sem camadas ocultas**.
* Configuração: taxa de aprendizado 0,001 com decaimento de 0,99 por época, **lotes de 32, 100 épocas**, perda *focal loss* (gamma 2), semente 42.
* Fotos: só a pasta `train` do dataset, espelhada e depois sorteada em 90% para treino e 10% para validação (61.125 e 6.792 mãos). A pasta `test` não entrou no treino.
* Na última época, o `.task` acertou 98,8% no treino e 99,8% na validação.

A versão de 2025 do `.task` tinha sido treinada com o padrão do Model Maker (lotes de 2 e 10 épocas) e acertava menos (cerca de 89%) até nas próprias fotos de treino. Com o treino mais longo, esse problema deixou de existir.

---

## 4. Resultados e como lê-los

### 4.1 Tabela principal

Os quatro modelos foram medidos nas **mesmas 11.402 fotos de teste** em que o MediaPipe achou a mão. Os números de RF, SVM e SNN são impressos por `python src/treino/treinar_modelos.py`, que também treina e salva os três em `models/`. A linha do `.task` usa a letra que ele mesmo deu a cada foto, sempre a mais provável (`score_threshold=0`).

| Modelo | Acurácia | Precisão (ponderada) | Recall (ponderado) | F1 (ponderado) |
|---|---|---|---|---|
| Random Forest | 0,997 | 0,997 | 0,997 | 0,997 |
| MediaPipe `.task` | 0,992 | 0,993 | 0,992 | 0,992 |
| SVM | 0,991 | 0,992 | 0,991 | 0,991 |
| SNN | 0,989 | 0,989 | 0,989 | 0,989 |

* **Acurácia:** fração das fotos com a letra certa (0,992 é 99,2% de acertos).
* **Precisão:** quando o modelo diz "é a letra X", quantas vezes é mesmo X.
* **Recall:** das fotos da letra X, quantas o modelo reconheceu.
* **F1:** junta precisão e recall num número só; só é alto se os dois forem.
* **Ponderada:** calculada letra por letra e depois tirada a média, com mais peso para as letras com mais fotos.

As letras com menor F1 são parecidas entre si: no `.task`, O (0,95), C (0,96), M e N (0,98); no SVM e na SNN, M e N (0,93 a 0,94); no Random Forest, F e T (0,99).

Os gráficos (comparação, F1 por letra, matrizes de RF, SVM e SNN, curvas da SNN, distribuição de fotos e detecção da mão) são gerados por `python src/avaliacao/graficos_modelos.py` em `results/figures/`.

### 4.2 Contando as fotos sem mão como erro

`python src/avaliacao/matriz_confusao_task.py` passa **todas as 11.548 fotos de teste** pelo `.task` e conta como erro ("Nenhum") as 146 em que a mão não foi encontrada. Assim o `.task` fica com **0,980 de acurácia**. Esse script gera `results/figures/matriz_confusao_task.png`.

### 4.3 Comparação de entradas

Para ver se o resumo de 128 números ajuda, RF e SVM também foram treinados com as 63 coordenadas dos landmarks, nas mesmas fotos:

| Entrada | RF | SVM |
|---|---|---|
| Coordenadas cruas da tela | 0,981 | 0,995 |
| Coordenadas normalizadas | 0,998 | 0,996 |
| Resumo de 128 números | 0,997 | 0,989 |
| Resumo de 128 números + espelhamento | 0,997 | 0,991 |

Com esta base, as coordenadas normalizadas acertam tanto quanto o resumo. A diferença entre as entradas é pequena perto da queda da validação por grupo (seção 5.2).

### 4.4 O que estes números dizem

* **Dizem** que, para as mesmas mãos e fundos da base, os quatro modelos separam bem as 21 letras.
* **Não dizem** quanto o sistema acerta com uma pessoa nova na frente da webcam (seção 5). Ao vivo, com uma mão nova, os modelos ainda confundem letras parecidas, como U com R.

---

## 5. Limitações

### 5.1 Vazamento de pessoa e de sessão

O teste tem as mesmas mãos, acessórios e fundo do treino (seção 1). É como uma prova com os mesmos exercícios da aula, mudando alguns números: dá para ir bem sem ter aprendido a matéria. Sem rótulo de pessoa, não é possível separar treino e teste por pessoa. Trocar a entrada dos modelos pelo resumo de 128 números não resolve isso, porque o problema está na divisão dos dados, e não na forma de descrever a mão.

### 5.2 Validação por grupo de gravação

Para estimar o efeito do vazamento, as 45.686 fotos com mão (treino e teste juntos) foram separadas em 6 grupos de gravações parecidas, pela cor da pele, cor do fundo e iluminação (algoritmo KMeans). Depois o SVM (resumo de 128 números com `StandardScaler`) foi treinado **sem** um grupo e medido nesse grupo, comparando com deixar de fora a mesma quantidade de fotos sorteadas.

| Grupo deixado de fora | Fotos | SVM sem o grupo | Sorteio do mesmo tamanho |
|---|---|---|---|
| 0 | 10.290 | 0,980 | 0,995 |
| 1 | 8.366 | 0,979 | 0,995 |
| 2 | 7.807 | 0,980 | 0,995 |
| **3** | 9.610 | **0,955** | 0,995 |
| 4 | 5.402 | 0,987 | 0,994 |
| **5** | 4.211 | **0,818** | 0,996 |

Em todos os grupos, cada letra continuou com mais de 990 exemplos no treino, então a queda não vem de faltar letra. No grupo 5, os erros mais comuns foram P lido como N (271), N lido como M (267), P lido como O (104) e C lido como O (88); no grupo 3, F lido como T (419). São trocas entre letras de formato parecido. Os grupos não são pessoas exatas, mas **tirar do treino o grupo 5 derrubou o acerto nesse grupo de 99,6% para 82%**. Os scripts dessa análise estão fora do repositório, junto com os arquivos do TCC.

### 5.3 Contar ou não as fotos sem mão

* **Só fotos com mão** (seção 4.1): mede o classificador e é a forma usada para os quatro modelos, então a comparação entre eles é justa.
* **Todas as fotos, "sem mão" como erro** (seção 4.2): mede o sistema inteiro, incluindo o detector, e tira cerca de 1,2 ponto do `.task`.

Ao comparar modelos, essas fotos precisam ser tratadas do mesmo jeito para todos.

### 5.4 RF, SVM e SNN não têm conjunto de validação

O `.task` separa 10% do treino para validação, mas RF, SVM e SNN não. A função de perda da SNN foi escolhida olhando o teste ([`treinar_modelos.py`, linhas 69 a 71](../../src/treino/treinar_modelos.py#L69-L71)), e as curvas da SNN em `graficos_modelos.py` medem o teste a cada época. Como nenhuma parada antecipada usa essas curvas, o efeito é pequeno, mas o teste deixa de ser totalmente inédito.

### 5.5 Espelhamento antes da divisão de validação do `.task`

No notebook do `.task`, as fotos de treino são espelhadas antes do sorteio entre treino e validação, então a versão espelhada de uma foto pode cair na validação enquanto a original fica no treino. Isso deixa a validação do Colab um pouco otimista, mas não afeta a pasta `test`, que nunca entra no treino.

---

## 6. Próximos passos

Em ordem de impacto:

1. **Teste com uma mão nova.** O autor grava cerca de 30 fotos por letra, com fundos e iluminações variados, numa pasta como `data/usuario_novo/`, que nunca entra no treino nem nas escolhas de configuração. O resultado passa a ter duas colunas: "teste do dataset (mesmas mãos)" e "usuário novo".
2. **Validação por grupo no repositório.** Trazer a análise da seção 5.2 para um script, por exemplo `src/avaliacao/validacao_grupos.py`, com `GroupKFold`, relatando a média e o pior grupo.
3. **Conjunto de validação separado do teste** para RF, SVM e SNN, usado para todas as escolhas (perda, épocas, `C` do SVM).

O andamento dessas tarefas fica no [backlog do projeto](../todo.md).
