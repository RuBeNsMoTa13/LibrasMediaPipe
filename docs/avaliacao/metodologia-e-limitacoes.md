<!--
Data/Hora: 2026-10-06 19:30 (UTC-3)
Branch: claude/auditoria-pipeline-3l8ukb
Commit: 3d6e7ce
Status: Novo
-->

# Metodologia da avaliação e limitações dos resultados

Este documento explica, para quem não é da área e para a banca do TCC, de onde vêm os dados, como eles chegam a cada modelo e o que as métricas podem e não podem afirmar. Ele resume uma auditoria técnica do pipeline feita em outubro de 2026, com números medidos sobre as fotos de `data/libras`.

## Resumo

1. **Os quatro modelos acertam entre 89% e 99% das fotos de teste, mas o teste tem as mesmas mãos do treino.** Esses números medem "as mesmas mãos, em outro momento da gravação", e não "uma pessoa nova usando o app".
2. **Em 2 dos 6 grupos de gravações parecidas, deixar o grupo fora do treino derrubou o acerto do SVM nesse grupo para 77% e 88%; nos outros 4 grupos, ele ficou entre 93% e 99%** (seção 5.2). É a estimativa mais próxima do uso ao vivo que esta base permite.
3. **O `.task` acerta menos (0,891) porque foi treinado de forma curta, e não porque foi avaliado de forma mais honesta.** Ele acerta quase a mesma fração nas próprias fotos de treino.
4. **O texto do TCC (versão de 18/09) descreve normalização no pulso, divisão 70/15/15 e uso da base V-LIBRASIL, mas o script que gerou a Tabela 1 não fez nenhuma dessas coisas** (seção 6). Nenhum script do repositório dividiu os dados em 70/15/15 ou usou a V-LIBRASIL. A normalização no pulso só foi usada depois, entre 4 e 5/10/2026, pelas primeiras versões de RF, SVM e SNN, e logo foi trocada pelo resumo de 128 números.

## Termos usados

* **Landmarks:** os 21 pontos que o MediaPipe marca na mão (pulso, juntas e pontas dos dedos).
* **Resumo de 128 números:** lista de 128 números que descreve o formato da mão, calculada por uma rede neural do Google (`gesture_embedder`) a partir dos landmarks. Na literatura, chama-se *embedding*.
* **Treino, validação e teste:** o modelo aprende com o treino; a validação serve para fazer escolhas durante o desenvolvimento; o teste mede o resultado final com fotos que o modelo nunca viu. Este projeto não tem validação.
* **Vazamento:** quando algo do teste já "apareceu" no treino de alguma forma, o que deixa a nota do teste mais alta do que seria com dados realmente novos.
* **Hiperparâmetros:** escolhas feitas antes do treino, como o número de épocas, o tamanho do lote ou o número de árvores.
* **Época:** uma passada completa do modelo por todas as fotos de treino.
* **Lote:** quantas fotos o modelo vê de uma vez antes de cada ajuste dos seus pesos.
* **Função de perda:** a conta que mede o quanto o modelo está errando durante o treino. O treino tenta deixar esse número cada vez menor.
* **Semente:** número que fixa os sorteios internos do treino (por exemplo, os pesos iniciais e a ordem das fotos), para que o resultado possa ser repetido. Treinar com sementes diferentes mostra quanto o resultado varia por acaso.
* **Dropout:** técnica que desliga, ao acaso, parte da rede durante o treino, para que ela não decore os exemplos.
* **Subajuste:** quando o modelo não aprende bem nem os exemplos que viu no treino.
* **KMeans:** algoritmo que separa itens em grupos parecidos sem saber de antemão quais são esses grupos.
* **GroupKFold:** forma de validação, do scikit-learn, que deixa um grupo inteiro de fora do treino de cada vez e mede o acerto nesse grupo.

---

## 1. De onde vêm os dados e como estão divididos

As fotos ficam em `data/libras/train/<letra>/` e `data/libras/test/<letra>/`, com 64 x 64 pixels cada. São 21 letras estáticas: A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W e Y.

| Pasta | Fotos | Por letra | Fotos em que o MediaPipe achou a mão |
|---|---|---|---|
| `data/libras/train` | 3.468 | 161 a 168 | 3.424 (98,7%) |
| `data/libras/test` | 1.153 | 45 a 58 | 1.135 (98,4%) |

**Como a divisão foi feita:**

* **Nenhum script do projeto divide os dados.** As pastas chegaram já separadas no commit `31a99a8` (02/06/2026), com um `.gitattributes` padrão do GitHub dentro (removido na limpeza de outubro), sinal de que foram baixadas prontas de outro lugar. A origem exata não está registrada.
* A proporção é de cerca de **75% para treino e 25% para teste** (3.468 e 1.153 de 4.621 fotos), sem validação.
* Os nomes dos arquivos são só números, **sem identificação de pessoa, vídeo ou sessão**. A numeração vai até cerca de 1.690 no treino e 570 no teste, o que indica que o repositório guarda uma parte (perto de 10%) de um conjunto maior que já vinha dividido.
* Na pasta `train/T` há 126 arquivos com nomes iniciados por `la`, `lb`, `lw` e `t`, que só existem no treino e parecem vir de outra fonte.

**Quantas pessoas aparecem.** Sem rótulo de pessoa não dá para contar com certeza, mas a inspeção visual mostra **pelo menos 4 a 5 mãos diferentes** (pele clara com anel, pele clara com pulseira, pele mais escura, mão com relógio e mão com manga escura). **As mesmas mãos, com os mesmos acessórios e o mesmo fundo, aparecem no treino e no teste.** A ideia de que a base tinha "uma pessoa só" está errada.

**O teste é um sorteio de fotos?** Não exatamente. Comparada a um pedaço do próprio treino separado por sorteio, a pasta `test` fica um pouco mais longe do treino, o que combina com outros trechos de gravação. Mesmo assim, o "vizinho mais próximo" (1-NN), que só copia a letra da foto de treino mais parecida, acerta **99,2%** do teste usando os landmarks. **A divisão parece ser por trecho de gravação, e não por pessoa.**

---

## 2. Como cada modelo recebe os dados

### 2.1 O caminho de uma imagem até a letra

1. **Imagem.** No treino, uma foto de `data/libras`. No app desktop, cada frame da webcam é espelhado antes de tudo ([`detectar_libras.py`, linha 293](../../src/desktop/detectar_libras.py#L293)).
2. **O MediaPipe acha a mão** e devolve, da primeira mão encontrada: os 21 landmarks na imagem (de 0 a 1), os mesmos 21 pontos em 3D (em metros, centrados na mão) e a probabilidade de ser a mão direita. **Fotos sem mão encontrada ficam fora do treino e do teste** ([`resumo.py`, linhas 104 e 105](../../src/classificadores/resumo.py#L104-L105)). Tudo fica guardado em `results/tables/resumo_libras.npz`.
3. **Resumo de 128 números.** Os landmarks passam pela rede `gesture_embedder.tflite`, lida de dentro do próprio `.task` ([`resumo.py`, linhas 33 a 65](../../src/classificadores/resumo.py#L33-L65)). **É essa rede do Google que ajusta posição e tamanho da mão.** Hoje o projeto não normaliza os landmarks por conta própria em nenhum classificador, e não é possível inspecionar como a rede faz isso. As primeiras versões de RF, SVM e SNN, de 4 e 5/10/2026, centralizavam a mão no pulso antes de classificar (seção 6).
4. **Classificação**, de dois jeitos: **(a)** o próprio `.task`, que passa o resumo pela pequena camada treinada no Colab (seção 3); ou **(b)** Random Forest, SVM ou SNN, que recebem exatamente o mesmo resumo. Os quatro modelos partem, portanto, da mesma base.
5. **No app,** a letra só é aceita se a confiança passar do limiar (50% por padrão, ajustável com `+` e `-`), com no mínimo 0,7 s entre letras, e a palavra pode ser falada em voz alta. A versão web usa só o `.task`, também com limiar padrão de 50%.

O app ao vivo usa a mesma função de resumo do treino ([`resumo.py`, linhas 68 a 72](../../src/classificadores/resumo.py#L68-L72)), então treino e uso real recebem os dados do mesmo jeito. Como o app espelha o frame e a pessoa pode usar a outra mão, cada mão de treino entra duas vezes, como está e espelhada ([`treinar_modelos.py`, linhas 111 a 113](../../src/treino/treinar_modelos.py#L111-L113)): o treino passa de 3.424 para 6.848 linhas. O teste não é espelhado.

### 2.2 Hiperparâmetros e normalizadores

| Modelo | Normalizador | Configuração | Onde |
|---|---|---|---|
| Random Forest | Nenhum (árvores de decisão não precisam) | 100 árvores, semente 42; o resto no padrão do scikit-learn (árvores sem limite de profundidade) | [`treinar_modelos.py`, linha 136](../../src/treino/treinar_modelos.py#L136) |
| SVM | `StandardScaler`, que deixa cada número com média 0 e desvio 1, calculado só no treino | Kernel RBF, `C=1` e `gamma='scale'` (padrões do scikit-learn), `probability=True` para o app mostrar a confiança | [`treinar_modelos.py`, linhas 139 e 140](../../src/treino/treinar_modelos.py#L139-L140) |
| SNN (snnTorch) | Padronização à mão com média e desvio do treino, guardados em `snn_libras.pt` | Entrada com os 128 números do resumo, duas camadas escondidas de 128 neurônios LIF e uma camada de saída com 21 neurônios LIF, uma por letra (decaimento 0,9); 25 passos de tempo; Adam com taxa 0,002; 50 épocas; lotes de 64; perda `mse_count_loss` (80% de disparos na letra certa, 10% nas outras) | [`treinar_modelos.py`, linhas 57 a 90 e 118 a 120](../../src/treino/treinar_modelos.py#L57-L90); [`snn.py`, linhas 26 a 53](../../src/classificadores/snn.py#L26-L53) |

Na SNN (rede neural de impulsos), cada neurônio "dispara" ou não a cada passo de tempo, como um neurônio biológico. O neurônio LIF (do inglês *leaky integrate-and-fire*) acumula o sinal que recebe, perde aos poucos parte do que acumulou (o decaimento) e dispara quando passa de um limite. A letra escolhida é a do neurônio de saída que disparou mais vezes nos 25 passos.

**Os dois normalizadores são ajustados só com o treino, então não há vazamento no pré-processamento.** O vazamento está na divisão dos dados (seção 5).

---

## 3. Como o `.task` foi treinado

O `.task` foi treinado no Google Colab com o MediaPipe Model Maker. **O notebook e a configuração não estão no repositório.** O que se sabe vem de abrir o arquivo e do log do treino.

**Confirmado:**

* O `.task` é um arquivo compactado com o detector de mão, a rede `gesture_embedder.tflite` (fixa, do Google), um classificador de 8 gestos padrão do Google (não usado) e o `custom_gesture_classifier.tflite`, **a única parte treinada no Colab**.
* Essa parte é pequena: `BatchNormalization`, depois `ReLU`, depois uma camada `Dense` com 22 saídas. É o que o Model Maker monta com `ModelOptions(layer_widths=[])`, ou seja, **sem camadas ocultas**.
* As 22 saídas são as 21 letras mais `none` ("nenhum gesto"). Como `data/libras` não tem pasta `none`, **o Colab não usou exatamente a base deste repositório**. Em nenhuma foto do repositório o `.task` respondeu `none`.
* O log copiado à mão em [`curvas_treino_task.py`, linhas 18 a 27](../../src/avaliacao/curvas_treino_task.py#L18-L27) mostra 10 épocas, com acurácia final de 0,791 no treino e 0,863 na validação do Colab. Esse script só desenha as Figuras 1 e 2 do TCC.

**Provável, mas não confirmado:** o log mostra só as 10 épocas. Elas batem com o padrão do Model Maker (taxa de aprendizado 0,001, lotes de 2 fotos, 10 épocas), então é provável que o lote de 2 fotos e a taxa de 0,001 também tenham sido usados, mas esses dois valores não aparecem no log. No tutorial padrão, o Model Maker separa treino, validação e teste por sorteio de fotos, então a validação do Colab teria o mesmo vazamento de pessoa da seção 5.

**Não se sabe:** quais fotos foram usadas, como foram divididas e o valor de *dropout*, que não fica salvo no arquivo.

---

## 4. Resultados e como lê-los

### 4.1 Tabela principal

Os quatro modelos foram medidos nas **mesmas 1.135 fotos de teste** em que o MediaPipe achou a mão. Os números são impressos por `python src/treino/treinar_modelos.py`, que também treina e salva RF, SVM e SNN em `models/`. A linha do `.task` usa a letra que ele mesmo deu a cada foto ([`treinar_modelos.py`, linhas 148 e 149](../../src/treino/treinar_modelos.py#L148-L149)).

| Modelo | Acurácia | Precisão (ponderada) | Recall (ponderado) | F1 (ponderado) |
|---|---|---|---|---|
| MediaPipe `.task` | 0,891 | 0,931 | 0,891 | 0,863 |
| Random Forest | 0,994 | 0,994 | 0,994 | 0,994 |
| SVM | 0,982 | 0,986 | 0,982 | 0,983 |
| SNN | 0,990 | 0,991 | 0,990 | 0,990 |

* **Acurácia:** fração das fotos com a letra certa (0,891 é 89,1% de acertos).
* **Precisão:** quando o modelo diz "é a letra X", quantas vezes é mesmo X.
* **Recall:** das fotos da letra X, quantas o modelo reconheceu.
* **F1:** junta precisão e recall num número só; só é alto se os dois forem.
* **Ponderada:** calculada letra por letra e depois tirada a média, com mais peso para as letras com mais fotos.

Como precisão, recall e F1 são calculados letra por letra e só depois resumidos numa média ponderada, o F1 da tabela não é o F1 da precisão e do recall gerais e pode ficar abaixo dos dois. É o que acontece com o `.task` (F1 de 0,863, abaixo de 0,931 e 0,891). Nas letras N e T, por exemplo, quase toda resposta "N" ou "T" do `.task` está certa (precisão 1,0), mas ele reconhece menos de 10% das fotos dessas letras (recall de 0,04 e 0,09). O F1 dessas duas letras fica perto de zero e puxa a média para baixo.

Os gráficos (comparação, F1 por letra, matrizes de RF, SVM e SNN, curvas da SNN, distribuição de fotos e detecção da mão) são gerados por `python src/avaliacao/graficos_modelos.py` em `results/figures/`.

### 4.2 Contando as fotos sem mão como erro

`python src/avaliacao/matriz_confusao_task.py` passa **todas as 1.153 fotos** pelo `.task` e conta como erro ("Nenhum") as 18 em que a mão não foi encontrada ([linhas 72 a 78](../../src/avaliacao/matriz_confusao_task.py#L72-L78)). Assim o `.task` fica com **0,877 de acurácia e F1 de 0,856**. Esse script gera a matriz da Figura 3 do TCC (`results/figures/matriz_confusao_task.png`).

### 4.3 O que estes números dizem

* **Dizem** que, para as mesmas mãos e fundos da base, os quatro modelos separam bem as 21 letras, e que RF, SVM e SNN aproveitam melhor o resumo de 128 números do que a camada pequena do `.task`.
* **Não dizem** quanto o sistema acerta com uma pessoa nova na frente da webcam (seção 5). Um sinal disso: ao vivo, com uma mão nova, os modelos ainda confundem letras parecidas, e U com R foi o caso relatado. No teste, essa confusão quase não aparece para RF, SVM e SNN (um único caso, no SVM).

---

## 5. Limitações

### 5.1 Vazamento de pessoa e de sessão

O teste tem as mesmas mãos, acessórios e fundo do treino (seção 1). É como uma prova com os mesmos exercícios da aula, mudando alguns números: dá para ir bem sem ter aprendido a matéria. Sem rótulo de pessoa, não é possível separar treino e teste por pessoa com o que está no repositório. Trocar a entrada dos modelos pelo resumo de 128 números não resolveu isso, porque o problema está na divisão dos dados, e não na forma de descrever a mão.

### 5.2 Validação por grupo de gravação

Para estimar o efeito do vazamento, a auditoria agrupou as 4.559 fotos com mão em 6 grupos de gravações parecidas, pela cor da pele, cor do fundo e iluminação (algoritmo KMeans). Depois treinou o SVM (resumo de 128 números com `StandardScaler`) **sem** um grupo e mediu o acerto nesse grupo, comparando com deixar de fora a mesma quantidade de fotos sorteadas.

| Grupo deixado de fora | Fotos | SVM sem o grupo | Sorteio do mesmo tamanho |
|---|---|---|---|
| 0 | 1.043 | 0,986 | 0,964 |
| 1 | 761 | 0,961 | 0,955 |
| **2** | 429 | **0,772** | 0,974 |
| **3** | 558 | **0,884** | 0,966 |
| 4 | 861 | 0,931 | 0,970 |
| 5 | 907 | 0,958 | 0,955 |

Nos grupos 2 e 3, todas as letras continuaram com mais de 100 exemplos no treino, então a queda não vem de faltar letra. Os erros mais comuns foram N lido como M, P lido como N e C lido como O, trocas entre letras de formato parecido. Os grupos não são pessoas exatas, mas **tirar do treino o grupo 2 ou o grupo 3 derrubou o acerto nesse grupo de cerca de 97% para 77% e 88%**. Nos outros 4 grupos, o SVM ficou entre 93% e 99%, perto do valor do sorteio (nos grupos 0, 1 e 5, até um pouco acima). Ou seja, o efeito depende de quanto as gravações deixadas de fora diferem das que ficaram no treino. Os scripts dessa análise ainda não fazem parte do repositório (seção 7).

### 5.3 O `.task` está subtreinado

| Modelo | Acerto no treino | Acerto no teste |
|---|---|---|
| `.task`, contando foto sem mão como erro | 0,884 | 0,877 |
| `.task`, só fotos com mão | 0,895 | 0,891 |
| Mesma camada, treinada de novo com o padrão do Model Maker (lotes de 2, 10 épocas), 3 sementes | 0,85 a 0,89 | 0,86 a 0,91 |
| Mesma camada, lotes de 32 e 200 épocas | 0,997 | **0,993** |

**O `.task` acerta o treino quase tão pouco quanto o teste.** Isso é subajuste: ele não aprendeu nem os exemplos que viu (se fosse só "mais honesto", acertaria muito no treino e menos no teste). O treino padrão do Model Maker, que é o mais provável aqui (seção 3), é curto e instável (lotes de 2 fotos, 10 épocas); com treino mais longo, a mesma camada chega a 99%. **O 0,89 do `.task` não é "mais realista", é um teto da configuração do Colab**, e os quatro números da tabela estão inflados pelo mesmo teste com vazamento.

### 5.4 Contar ou não as fotos sem mão

* **Só fotos com mão** (seção 4.1): mede o classificador e é a forma usada para os quatro modelos, então a comparação entre eles é justa.
* **Todas as fotos, "sem mão" como erro** (seção 4.2): mede o sistema inteiro, incluindo o detector, e tira cerca de 1,4 ponto do `.task`.

Na Tabela 1 do TCC, só a linha do `.task` contava "sem mão" como erro. Ao comparar modelos, essas fotos precisam ser tratadas do mesmo jeito para todos.

### 5.5 Não há conjunto de validação

A troca da função de perda da SNN foi decidida olhando o teste ([`treinar_modelos.py`, linhas 60 a 62](../../src/treino/treinar_modelos.py#L60-L62)), e as curvas da SNN medem o teste a cada época ([`graficos_modelos.py`, linhas 107 a 112](../../src/avaliacao/graficos_modelos.py#L107-L112)). Como nenhuma parada antecipada usa essas curvas, o efeito é pequeno, mas o teste deixa de ser totalmente inédito.

---

## 6. Diferenças entre o texto do TCC (versão de 18/09) e o código

| O que o texto diz | O que o código faz |
|---|---|
| **Dados:** abordagem híbrida, com a base V-LIBRASIL e coleta própria pela webcam. | Só existe `data/libras`, que chegou pronta e dividida. Não há no repositório arquivos da V-LIBRASIL nem de coleta própria. |
| **Normalização:** pulso (landmark 0) como origem. | A Tabela 1 saiu de `comparar_modelos.py` (removido em outubro), que usava as 63 coordenadas cruas da tela, sem pulso, sem escala e sem `StandardScaler`. A normalização no pulso existiu em `landmarks.py` (também removido): foi usada pelas primeiras versões de RF, SVM e SNN, no treino e no app, entre 4 e 5/10/2026, e depois foi trocada pelo resumo de 128 números. Ela nunca foi usada no script da Tabela 1. Hoje a normalização acontece dentro da rede do Google (seção 2). |
| **Divisão:** estratificada em 70% treino, 15% validação e 15% teste. | Nenhum script divide os dados; as pastas vieram com cerca de 75% e 25%, sem validação. |
| **Materiais:** TensorFlow/Keras e Pandas. | O código do repositório usa scikit-learn, PyTorch e snnTorch, sem Pandas nem TensorFlow. O TensorFlow só aparece no Colab, dentro do Model Maker. |
| **Modelagem:** classificadores clássicos e "rede neural profunda". | A parte treinada do `.task` é uma única camada sobre o resumo de 128 números; a rede maior é do Google e não foi treinada no projeto. A SNN do código não aparece no texto. |
| **Tabela 1:** `.task` 0,88, SVM 0,92 e RF 0,97, medidos "sobre o conjunto de testes com imagens inéditas". | A auditoria reproduziu 0,877, 0,920 e 0,966. O 0,92 do SVM vem da falta de `StandardScaler` (com ele, 0,985). Só o `.task` contava "sem mão" como erro. As fotos são inéditas, as pessoas não. Os números atuais estão na seção 4.1. |
| **Figuras 1 e 2:** a "proximidade entre os índices de validação e treinamento atesta robusta capacidade de generalização e ausência de sobreajuste". | O treino (0,791) ficou abaixo da validação (0,863) e o `.task` acerta só 0,895 das próprias fotos de treino: o sinal é de subajuste (seção 5.3), e a validação do Colab provavelmente tem as mesmas mãos. |

Hoje as Figuras 1 e 2 vêm de `src/avaliacao/curvas_treino_task.py` (`task_curva_acuracia.png` e `task_curva_perda.png`), a Figura 3 vem de `src/avaliacao/matriz_confusao_task.py` (`matriz_confusao_task.png`) e a tabela com os quatro modelos é impressa por `src/treino/treinar_modelos.py`.

---

## 7. Próximos passos

Em ordem de impacto:

1. **Teste com uma mão nova.** O autor grava cerca de 30 fotos por letra, com fundos e iluminações variados, numa pasta como `data/usuario_novo/`, que nunca entra no treino nem nas escolhas de configuração. O TCC passa a mostrar duas colunas: "teste do dataset (mesmas mãos)" e "usuário novo".
2. **Validação por grupo.** Verificar se o conjunto original completo identifica pessoa ou vídeo; se não, usar os grupos aproximados da seção 5.2 com `GroupKFold` num script novo (por exemplo, `src/avaliacao/validacao_grupos.py`), relatando a média e o pior grupo.
3. **Conjunto de validação separado do teste**, usado para todas as escolhas (perda, épocas, `C` do SVM).
4. **Retreinar o `.task`** com lotes maiores (por exemplo, 32) e mais épocas, guardando o notebook e os hiperparâmetros no repositório, e avaliar os quatro modelos no mesmo conjunto, tratando "sem mão" igual para todos.
5. **Corrigir o texto do TCC** conforme a seção 6, ou implementar e rodar o pipeline que o texto descreve.

O andamento dessas tarefas fica no [backlog do projeto](../todo.md).
