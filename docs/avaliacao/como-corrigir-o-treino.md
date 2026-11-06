# Como corrigir o treino e a avaliação dos modelos

Este guia explica por que as matrizes de confusão do Random Forest, do SVM e da
SNN saem quase perfeitas, por que a do MediaPipe (`.task`) sai pior, e o que
fazer em cada caso. Os números vêm da auditoria de 05/10/2026
(`docs/avaliacao/metodologia-e-limitacoes.md`) e de uma nova execução de
`src/avaliacao/graficos_modelos.py` em 07/10/2026.

| Modelo | Acurácia no teste atual | Situação |
|---|---|---|
| MediaPipe `.task` (rede densa do Model Maker) | 0,891 | subtreinado |
| MediaPipe + Random Forest | 0,994 | teste fácil demais |
| MediaPipe + SVM (RBF) | 0,982 | teste fácil demais |
| MediaPipe + SNN (LIF) | 0,990 | teste fácil demais |

## 1. O problema não está no código de treino do RF, do SVM e da SNN

O código treina esses três modelos do jeito certo: treina só com
`data/libras/train`, padroniza com a média e o desvio do treino e testa em
`data/libras/test`. As curvas da SNN mostram isso: a perda cai, e a acurácia de
treino (0,997) e a de teste (0,990) ficam juntas, sem sinal de overfitting.

A matriz sai quase perfeita porque **o conjunto de teste é parecido demais com
o de treino**:

- As pastas `train/` e `test/` vieram prontas. Nenhum script do repositório
  faz a divisão.
- Há pelo menos 4 ou 5 mãos diferentes nas fotos (anel, pulseira, relógio, tons
  de pele) e **as mesmas mãos aparecem no treino e no teste**.
- Um classificador que só procura a foto de treino mais parecida (1-NN) já
  acerta 99,2% do teste. Ou seja, o teste mede se o modelo reconhece **as mesmas
  pessoas, no mesmo fundo e na mesma luz**, e não uma pessoa nova.
- Os 128 números que os três modelos recebem vêm do `gesture_embedder`, uma rede
  que o Google já treinou com muitas mãos. Com uma entrada tão boa, separar 21
  letras dessas mesmas pessoas é fácil.

Quando a auditoria tirou um grupo de gravações do treino (agrupadas por cor de
pele, fundo e iluminação) e testou só nele, o SVM caiu para **77% a 88%**, com
erros N→M, P→N e C→O. Esse é o número mais perto do que acontece com um usuário
novo na webcam.

**Conclusão:** não adianta mexer nos hiperparâmetros do RF, do SVM ou da SNN
para "piorar" a matriz. O que precisa mudar é **como o teste é montado**.

## 2. O que fazer, em ordem

### Passo 1. Separar validação de teste

Hoje `src/avaliacao/graficos_modelos.py` mede a acurácia da SNN no teste a cada
época (curva `snn_curva_acuracia.png`). Nenhuma decisão é tomada com isso, mas
no TCC a curva deve se chamar "validação", e a validação não pode ser o teste.

Tire uns 15% do treino para validação, estratificado por letra, e use o teste
só uma vez, no fim:

```python
from sklearn.model_selection import train_test_split

idx = np.arange(len(y_treino_i))
idx_tr, idx_val = train_test_split(idx, test_size=0.15,
                                   stratify=y_treino_i, random_state=SEED)
# espelhe só idx_tr para treinar; use idx_val na curva da SNN
# e para escolher o número de épocas; o teste fica para o fim
```

### Passo 2. Avaliar por grupo de gravação (validação cruzada por grupo)

Como não existe rótulo de pessoa, crie grupos pelas fotos: a cor mediana da
pele dentro do contorno da mão e a cor do fundo, agrupadas com KMeans (4 a 6
grupos). O script `auditoria/scripts/grupos.py` nos arquivos do projeto faz
exatamente isso. Depois junte treino e teste e use `LeaveOneGroupOut`:

```python
from sklearn.model_selection import LeaveOneGroupOut

X = np.concatenate([treino["resumo"], teste["resumo"]])
y = np.concatenate([treino["y"], teste["y"]])
# grupos: um número por foto, vindo do KMeans de cor de pele e fundo

for tr, te in LeaveOneGroupOut().split(X, y, grupos):
    modelo.fit(X[tr], y[tr])          # espelhe só X[tr], nunca o grupo de teste
    print(accuracy_score(y[te], modelo.predict(X[te])))
```

Apresente no TCC a média e o desvio entre os grupos. As matrizes de confusão
desse teste vão mostrar os erros reais (N↔M, P↔N, C↔O, U↔R).

### Passo 3. Testar com uma pessoa que não está no dataset

É a prova mais forte. Fotografe uma pessoa nova (você ou um colega), cerca de
30 fotos por letra, em outro fundo, e use essas fotos **só como teste**, nunca
no treino. Assim o TCC pode dizer "acurácia com usuário novo" com segurança.

### Passo 4. Retreinar o `.task` do MediaPipe (este sim está mal treinado)

O `.task` erra até as fotos de treino (acerta só 0,895 no treino), então o
problema dele é treino curto, não teste honesto. A cabeça dele foi treinada com
o padrão do Model Maker (lotes de 2 fotos, 10 épocas). Com lotes de 32 e 200
épocas, a mesma cabeça chega a 0,99, igual aos outros modelos.

No Colab:

```python
from mediapipe_model_maker import gesture_recognizer

dados = gesture_recognizer.Dataset.from_folder(
    dirname="data/libras/train",
    hparams=gesture_recognizer.HandDataPreprocessingParams())
treino, validacao = dados.split(0.85)

hparams = gesture_recognizer.HParams(export_dir="exported_model",
                                     batch_size=32, epochs=200,
                                     learning_rate=0.001)
opcoes = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
modelo = gesture_recognizer.GestureRecognizer.create(
    train_data=treino, validation_data=validacao, options=opcoes)
modelo.export_model()   # gera gesture_recognizer.task
```

Observações:

- O Model Maker exige uma pasta `none` (fotos sem letra nenhuma). Crie
  `data/libras/train/none` com algumas fotos de mão parada ou de fundo.
- Salve esse código no repositório (por exemplo `src/treino/treinar_task.py`),
  porque hoje a configuração do Colab não está em lugar nenhum.
- Avalie o `.task` com a mesma regra dos outros modelos: ou todos descartam as
  fotos sem mão, ou todos contam como erro. Hoje
  `src/avaliacao/matriz_confusao_task.py` conta como erro (0,877) e
  `graficos_modelos.py` descarta (0,891).

### Passo 5. Ajustar o texto do TCC

- Diga que o teste tem as mesmas pessoas do treino e mostre os dois números: o
  do teste atual e o do teste por grupo ou com pessoa nova.
- Tire a frase sobre normalização no pulso e a divisão 70/15/15: o código não
  faz isso. Quem normaliza posição e tamanho é o `gesture_embedder` do Google.

## 3. Resumo

| O que | Onde mexer | Muda a matriz? |
|---|---|---|
| Validação separada do teste | `graficos_modelos.py`, `treinar_modelos.py` | Não, mas corrige a curva da SNN |
| Teste por grupo de gravação | novo script de avaliação | Sim, aparecem os erros reais |
| Teste com pessoa nova | fotos novas em uma pasta só de teste | Sim, é o número mais honesto |
| Retreinar o `.task` (batch 32, 200 épocas) | Colab, salvo em `src/treino/` | Sim, o `.task` sobe para ~0,99 |
| Mesma regra para "sem mão" | `matriz_confusao_task.py` | Pouco (cerca de 1,4 ponto) |
