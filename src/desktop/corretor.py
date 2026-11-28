"""Corretor automático das palavras soletradas (o mesmo algoritmo da versão web).

Quando a pessoa termina uma palavra (espaço), o corretor procura na lista de palavras do
português (web/palavras_pt.txt) a mais parecida. Ele não troca sozinho: o HUD mostra a
sugestão e a pessoa aceita (Tab ou botão Aceitar) ou ignora.

A comparação ignora acentos, então "nao" vira "não". Também acha palavras com letras que
têm movimento e ficam de fora do alfabeto reconhecido (H, J, K, X, Z), como "oje" -> "hoje".
"""
import math
import unicodedata
from pathlib import Path

WORDS_PATH = Path(__file__).resolve().parent.parent.parent / "web" / "palavras_pt.txt"


def normalize(word):
    """Minúsculas e sem acento (ç vira c)."""
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


# Letras com movimento: ficam de fora do alfabeto reconhecido, então quem soletra não
# consegue escrevê-las. Faltar uma delas custa só meia edição ("oe" -> "hoje").
MISSING_LETTERS = set("hjkxz")


def distance(a, b, limit):
    """Distância de edição de a até b (inserir, apagar, trocar ou inverter duas letras vizinhas).
    Para de calcular e devolve limit + 1 quando passa do limite."""
    prev2 = None
    prev = [0]
    for ch in b:
        prev.append(prev[-1] + (0.5 if ch in MISSING_LETTERS else 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            insert = 0.5 if b[j - 1] in MISSING_LETTERS else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + insert, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
            best = min(best, cur[j])
        if best > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class Corretor:
    def __init__(self, path=WORDS_PATH):
        self.words = {}  # palavra sem acento -> (grafia certa, posição na lista: menor = mais comum)
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as e:
            print("Aviso: corretor automático sem lista de palavras:", e)
            lines = []
        for line in lines:
            word = line.strip()
            if word and not word.startswith("#"):
                self.words.setdefault(normalize(word), (word, len(self.words)))

    def suggest(self, word):
        """Grafia sugerida para a palavra, ou None se ela já está certa ou não há nada parecido."""
        key = normalize(word)
        if len(key) < 2 or not self.words:
            return None
        if key in self.words:
            right = self.words[key][0]
            return right if right != word.lower() else None  # só falta o acento
        limit = 1 if len(key) <= 4 else 2
        best = None
        for other, (right, rank) in self.words.items():
            if len(key) - len(other) > limit or len(other) - len(key) > 2 * limit:
                continue
            d = distance(key, other, limit)
            # menos edições primeiro; entre as parecidas, a mais comum, com um pouco de preferência
            # pelo mesmo tamanho (letras trocadas de lugar)
            score = d + 0.12 * math.log10(rank + 1) + 0.1 * abs(len(other) - len(key))
            if d <= limit and (best is None or score < best[0]):
                best = (score, right)
        return best[1] if best else None
