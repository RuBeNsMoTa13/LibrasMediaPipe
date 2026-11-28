"""Corretor automático das palavras soletradas (o mesmo algoritmo da versão web).

Quando a pessoa termina uma palavra (espaço), o corretor procura na lista de palavras do
português (web/palavras_pt.txt) a mais parecida. Ele não troca sozinho: o HUD mostra a
sugestão (até 3 opções) e a pessoa escolhe (Tab, 1, 2, 3 ou clique) ou ignora. A palavra
anterior também conta: depois de "tudo", "bem" passa na frente (web/pares_pt.txt).

A comparação ignora acentos, então "nao" vira "não". Também acha palavras com letras que
têm movimento e ficam de fora do alfabeto reconhecido (H, J, K, X, Z), como "oje" -> "hoje".
"""
import math
import unicodedata
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
WORDS_PATH = WEB_DIR / "palavras_pt.txt"
PAIRS_PATH = WEB_DIR / "pares_pt.txt"
MAX_SUGGESTIONS = 3
PAIR_BONUS = 1.0  # quanto a palavra anterior ajuda: "tudo" + "bmel" prefere "bem" a "mel"


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


def _read_lines(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        print("Aviso: corretor automático sem o arquivo", path, e)
        return []
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


class Corretor:
    def __init__(self, path=WORDS_PATH, pairs_path=PAIRS_PATH):
        self.words = {}  # palavra sem acento -> (grafia certa, posição na lista: menor = mais comum)
        for word in _read_lines(path):
            self.words.setdefault(normalize(word), (word, len(self.words)))
        # pares comuns, sem acento: "tudo bem" -> {("tudo", "bem")}
        self.pairs = {tuple(normalize(p).split()[:2]) for p in _read_lines(pairs_path) if " " in p}

    def suggest(self, word, previous=""):
        """Até 3 grafias sugeridas, a melhor primeiro. Lista vazia se a palavra já está certa
        ou não há nada parecido. previous é a palavra escrita antes dela, se houver."""
        key = normalize(word)
        if len(key) < 2 or not self.words:
            return []
        prev = normalize(previous)
        if key in self.words:
            # palavra existe: só sugere o acento ou um par comum bem parecido ("bom dai" -> "dia")
            right = self.words[key][0]
            out = [right] if right != word.lower() else []
            for first, second in self.pairs:
                if first == prev and second != key and distance(key, second, 1) <= 1:
                    out.insert(0, self.words.get(second, (second,))[0])
            return out[:MAX_SUGGESTIONS]
        limit = 1 if len(key) <= 3 else 2
        found = []
        for other, (right, rank) in self.words.items():
            if len(key) - len(other) > limit or len(other) - len(key) > 2 * limit:
                continue
            d = distance(key, other, limit)
            if d > limit:
                continue
            # menos edições primeiro; entre as parecidas, a mais comum, com um pouco de preferência
            # pelo mesmo tamanho (letras trocadas de lugar) e pelo par com a palavra anterior
            score = d + 0.12 * math.log10(rank + 1) + 0.1 * abs(len(other) - len(key))
            if (prev, other) in self.pairs:
                score -= PAIR_BONUS
            found.append((score, right))
        found.sort()
        return [right for _, right in found[:MAX_SUGGESTIONS]]
