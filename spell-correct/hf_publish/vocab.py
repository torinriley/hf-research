"""Tiny fixed character vocabulary: a-z plus special tokens."""
import string

SPECIALS = ["<pad>", "<bos>", "<eos>"]
CHARS = list(string.ascii_lowercase)
ITOS = SPECIALS + CHARS
STOI = {c: i for i, c in enumerate(ITOS)}

PAD, BOS, EOS = STOI["<pad>"], STOI["<bos>"], STOI["<eos>"]
VOCAB_SIZE = len(ITOS)


def encode(word: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    ids = [STOI[c] for c in word]
    if add_bos:
        ids = [BOS] + ids
    if add_eos:
        ids = ids + [EOS]
    return ids


def decode(ids: list[int]) -> str:
    chars = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        chars.append(ITOS[i])
    return "".join(chars)
