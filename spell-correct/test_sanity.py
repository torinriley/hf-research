"""Sanity checks for the pieces evaluation correctness depends on."""
import torch

from baseline import NearestNeighborCorrector, levenshtein
from model import ModelConfig, SpellCorrectorTransformer
from vocab import decode, encode


def test_levenshtein_known_values():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("flaw", "lawn") == 2


def test_vocab_roundtrip():
    for word in ["hello", "a", "antidisestablishmentarianism", "zzz"]:
        ids = encode(word)
        assert decode(ids) == word


def test_vocab_decode_stops_at_eos():
    from vocab import BOS, EOS, PAD

    # bos, h, e, y, eos, pad, pad -> should decode to "hey", ignoring padding after eos
    ids = [BOS, encode("hey", add_bos=False, add_eos=False)[0], encode("hey", add_bos=False, add_eos=False)[1],
           encode("hey", add_bos=False, add_eos=False)[2], EOS, PAD, PAD]
    assert decode(ids) == "hey"


def test_nearest_neighbor_exact_hit():
    vocab = ["cat", "dog", "catalog", "catastrophe"]
    corrector = NearestNeighborCorrector(vocab)
    assert corrector.correct("cat") == "cat"


def test_nearest_neighbor_finds_close_typo():
    vocab = ["accentuation", "accensor", "abjudge"]
    corrector = NearestNeighborCorrector(vocab)
    assert corrector.correct("accetnuation") == "accentuation"


def test_model_forward_shapes():
    cfg = ModelConfig(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, max_len=16)
    model = SpellCorrectorTransformer(cfg)
    batch, src_len, tgt_len = 3, 5, 4
    src = torch.randint(3, cfg.vocab_size, (batch, src_len))
    tgt = torch.randint(3, cfg.vocab_size, (batch, tgt_len))
    src_mask = torch.zeros(batch, src_len, dtype=torch.bool)
    tgt_mask = torch.zeros(batch, tgt_len, dtype=torch.bool)
    out = model(src, tgt, src_mask, tgt_mask)
    assert out.shape == (batch, tgt_len, cfg.vocab_size)


def test_model_forward_is_deterministic_in_eval_mode():
    cfg = ModelConfig(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, max_len=16)
    model = SpellCorrectorTransformer(cfg)
    model.eval()
    src = torch.randint(3, cfg.vocab_size, (2, 5))
    tgt = torch.randint(3, cfg.vocab_size, (2, 4))
    src_mask = torch.zeros(2, 5, dtype=torch.bool)
    tgt_mask = torch.zeros(2, 4, dtype=torch.bool)
    with torch.no_grad():
        out1 = model(src, tgt, src_mask, tgt_mask)
        out2 = model(src, tgt, src_mask, tgt_mask)
    assert torch.allclose(out1, out2)
