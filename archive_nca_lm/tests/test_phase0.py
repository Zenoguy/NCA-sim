"""
Unit Test Suite for Phase 0: Groundwork, Tokenizer, Dataset, & Baselines.
"""

import math
import numpy as np
import pytest
import torch

from data.dataset import AutoregressiveDataset, get_dataloader
from eval.perplexity import loss_to_perplexity, compute_discrete_nll
from models.ngram import NGramLanguageModel


def test_loss_to_perplexity():
    assert loss_to_perplexity(0.0) == 1.0
    assert abs(loss_to_perplexity(math.log(100.0)) - 100.0) < 1e-4
    assert loss_to_perplexity(float('nan')) == float('inf')
    assert loss_to_perplexity(float('inf')) == float('inf')
    assert loss_to_perplexity(100.0) == math.exp(50.0)  # clamped guard


def test_discrete_nll_uniform():
    # A uniform distribution over V=100 has log P = log(1/100) = -log(100)
    V = 100
    log_probs = np.full(1000, -math.log(V))
    metrics = compute_discrete_nll(log_probs)
    assert abs(metrics["loss"] - math.log(V)) < 1e-4
    assert abs(metrics["perplexity"] - V) < 1e-2
    assert metrics["total_tokens"] == 1000


def test_autoregressive_dataset():
    # Create synthetic token sequence: 0, 1, 2, ..., 999
    tokens = np.arange(1000, dtype=np.int64)
    seq_len = 128
    dataset = AutoregressiveDataset(tokens, seq_len=seq_len, stride=seq_len)
    
    # Verify sample count: (1000 - 129) // 128 + 1 = 871 // 128 + 1 = 7
    expected_samples = (1000 - (seq_len + 1)) // seq_len + 1
    assert len(dataset) == expected_samples

    # Check shapes
    inputs, targets = dataset[0]
    assert inputs.shape == (seq_len,)
    assert targets.shape == (seq_len,)

    # Check causal shift alignment: target is input shifted by 1
    assert torch.equal(inputs, torch.arange(0, 128))
    assert torch.equal(targets, torch.arange(1, 129))
    assert torch.equal(inputs[1:], targets[:-1])
    
    # Run built-in assertion
    assert dataset.verify_causal_alignment()


def test_dataloader_batching():
    tokens = np.arange(500, dtype=np.int64)
    loader = get_dataloader(tokens, seq_len=64, batch_size=4, shuffle=False)
    batch_in, batch_tgt = next(iter(loader))
    assert batch_in.shape == (4, 64)
    assert batch_tgt.shape == (4, 64)
    assert torch.equal(batch_in[:, 1:], batch_tgt[:, :-1])


def test_ngram_model_normalization():
    # Toy corpus with repetitive structure
    tokens = [1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 5, 2, 3, 1] * 20
    vocab_size = 10
    model = NGramLanguageModel(n=3, discount=0.75, vocab_size=vocab_size)
    model.fit(tokens)

    # Test that probabilities for all words in vocabulary sum to ~1.0
    test_contexts = [
        (),
        (1,),
        (1, 2),
        (2, 3),
        (9, 9),  # unseen context
    ]

    for ctx in test_contexts:
        total_prob = sum(model.score(ctx, w) for w in range(vocab_size))
        assert abs(total_prob - 1.0) < 0.05, f"Context {ctx} total prob was {total_prob} (expected ~1.0)"


def test_ngram_evaluation():
    train_tokens = [1, 2, 3, 4, 5] * 50
    test_tokens = [1, 2, 3, 4, 5] * 10
    model = NGramLanguageModel(n=3, discount=0.5, vocab_size=10)
    model.fit(train_tokens)
    metrics = model.evaluate(test_tokens)
    assert metrics["loss"] > 0.0
    assert metrics["perplexity"] > 1.0
    assert metrics["total_tokens"] == len(test_tokens)


def test_tokenizer_roundtrip(tmp_path):
    from data.tokenizer import BPETokenizerWrapper
    sample_text = "The quick brown fox jumps over the lazy dog. 12345! Neural Cellular Automata."
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(sample_text * 10, encoding="utf-8")

    save_path = tmp_path / "tokenizer.json"
    tokenizer = BPETokenizerWrapper.train_from_files(
        files=[corpus_file],
        vocab_size=300,
        min_frequency=1,
        save_path=save_path,
    )

    assert save_path.exists()
    assert tokenizer.vocab_size <= 300

    encoded = tokenizer.encode(sample_text)
    assert len(encoded) > 0
    decoded = tokenizer.decode(encoded)
    assert decoded.strip() == sample_text.strip()

    # Test loading
    loaded_tok = BPETokenizerWrapper(save_path)
    assert loaded_tok.vocab_size == tokenizer.vocab_size
    assert loaded_tok.encode(sample_text) == encoded

