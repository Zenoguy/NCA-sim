"""
Byte-Level BPE Tokenizer Module for NCA-LM.
Provides unified training, serialization, encoding, and decoding using HuggingFace Tokenizers.
"""

import os
from pathlib import Path
from typing import List, Union
import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer


class BPETokenizerWrapper:
    """
    Wrapper around HuggingFace's ByteLevelBPETokenizer ensuring consistent
    vocabulary size, special tokens, and caching across all models.
    """
    SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "<pad>"

    def __init__(self, tokenizer_path: Union[str, Path] = None):
        self.tokenizer = None
        self.vocab_size = None
        if tokenizer_path and Path(tokenizer_path).exists():
            self.load(tokenizer_path)

    @classmethod
    def train_from_files(
        cls,
        files: List[Union[str, Path]],
        vocab_size: int = 8192,
        min_frequency: int = 2,
        save_path: Union[str, Path] = None,
    ) -> "BPETokenizerWrapper":
        """Train a ByteLevel BPE tokenizer on a list of text files."""
        instance = cls()
        instance.tokenizer = ByteLevelBPETokenizer()
        
        file_paths = [str(f) for f in files]
        print(f"Training ByteLevel BPE tokenizer on {file_paths} (vocab_size={vocab_size})...")
        instance.tokenizer.train(
            files=file_paths,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
        )
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        print(f"Trained tokenizer with {instance.vocab_size:,} vocabulary tokens.")

        if save_path:
            instance.save(save_path)

        return instance

    def save(self, path: Union[str, Path]):
        """Save tokenizer configuration and vocabulary to a single JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        print(f"Saved tokenizer to {path}")

    def load(self, path: Union[str, Path]):
        """Load tokenizer from a saved JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(self.BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(self.EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id(self.UNK_TOKEN)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not initialized. Train or load first.")
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a string."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not initialized. Train or load first.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_file(self, text_path: Union[str, Path], cache_path: Union[str, Path] = None) -> np.ndarray:
        """
        Encode an entire text file into a 1D numpy array (uint16 for memory efficiency).
        Uses a binary cache (.npy) if available.
        """
        text_path = Path(text_path)
        if cache_path is None:
            cache_path = text_path.with_suffix(".npy")
        else:
            cache_path = Path(cache_path)

        if cache_path.exists() and cache_path.stat().st_mtime > text_path.stat().st_mtime:
            return np.load(cache_path)

        print(f"Encoding {text_path} with tokenizer...")
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        token_ids = self.encode(content, add_special_tokens=False)
        arr = np.array(token_ids, dtype=np.uint16)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
        print(f"Cached {len(arr):,} tokens to {cache_path}")
        return arr
