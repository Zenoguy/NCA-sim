"""
N-Gram Language Model Baseline with Interpolated Kneser-Ney Smoothing.
Serves as the empirical floor (Level 0) for language modeling tasks.
"""

from collections import defaultdict, Counter
import math
from typing import Dict, List, Tuple, Union
import numpy as np
from eval.perplexity import compute_discrete_nll


class NGramLanguageModel:
    """
    Interpolated Kneser-Ney N-gram Language Model operating on integer token IDs.
    Supports arbitrary order n (e.g. 3, 5) with recursive lower-order backoff.
    """
    def __init__(self, n: int = 3, discount: float = 0.75, vocab_size: int = 8192):
        self.n = n
        self.discount = discount
        self.vocab_size = vocab_size
        self.counts = [defaultdict(Counter) for _ in range(n + 1)]
        self.context_totals = [defaultdict(int) for _ in range(n + 1)]
        # Continuation counts for lower order contexts
        self.continuations_in = [defaultdict(set) for _ in range(n + 1)]
        self.is_fitted = False

    def fit(self, tokens: Union[List[int], np.ndarray]):
        """Count n-gram frequencies across the entire token sequence."""
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        total = len(tokens)
        print(f"Fitting {self.n}-gram model on {total:,} tokens (vocab_size={self.vocab_size})...")

        for i in range(total):
            word = tokens[i]
            # Order 1: unigram with empty context ()
            self.counts[1][()][word] += 1
            self.context_totals[1][()] += 1

            # Higher orders 2 to n: context length is order - 1
            for order in range(2, self.n + 1):
                ctx_len = order - 1
                if i >= ctx_len:
                    context = tuple(tokens[i - ctx_len:i])
                    self.counts[order][context][word] += 1
                    self.context_totals[order][context] += 1
                    lower_context = context[1:]
                    self.continuations_in[order - 1][lower_context].add((context[0], word))

        self.is_fitted = True
        print(f"Fitted {self.n}-gram: {len(self.counts[self.n]):,} unique order-{self.n} contexts.")

    def score(self, context: Tuple[int, ...], word: int) -> float:
        """
        Compute interpolated Kneser-Ney conditional probability P(word | context).
        Recursively falls back to lower orders when context has low or zero count.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling score().")

        # Trim context to maximum length (n - 1)
        if len(context) >= self.n:
            context = context[-(self.n - 1):]

        order = len(context) + 1
        d = self.discount

        # Base case: unigram
        if order == 1:
            total_unigrams = self.context_totals[1].get((), 0)
            if total_unigrams > 0:
                count = self.counts[1][()][word]
                # Smoothed unigram with uniform backoff
                return (count + 0.1) / (total_unigrams + 0.1 * self.vocab_size)
            return 1.0 / self.vocab_size

        count = self.counts[order][context][word]
        total = self.context_totals[order].get(context, 0)

        lower_context = context[1:] if len(context) > 1 else ()
        lower_prob = self.score(lower_context, word)

        if total > 0:
            distinct_words = len(self.counts[order][context])
            alpha = (d * distinct_words) / total
            prob = max(count - d, 0.0) / total + alpha * lower_prob
            return prob
        else:
            return lower_prob

    def evaluate(self, tokens: Union[List[int], np.ndarray]) -> Dict[str, float]:
        """
        Compute negative log likelihood and perplexity over a test sequence.
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        log_probs = []
        context_len = self.n - 1
        
        # Avoid zero probability crashes with minimum epsilon
        eps = 1e-12

        for i in range(len(tokens)):
            start = max(0, i - context_len)
            context = tuple(tokens[start:i])
            word = tokens[i]
            prob = self.score(context, word)
            prob = max(prob, eps)
            log_probs.append(math.log(prob))

        metrics = compute_discrete_nll(np.array(log_probs))
        metrics["order"] = self.n
        return metrics
