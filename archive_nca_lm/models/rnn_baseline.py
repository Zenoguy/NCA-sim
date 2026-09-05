"""
Recurrent Language Model Baseline (GRU).
Serves as the classic recurrent sequence modeling comparison point.
"""

import torch
import torch.nn as nn


class GRULM(nn.Module):
    """
    Multi-layer GRU Language Model with tied embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # PyTorch GRU supports dropout between layers if num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T]
        emb = self.dropout(self.tok_embed(x))  # [B, T, d_model]
        out, _ = self.gru(emb, h_0)             # [B, T, d_model]
        out = self.ln_f(out)
        logits = self.head(out)                # [B, T, vocab_size]
        return logits
