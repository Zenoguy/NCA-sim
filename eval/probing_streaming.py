"""
Probe 3D: Streaming State Complexity Module.

Measures the analytical and empirical state memory footprint M(T) as sequence length T
scales from 128 to 2,048 tokens in incremental autoregressive streaming.

Compares:
  - Transformer: Expanding KV cache O(T), dM/dT > 0.
  - NCA (Causal Dilated Conv): Bounded circular history buffer of size RF=127, dM/dT = 0 for T >= RF.
  - GRU: Recurrent state vector O(1), dM/dT = 0 for all T >= 1.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch


def compute_streaming_state_memory(
    model_type: str,
    T: int,
    batch_size: int = 1,
    d_model: int = 384,
    num_layers: int = 4,
    K: int = 6,
    radius: int = 2,
    dtype_bytes: int = 4,
) -> float:
    """
    Computes required active streaming state memory in Megabytes (MB) at step T.

    Args:
        model_type: 'transformer', 'nca', or 'gru'.
        T: Current token index in streaming generation.
        batch_size: Batch size (default: 1 for interactive generation).
        d_model: Hidden dimension.
        num_layers: Number of layers (for Transformer/GRU).
        K: Number of micro-steps (for NCA).
        radius: Kernel radius (for NCA).
        dtype_bytes: Bytes per float (4 for float32, 2 for float16/bfloat16).

    Returns:
        State memory in MB.
    """
    if model_type == "transformer":
        # KV-cache accumulates keys and values for all T tokens across all layers
        # Shape per token: 2 * num_layers * d_model
        total_elements = 2 * num_layers * d_model * T * batch_size
    elif model_type == "nca":
        # True incremental NCA requires a rolling circular buffer bounded by its receptive field RF
        rf = 1 + radius * (2**K - 1)  # 127 tokens at K=6
        effective_buffer_len = min(T, rf)
        # Hidden state buffer across micro-steps: batch_size * d_model * buffer_len
        total_elements = batch_size * d_model * effective_buffer_len
    elif model_type == "gru":
        # Recurrent state: num_layers * hidden_size * batch_size (constant for all T >= 1)
        total_elements = num_layers * d_model * batch_size
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    bytes_total = total_elements * dtype_bytes
    return bytes_total / (1024 * 1024)  # Convert to MB


def evaluate_streaming_state_complexity(
    seq_lengths: Optional[List[int]] = None,
    batch_size: int = 1,
    models_to_test: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Union[Dict, List]]:
    """
    Evaluates streaming state complexity scaling laws across sequence lengths T.

    Args:
        seq_lengths: List of token steps, e.g. [128, 256, 512, 1024, 2048].
        batch_size: Batch size.
        models_to_test: Dictionary of model specifications.

    Returns:
        Scaling data, regression slopes b (MB per 128 tokens), and comparisons.
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]

    if models_to_test is None:
        models_to_test = {
            "primary_transformer": {
                "type": "transformer",
                "name": "Primary Transformer (10M)",
                "d_model": 384,
                "num_layers": 4,
            },
            "nca_variant_d": {
                "type": "nca",
                "name": "NCA Variant D (Shared 9.7M, d=576)",
                "d_model": 576,
                "K": 6,
                "radius": 2,
            },
            "nca_variant_a": {
                "type": "nca",
                "name": "NCA Variant A (Shared 3.6M, d=288)",
                "d_model": 288,
                "K": 6,
                "radius": 2,
            },
            "gru_baseline": {
                "type": "gru",
                "name": "GRU Baseline (10M)",
                "d_model": 512,
                "num_layers": 3,
            },
        }

    results = {}
    T_arr = np.array(seq_lengths, dtype=np.float64)

    for key, spec in models_to_test.items():
        m_type = spec["type"]
        d_model = spec.get("d_model", 384)
        num_layers = spec.get("num_layers", 4)
        K = spec.get("K", 6)
        radius = spec.get("radius", 2)

        curve = []
        memories = []

        for T in seq_lengths:
            mem_mb = compute_streaming_state_memory(
                model_type=m_type,
                T=T,
                batch_size=batch_size,
                d_model=d_model,
                num_layers=num_layers,
                K=K,
                radius=radius,
            )
            memories.append(mem_mb)
            curve.append({"T": T, "state_memory_mb": round(mem_mb, 4)})

        mem_arr = np.array(memories, dtype=np.float64)

        # Fit linear regression: M(T) = a + b * T
        # Slope b in MB per token -> scale to MB per 128 tokens
        coeffs = np.polyfit(T_arr, mem_arr, deg=1)
        slope_per_token = float(coeffs[0])
        intercept = float(coeffs[1])
        slope_per_128_tokens = slope_per_token * 128.0

        results[key] = {
            "name": spec.get("name", key),
            "type": m_type,
            "curve": curve,
            "intercept_mb": round(intercept, 4),
            "slope_mb_per_token": round(slope_per_token, 6),
            "marginal_slope_mb_per_128_tokens": round(slope_per_128_tokens, 4),
            "asymptotic_complexity": "O(1)" if abs(slope_per_128_tokens) < 1e-4 else "O(T)",
        }

    return results
