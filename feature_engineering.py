"""Feature engineering module adapted from OneTrans_Pytorch.

Implements scalar/array/sequence feature processing:
- safe_float: Numerical safety conversion
- squash_numeric: Log compression
- summarize_array: 6 statistical summaries
- sanitize_sequence: Sequence cleaning
"""

import hashlib
import math
from typing import Any, Iterable

ARRAY_STATS = ("mean", "std", "min", "max", "last", "length")


def safe_float(value: Any) -> float:
    """Safely convert any value to float.
    
    - None/NaN/Inf → 0.0
    - Bool → float(value)
    - String → float(value) or SHA1 hash normalized to [0, 1]
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
            return int(digest[:12], 16) / float(16**12)
    return 0.0


def squash_numeric(value: float) -> float:
    """Log compression: copysign(log1p(abs(x)), x).
    
    Preserves sign while compressing extreme values.
    """
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def scalar_feature(value: Any) -> float:
    """Process scalar feature: safe_float → squash_numeric."""
    return squash_numeric(safe_float(value))


def sanitize_sequence(values: Any) -> list[float]:
    """Clean sequence values: [squash_numeric(safe_float(v)) for v in values]."""
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    return [squash_numeric(safe_float(v)) for v in values]


def summarize_array(values: Any) -> list[float]:
    """Compute 6 statistical summaries for array features.
    
    Returns: [mean, std, min, max, last, log_length]
    """
    arr = sanitize_sequence(values)
    if not arr:
        return [0.0] * len(ARRAY_STATS)
    
    mean = sum(arr) / len(arr)
    variance = sum((v - mean) ** 2 for v in arr) / len(arr)
    return [
        mean,
        math.sqrt(variance),
        min(arr),
        max(arr),
        arr[-1],
        math.log1p(len(arr)),
    ]


def build_feature_vector(
    scalar_values: list[Any],
    array_values: list[Any],
    seq_values: list[Any],
    seq_len: int,
) -> tuple[list[float], list[list[float]]]:
    """Build feature vector from raw values.
    
    Args:
        scalar_values: List of scalar feature values
        array_values: List of array feature values
        seq_values: List of sequence feature values
        seq_len: Maximum sequence length
    
    Returns:
        (non_seq_vec, seq_matrix)
        - non_seq_vec: scalar features + array summaries
        - seq_matrix: [seq_len, num_seq_channels]
    """
    # Non-sequence features
    non_seq = [scalar_feature(v) for v in scalar_values]
    for arr in array_values:
        non_seq.extend(summarize_array(arr))
    
    # Sequence features
    seq_channels = [sanitize_sequence(v) for v in seq_values]
    max_len = min(seq_len, max((len(ch) for ch in seq_channels), default=0))
    max_len = max(max_len, 1)
    
    seq_matrix = [[0.0] * len(seq_channels) for _ in range(max_len)]
    for ch_idx, channel in enumerate(seq_channels):
        for step_idx, value in enumerate(channel[:max_len]):
            seq_matrix[step_idx][ch_idx] = value
    
    return non_seq, seq_matrix
