"""
Entropy calculation utilities for analyzing model uncertainty.
"""

import numpy as np
from typing import Dict, List, Any


def calculate_entropy(token_logprobs_dict: Dict[int, float]) -> float:
    """
    Calculate entropy from token logprobs dictionary.

    Args:
        token_logprobs_dict: Dictionary mapping token_id to logprob (from vLLM top-k)

    Returns:
        Entropy value (in nats)
    """
    if not token_logprobs_dict:
        return 0.0

    # Convert logprobs to probabilities
    logprobs = np.array(list(token_logprobs_dict.values()))
    probs = np.exp(logprobs)

    # Normalize (since we only have top-k)
    probs = probs / np.sum(probs)

    # Calculate entropy: H = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return float(entropy)


def calculate_token_entropies(logprobs_list: List[Dict[int, float]]) -> List[float]:
    """
    Calculate entropy for each token in a sequence.

    Args:
        logprobs_list: List of token logprobs dictionaries

    Returns:
        List of entropy values
    """
    return [calculate_entropy(lp) for lp in logprobs_list]


def get_entropy_statistics(entropies: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of entropy values.

    Args:
        entropies: List of entropy values

    Returns:
        Dictionary with mean, max, min, std, median
    """
    if not entropies:
        return {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
            "median": 0.0,
        }

    entropies_array = np.array(entropies)

    return {
        "mean": float(np.mean(entropies_array)),
        "max": float(np.max(entropies_array)),
        "min": float(np.min(entropies_array)),
        "std": float(np.std(entropies_array)),
        "median": float(np.median(entropies_array)),
    }


def extract_top_k_probs(token_logprobs_dict: Dict[int, float]) -> Dict[str, float]:
    """
    Extract top-k token probabilities for storage.

    Args:
        token_logprobs_dict: Dictionary mapping token_id to logprob

    Returns:
        Dictionary mapping token_id (as string) to probability
    """
    if not token_logprobs_dict:
        return {}

    # Convert to probabilities
    items = [(tid, np.exp(lp)) for tid, lp in token_logprobs_dict.items()]

    # Normalize
    total_prob = sum(p for _, p in items)
    normalized = {str(tid): float(p / total_prob) for tid, p in items}

    return normalized


def analyze_entropy_pattern(step_entropies: List[float]) -> Dict[str, Any]:
    """
    Analyze the pattern of entropy changes across steps.

    Args:
        step_entropies: List of average entropy values for each step

    Returns:
        Dictionary with pattern analysis (trend, stability, etc.)
    """
    if len(step_entropies) < 2:
        return {
            "trend": "insufficient_data",
            "is_increasing": False,
            "is_decreasing": False,
            "is_stable": False,
            "volatility": 0.0,
        }

    # Calculate differences
    diffs = np.diff(step_entropies)

    # Determine trend
    mean_diff = np.mean(diffs)
    trend = "stable"
    if mean_diff > 0.1:
        trend = "increasing"
    elif mean_diff < -0.1:
        trend = "decreasing"

    # Calculate volatility (standard deviation of differences)
    volatility = float(np.std(diffs))

    return {
        "trend": trend,
        "is_increasing": trend == "increasing",
        "is_decreasing": trend == "decreasing",
        "is_stable": trend == "stable",
        "volatility": volatility,
        "mean_change": float(mean_diff),
    }
