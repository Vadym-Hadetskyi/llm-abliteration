"""
Abliteration Research Library

Core functionality for LLM abliteration experiments focused on domain-specific
refusal mechanism removal for cybersecurity research.
"""

from .core import (
    # Model management
    load_model,
    save_model,

    # Activation extraction
    extract_activations,
    compute_refusal_direction,

    # Abliteration
    orthogonalize_weight,
    abliterate_model,

    # Evaluation
    test_prompt,
    classify_refusal,
    evaluate_on_dataset,

    # Analysis
    compute_cosine_similarity,
    compute_similarity_matrix,

    # Utilities
    load_prompts,
    save_results,
)

__version__ = "0.1.0"
__all__ = [
    "load_model",
    "save_model",
    "extract_activations",
    "compute_refusal_direction",
    "orthogonalize_weight",
    "abliterate_model",
    "test_prompt",
    "classify_refusal",
    "evaluate_on_dataset",
    "compute_cosine_similarity",
    "compute_similarity_matrix",
    "load_prompts",
    "save_results",
]
