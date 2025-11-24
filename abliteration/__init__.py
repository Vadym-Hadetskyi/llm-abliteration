"""
Abliteration Research Library

Core functionality for LLM abliteration experiments focused on domain-specific
refusal mechanism removal for cybersecurity research.

Supports both dense models and Mixture-of-Experts (MoE) architectures.
"""

from .core import (
    # Model management
    load_model,
    save_model,

    # Activation extraction
    extract_activations,
    compute_refusal_direction,

    # Abliteration (dense models)
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

# MoE-specific functionality
from .moe_core import (
    # MoE architecture detection
    detect_moe_architecture,
    print_moe_summary,

    # MoE abliteration
    abliterate_model_moe,

    # Expert tracking
    ExpertActivationTracker,

    # Expert identification (fTRI)
    compute_expert_resonance,
    identify_refusal_experts,
)

__version__ = "0.2.1"  # Fixed GPU memory handling for large MoE models

__all__ = [
    # Core (dense model support)
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

    # MoE-specific
    "detect_moe_architecture",
    "print_moe_summary",
    "abliterate_model_moe",
    "ExpertActivationTracker",
    "compute_expert_resonance",
    "identify_refusal_experts",
]
