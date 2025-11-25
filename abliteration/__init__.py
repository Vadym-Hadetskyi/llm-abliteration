"""
Abliteration Research Library

Core functionality for LLM abliteration experiments focused on domain-specific
refusal mechanism removal for cybersecurity research.

Supports both dense models and Mixture-of-Experts (MoE) architectures.
"""

from .core import (
    # Model management
    load_model,
    load_model_for_abliteration,  # Optimized loading for large models
    save_model,
    free_memory,

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

# Dequantization utilities for compressed models (Kimi K2, etc.)
from .dequant import (
    dequantize_model_shards,
    dequantize_int4_tensor,
    estimate_dequantized_size,
)

# Quantization utilities for saving abliterated models
from .quantize import (
    quantize_model_int4,
    quantize_model_simple,
    save_quantized_model,
    estimate_quantized_size,
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

__version__ = "0.4.0"  # Added INT4 quantization for saving abliterated models

__all__ = [
    # Core (dense model support)
    "load_model",
    "load_model_for_abliteration",
    "save_model",
    "free_memory",
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

    # Dequantization utilities
    "dequantize_model_shards",
    "dequantize_int4_tensor",
    "estimate_dequantized_size",

    # Quantization utilities
    "quantize_model_int4",
    "quantize_model_simple",
    "save_quantized_model",
    "estimate_quantized_size",

    # MoE-specific
    "detect_moe_architecture",
    "print_moe_summary",
    "abliterate_model_moe",
    "ExpertActivationTracker",
    "compute_expert_resonance",
    "identify_refusal_experts",
]
