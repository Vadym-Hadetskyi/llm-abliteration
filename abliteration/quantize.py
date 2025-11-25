"""
INT4 Quantization Utilities for Abliterated Models.

This module provides functionality to re-quantize abliterated models from BF16 back to INT4
using the same compressed-tensors format as the original Kimi K2 model.

Uses llm-compressor from Neural Magic for efficient quantization.

Usage:
    from abliteration.quantize import quantize_model_int4, save_quantized_model

    # Quantize in-memory model
    quantize_model_int4(model)

    # Save with compressed-tensors format
    save_quantized_model(model, tokenizer, output_dir)
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_kimi_k2_quantization_config() -> Dict[str, Any]:
    """
    Get the quantization configuration matching Kimi K2's original format.

    Based on the config.json from moonshotai/Kimi-K2-Thinking:
    - INT4 symmetric quantization
    - Group size 32
    - Only routed experts are quantized (attention, shared_experts, lm_head are excluded)

    Returns:
        Quantization config dict for llm-compressor
    """
    return {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "format": "pack-quantized",
        "ignore": [
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"
        ],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }


def quantize_model_int4(
    model: AutoModelForCausalLM,
    calibration_data: Optional[Any] = None,
    verbose: bool = True,
) -> None:
    """
    Quantize model weights to INT4 in-place using llm-compressor.

    This applies the same quantization scheme as the original Kimi K2 model:
    - INT4 symmetric quantization with group size 32
    - Only routed experts are quantized
    - Attention layers and shared experts remain in BF16

    Args:
        model: Model to quantize (modified in-place)
        calibration_data: Optional calibration dataset for better quantization
        verbose: Print progress information

    Note:
        Requires llm-compressor: pip install llmcompressor
    """
    try:
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError:
        raise ImportError(
            "llm-compressor is required for INT4 quantization.\n"
            "Install with: pip install llmcompressor\n"
            "See: https://github.com/vllm-project/llm-compressor"
        )

    if verbose:
        print("\n🔧 Quantizing model to INT4...")
        print("   Method: Symmetric INT4 with group size 32")
        print("   Targets: Routed experts only (matching original Kimi K2)")

    # Define the quantization recipe matching Kimi K2's original format
    # We use GPTQ-style quantization for quality
    recipe = GPTQModifier(
        targets=["Linear"],
        scheme="W4A16",  # 4-bit weights, 16-bit activations
        ignore=[
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"
        ],
        dampening_frac=0.1,
    )

    # Apply quantization
    # Note: For best results, provide calibration_data (a small dataset of prompts)
    # Without calibration, uses simple min-max quantization
    oneshot(
        model=model,
        recipe=recipe,
        dataset=calibration_data,
        max_seq_length=2048 if calibration_data else None,
        num_calibration_samples=512 if calibration_data else None,
    )

    if verbose:
        print("✅ Quantization complete!")


def quantize_model_simple(
    model: AutoModelForCausalLM,
    verbose: bool = True,
) -> None:
    """
    Simple INT4 quantization without calibration data.

    Uses straightforward min-max quantization per group. Faster than GPTQ
    but potentially lower quality. Good for testing or when calibration
    data is not available.

    Args:
        model: Model to quantize (modified in-place)
        verbose: Print progress information
    """
    if verbose:
        print("\n🔧 Quantizing model to INT4 (simple mode)...")
        print("   Method: Min-max symmetric INT4 with group size 32")

    # Patterns to skip (same as Kimi K2 original)
    import re
    skip_patterns = [
        re.compile(r"lm_head"),
        re.compile(r".*self_attn.*"),
        re.compile(r".*shared_experts.*"),
        re.compile(r".*mlp\.(gate|up|gate_up|down)_proj.*"),
    ]

    def should_skip(name: str) -> bool:
        return any(p.match(name) for p in skip_patterns)

    num_bits = 4
    group_size = 32
    quantized_count = 0
    skipped_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this layer should be skipped
        if should_skip(name):
            skipped_count += 1
            continue

        # Only quantize 2D weight matrices
        if param.dim() != 2:
            continue

        if verbose and quantized_count == 0:
            print(f"   First quantized layer: {name}")

        # Quantize this parameter
        weight = param.data
        orig_shape = weight.shape
        orig_dtype = weight.dtype

        # Reshape for group quantization
        # Shape: (out_features, in_features) -> (out_features, num_groups, group_size)
        if weight.shape[1] % group_size != 0:
            # Pad if needed (shouldn't happen for Kimi K2)
            skipped_count += 1
            continue

        weight_grouped = weight.reshape(weight.shape[0], -1, group_size)

        # Compute per-group scales (symmetric quantization)
        max_val = weight_grouped.abs().max(dim=2, keepdim=True).values
        scale = max_val / ((1 << (num_bits - 1)) - 1)  # For symmetric: 7 for INT4
        scale = scale.clamp(min=1e-10)  # Avoid division by zero

        # Quantize and dequantize (fake quantization for now)
        # This simulates INT4 precision while keeping BF16 format
        weight_q = (weight_grouped / scale).round().clamp(-8, 7)
        weight_dq = (weight_q * scale).reshape(orig_shape)

        param.data = weight_dq.to(orig_dtype)
        quantized_count += 1

    if verbose:
        print(f"   Quantized {quantized_count} layers, skipped {skipped_count}")
        print("✅ Quantization complete!")


def save_quantized_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    save_compressed: bool = True,
    verbose: bool = True,
) -> None:
    """
    Save model with quantization config for HuggingFace Hub compatibility.

    If save_compressed=True and model was quantized with llm-compressor,
    saves in the efficient compressed-tensors format (INT4 packed).

    Args:
        model: Model to save (should already be quantized)
        tokenizer: Tokenizer to save
        output_dir: Output directory
        save_compressed: If True, save in compressed format (smaller files)
        verbose: Print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n💾 Saving quantized model to: {output_dir}")

    # Check if model has quantization config from llm-compressor
    has_compression = hasattr(model.config, 'quantization_config')

    if save_compressed and has_compression:
        if verbose:
            print("   Format: compressed-tensors (INT4 packed)")
        # llm-compressor handles compressed saving automatically
        model.save_pretrained(output_dir, safe_serialization=True)
    else:
        if verbose:
            print("   Format: safetensors (standard)")
        # Standard save (weights are fake-quantized but stored as BF16)
        model.save_pretrained(output_dir, safe_serialization=True)

        # Add quantization config to saved config
        if not has_compression:
            import json
            config_path = output_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                config["quantization_config"] = get_kimi_k2_quantization_config()
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                if verbose:
                    print("   Added quantization_config to config.json")

    tokenizer.save_pretrained(output_dir)

    if verbose:
        # Estimate saved size
        total_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors"))
        print(f"   Total size: {total_size / 1024**3:.1f} GB")
        print("✅ Model saved successfully!")


def estimate_quantized_size(model: AutoModelForCausalLM) -> Dict[str, float]:
    """
    Estimate the size of the model in different formats.

    Args:
        model: Model to estimate

    Returns:
        Dict with 'bf16_gb', 'int4_gb', 'compression_ratio'
    """
    total_params = sum(p.numel() for p in model.parameters())

    bf16_bytes = total_params * 2  # 2 bytes per param
    int4_bytes = total_params * 0.5  # 0.5 bytes per param (4 bits)

    # Account for scale factors (1 per 32 elements, stored as BF16)
    num_scales = total_params // 32
    scale_bytes = num_scales * 2

    int4_total = int4_bytes + scale_bytes

    return {
        "total_params": total_params,
        "bf16_gb": bf16_bytes / 1024**3,
        "int4_gb": int4_total / 1024**3,
        "compression_ratio": bf16_bytes / int4_total,
    }
