"""
Lazy Dequantization Utilities for Compressed-Tensors Models.

This module provides memory-efficient dequantization for INT4 compressed models
like Kimi K2 Thinking. Instead of loading the entire model into memory at once,
it processes shards one at a time.

Based on the approach from:
https://huggingface.co/moonshotai/Kimi-K2-Thinking/discussions/2

Usage:
    # Dequantize a compressed model to BF16 shards
    from abliteration.dequant import dequantize_model_shards

    dequantize_model_shards(
        input_dir="/workspace/models/kimi-k2-thinking",
        output_dir="/workspace/models/kimi-k2-thinking-bf16",
    )
"""

import torch
import os
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json
import shutil


def dequantize_int4_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    num_bits: int = 4,
    group_size: int = 32,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize a packed INT4 tensor to higher precision.

    This function unpacks INT4 values that are packed 8 per int32,
    applies the scale factors, and returns the dequantized tensor.

    Args:
        packed: Packed INT4 tensor (shape: [out_features, in_features // 8])
        scale: Scale factors (shape: [out_features, in_features // group_size])
        num_bits: Number of bits per weight (default: 4)
        group_size: Quantization group size (default: 32)
        output_dtype: Output tensor dtype (default: bfloat16)

    Returns:
        Dequantized tensor of shape [out_features, in_features]
    """
    pack_factor = 32 // num_bits  # 8 values packed per int32
    mask = (1 << num_bits) - 1  # 0xF for 4-bit

    # Unpack: each int32 contains 8 4-bit values
    unpacked = torch.zeros(
        (packed.shape[0], packed.shape[1] * pack_factor),
        device=packed.device,
        dtype=torch.int32,
    )

    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (packed >> (num_bits * i)) & mask

    # Convert from unsigned to signed (center around 0)
    unpacked = unpacked - (mask + 1) // 2  # Subtract 8 for 4-bit

    # Apply scale factors
    # Scale shape: [out_features, num_groups] where num_groups = in_features / group_size
    # Unpacked shape: [out_features, in_features]
    # Need to broadcast scale across groups
    scale = scale.unsqueeze(2)  # [out_features, num_groups, 1]
    unpacked = unpacked.to(torch.float32)
    unpacked = unpacked.reshape(packed.shape[0], -1, group_size)  # [out, num_groups, group_size]
    dequantized = (unpacked * scale).reshape(packed.shape[0], -1)  # [out, in]

    return dequantized.to(output_dtype)


def process_safetensor_shard(
    input_path: str,
    output_path: str,
    output_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Dict[str, Tuple[int, ...]]:
    """
    Process a single safetensor shard, dequantizing any compressed tensors.

    Args:
        input_path: Path to input safetensor file
        output_path: Path to output safetensor file
        output_dtype: Output dtype for dequantized tensors
        verbose: Print progress information

    Returns:
        Dict mapping tensor names to their shapes
    """
    output_tensors = {}
    tensor_shapes = {}

    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Group keys to find packed tensors and their scales
        packed_keys = [k for k in keys if "_packed" in k]
        scale_keys = [k for k in keys if "_scale" in k]
        regular_keys = [k for k in keys if "_packed" not in k and "_scale" not in k]

        # Process packed tensors (dequantize)
        for packed_key in packed_keys:
            # Find corresponding scale
            base_name = packed_key.replace("_packed", "")
            scale_key = base_name + "_scale"

            if scale_key in scale_keys:
                if verbose:
                    print(f"   Dequantizing: {base_name}")

                packed = f.get_tensor(packed_key)
                scale = f.get_tensor(scale_key)

                # Dequantize
                dequantized = dequantize_int4_tensor(packed, scale, output_dtype=output_dtype)

                # Store with original weight name (remove _packed suffix)
                weight_key = packed_key.replace("_packed", "")
                output_tensors[weight_key] = dequantized
                tensor_shapes[weight_key] = tuple(dequantized.shape)
            else:
                if verbose:
                    print(f"   ⚠️  No scale found for {packed_key}, copying as-is")
                output_tensors[packed_key] = f.get_tensor(packed_key)

        # Copy regular tensors as-is (optionally convert dtype)
        for key in regular_keys:
            tensor = f.get_tensor(key)

            # Convert floating point tensors to output dtype
            if tensor.is_floating_point():
                tensor = tensor.to(output_dtype)

            output_tensors[key] = tensor
            tensor_shapes[key] = tuple(tensor.shape)

    # Save output shard
    save_file(output_tensors, output_path)

    return tensor_shapes


def dequantize_model_shards(
    input_dir: str,
    output_dir: str,
    output_dtype: torch.dtype = torch.bfloat16,
    copy_non_safetensors: bool = True,
    verbose: bool = True,
) -> None:
    """
    Dequantize all shards of a compressed-tensors model.

    This function processes each safetensor shard individually,
    allowing dequantization of very large models with limited RAM.

    Memory usage: ~2x the size of the largest single shard.

    Args:
        input_dir: Directory containing the compressed model
        output_dir: Directory to save the dequantized model
        output_dtype: Output dtype (default: bfloat16)
        copy_non_safetensors: Copy config.json, tokenizer files, etc.
        verbose: Print progress information

    Example:
        dequantize_model_shards(
            input_dir="/workspace/models/kimi-k2-thinking",
            output_dir="/workspace/models/kimi-k2-thinking-bf16",
        )
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Dequantizing model shards")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Output dtype: {output_dtype}")

    # Find all safetensor files
    safetensor_files = sorted(input_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensor files found in {input_dir}")

    print(f"   Found {len(safetensor_files)} shard(s)")

    # Process each shard
    all_tensor_shapes = {}
    for shard_file in tqdm(safetensor_files, desc="Processing shards"):
        output_shard = output_path / shard_file.name

        if verbose:
            print(f"\n📦 Processing: {shard_file.name}")

        shapes = process_safetensor_shard(
            str(shard_file),
            str(output_shard),
            output_dtype=output_dtype,
            verbose=verbose,
        )
        all_tensor_shapes.update(shapes)

    # Copy non-safetensor files (config, tokenizer, etc.)
    if copy_non_safetensors:
        print(f"\n📋 Copying configuration files...")
        for file in input_path.iterdir():
            if file.suffix != ".safetensors" and file.is_file():
                dest = output_path / file.name
                shutil.copy2(file, dest)
                if verbose:
                    print(f"   Copied: {file.name}")

        # Update config.json to remove quantization config
        config_path = output_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            # Remove quantization-related config
            keys_to_remove = ["quantization_config", "compression_config"]
            for key in keys_to_remove:
                if key in config:
                    del config[key]
                    if verbose:
                        print(f"   Removed {key} from config.json")

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    # Update model.safetensors.index.json if it exists
    index_file = output_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)

        # Update weight_map to use dequantized tensor names
        if "weight_map" in index:
            new_weight_map = {}
            for key, shard in index["weight_map"].items():
                # Remove _packed suffix if present
                new_key = key.replace("_packed", "")
                # Skip scale tensors (they're absorbed into dequantized weights)
                if "_scale" not in new_key:
                    new_weight_map[new_key] = shard
            index["weight_map"] = new_weight_map

        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        if verbose:
            print(f"   Updated model.safetensors.index.json")

    print(f"\n✅ Dequantization complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Total tensors: {len(all_tensor_shapes)}")


def estimate_dequantized_size(input_dir: str) -> Dict[str, float]:
    """
    Estimate the size of a dequantized model without actually dequantizing.

    Args:
        input_dir: Directory containing the compressed model

    Returns:
        Dict with 'compressed_gb', 'bf16_gb', 'fp32_gb' estimates
    """
    input_path = Path(input_dir)
    safetensor_files = list(input_path.glob("*.safetensors"))

    total_compressed = 0
    total_elements = 0

    for shard_file in safetensor_files:
        total_compressed += shard_file.stat().st_size

        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # For packed tensors, estimate unpacked size
                if "_packed" in key:
                    # 8 values packed per int32, each becomes 1 element
                    total_elements += tensor.numel() * 8
                elif "_scale" not in key:
                    total_elements += tensor.numel()

    return {
        "compressed_gb": total_compressed / 1024**3,
        "bf16_gb": total_elements * 2 / 1024**3,  # 2 bytes per bf16
        "fp32_gb": total_elements * 4 / 1024**3,  # 4 bytes per fp32
    }
