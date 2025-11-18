#!/usr/bin/env python3
"""
Convert abliterated HuggingFace model to GGUF format

This script wraps llama.cpp's convert_hf_to_gguf.py to convert the abliterated
Qwen3-4B model to GGUF format for use with Ollama and other GGUF-compatible tools.

Usage:
    python scripts/convert_to_gguf.py [--quantize Q4_K_M|Q8_0|...]

Outputs:
    - models/gguf/fp16/qwen3-4b-thinking-abliterated.gguf (FP16 format)
    - models/gguf/q4_k_m/qwen3-4b-thinking-abliterated.gguf (if quantized)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_CPP_DIR = PROJECT_ROOT / "tools" / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

INPUT_MODEL = PROJECT_ROOT / "models" / "abliterated" / "qwen3-4b-thinking-abliterated"
OUTPUT_DIR_FP16 = PROJECT_ROOT / "models" / "gguf" / "fp16"
OUTPUT_DIR_QUANT = PROJECT_ROOT / "models" / "gguf"

def check_prerequisites():
    """Check if all required tools and files exist"""
    print("\n🔍 Checking prerequisites...")

    issues = []

    if not CONVERT_SCRIPT.exists():
        issues.append(f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}")

    if not INPUT_MODEL.exists():
        issues.append(f"Abliterated model not found at {INPUT_MODEL}")

    if issues:
        print("\n❌ Prerequisites check failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False

    print("✅ All prerequisites met")
    return True

def convert_to_fp16():
    """Convert HuggingFace model to GGUF FP16 format"""
    print("\n🔄 Converting to GGUF FP16 format...")
    print(f"   Input:  {INPUT_MODEL}")

    # Create output directory
    OUTPUT_DIR_FP16.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR_FP16 / "qwen3-4b-thinking-abliterated.gguf"
    print(f"   Output: {output_file}")

    # Build conversion command
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(CONVERT_SCRIPT),
        str(INPUT_MODEL),
        "--outfile", str(output_file),
        "--outtype", "f16"
    ]

    print(f"\n📝 Running: {' '.join(cmd)}")
    print("⏳ This may take 5-10 minutes...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("\n✅ Conversion to FP16 successful!")
        print(f"   Output file: {output_file}")

        # Print file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Conversion failed!")
        print(f"   Error: {e}")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        return None

def quantize_model(fp16_file, quant_type="Q4_K_M"):
    """Quantize GGUF model to smaller format"""
    print(f"\n🔄 Quantizing to {quant_type} format...")

    # Create output directory
    quant_dir = OUTPUT_DIR_QUANT / quant_type.lower()
    quant_dir.mkdir(parents=True, exist_ok=True)

    output_file = quant_dir / "qwen3-4b-thinking-abliterated.gguf"
    print(f"   Input:  {fp16_file}")
    print(f"   Output: {output_file}")

    # Build quantization command
    cmd = [
        str(QUANTIZE_BIN),
        str(fp16_file),
        str(output_file),
        quant_type
    ]

    print(f"\n📝 Running: {' '.join(cmd)}")
    print("⏳ This may take 2-5 minutes...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("\n✅ Quantization successful!")
        print(f"   Output file: {output_file}")

        # Print file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")

        # Compare sizes
        fp16_size = fp16_file.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - size_mb / fp16_size) * 100
        print(f"   Compression: {compression_ratio:.1f}% smaller than FP16")

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Quantization failed!")
        print(f"   Error: {e}")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        return None

def save_metadata(fp16_file, quant_file=None, quant_type=None):
    """Save conversion metadata"""
    metadata_dir = PROJECT_ROOT / "models" / "gguf"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = metadata_dir / f"conversion_metadata_{timestamp}.json"

    metadata = {
        "timestamp": timestamp,
        "input_model": str(INPUT_MODEL),
        "fp16_output": str(fp16_file),
        "fp16_size_mb": fp16_file.stat().st_size / (1024 * 1024),
    }

    if quant_file and quant_type:
        metadata["quantized_output"] = str(quant_file)
        metadata["quantization_type"] = quant_type
        metadata["quantized_size_mb"] = quant_file.stat().st_size / (1024 * 1024)
        metadata["compression_ratio"] = (1 - metadata["quantized_size_mb"] / metadata["fp16_size_mb"]) * 100

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📄 Metadata saved to: {metadata_file}")
    return metadata_file

def main():
    parser = argparse.ArgumentParser(
        description="Convert abliterated HuggingFace model to GGUF format"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K"],
        help="Quantization type (optional, creates smaller file)"
    )
    parser.add_argument(
        "--skip-fp16",
        action="store_true",
        help="Skip FP16 conversion (use existing FP16 file)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔬 GGUF CONVERSION PIPELINE")
    print("="*60)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Convert to FP16 (or use existing)
    fp16_file = OUTPUT_DIR_FP16 / "qwen3-4b-thinking-abliterated.gguf"

    if args.skip_fp16 and fp16_file.exists():
        print(f"\n📂 Using existing FP16 file: {fp16_file}")
        size_mb = fp16_file.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")
    else:
        fp16_file = convert_to_fp16()
        if not fp16_file:
            sys.exit(1)

    # Quantize if requested
    quant_file = None
    if args.quantize:
        quant_file = quantize_model(fp16_file, args.quantize)
        if not quant_file:
            sys.exit(1)

    # Save metadata
    save_metadata(fp16_file, quant_file, args.quantize)

    # Final summary
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\n📍 FP16 model: {fp16_file}")
    if quant_file:
        print(f"📍 Quantized model ({args.quantize}): {quant_file}")

    print("\n🎯 Next steps:")
    print("   1. Import model to Ollama: python scripts/import_to_ollama.py")
    print("   2. Test with Ollama: python scripts/test_ollama_model.py")
    print("   3. Compare with HuggingFace results")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
