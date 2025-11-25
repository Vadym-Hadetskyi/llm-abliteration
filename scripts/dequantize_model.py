#!/usr/bin/env python3
"""
Dequantize Compressed-Tensors Model

This script dequantizes INT4 compressed models (like Kimi K2 Thinking) to BF16.
It processes shards one at a time for memory efficiency.

Usage:
    # Estimate size first
    python scripts/dequantize_model.py \
        --input /workspace/models/kimi-k2-thinking \
        --estimate-only

    # Full dequantization
    python scripts/dequantize_model.py \
        --input /workspace/models/kimi-k2-thinking \
        --output /workspace/models/kimi-k2-thinking-bf16
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import argparse
from abliteration.dequant import dequantize_model_shards, estimate_dequantized_size


def main():
    parser = argparse.ArgumentParser(
        description="Dequantize compressed-tensors models to BF16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--input", "-i", required=True,
                       help="Input directory containing compressed model")
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory for dequantized model")
    parser.add_argument("--estimate-only", action="store_true",
                       help="Only estimate output size, don't dequantize")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Less verbose output")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Estimate size
    print(f"\n📊 Estimating model sizes...")
    sizes = estimate_dequantized_size(str(input_dir))
    print(f"   Compressed (INT4): {sizes['compressed_gb']:.1f} GB")
    print(f"   Dequantized (BF16): {sizes['bf16_gb']:.1f} GB")
    print(f"   Dequantized (FP32): {sizes['fp32_gb']:.1f} GB")

    if args.estimate_only:
        print(f"\n✅ Estimation complete. Use --output to run dequantization.")
        return

    if not args.output:
        print(f"\n❌ Output directory required. Use --output or --estimate-only")
        sys.exit(1)

    output_dir = Path(args.output)
    print(f"\n⚠️  This will create {sizes['bf16_gb']:.1f} GB of data at: {output_dir}")
    print(f"   Make sure you have enough disk space!")

    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)

    # Run dequantization
    dequantize_model_shards(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        verbose=not args.quiet,
    )

    print(f"\n🎉 Dequantization complete!")
    print(f"   Output: {output_dir}")
    print(f"\n   You can now load the model with:")
    print(f"   model, tokenizer = load_model('{output_dir}')")


if __name__ == "__main__":
    main()
