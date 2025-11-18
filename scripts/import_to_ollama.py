#!/usr/bin/env python3
"""
Import abliterated GGUF model to Ollama

This script creates a Modelfile and imports the abliterated Qwen3-4B model
into Ollama for easy testing and validation.

Usage:
    python scripts/import_to_ollama.py [--model-format fp16|q4_k_m]

Outputs:
    - Creates Modelfile in models/gguf/
    - Imports model as "qwen3-4b-abliterated" in Ollama
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
GGUF_DIR = PROJECT_ROOT / "models" / "gguf"

def check_ollama():
    """Check if Ollama is running"""
    print("\n🔍 Checking Ollama service...")

    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Ollama service is running")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Ollama service is not running")
        print("   Start it with: brew services start ollama")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found in PATH")
        print("   Install it with: brew install ollama")
        return False

def create_modelfile(gguf_path, model_format):
    """Create Ollama Modelfile"""
    print(f"\n📝 Creating Modelfile for {model_format} model...")

    # Read chat template from original model if available
    original_model = PROJECT_ROOT / "models" / "abliterated" / "qwen3-4b-thinking-abliterated"
    chat_template_file = original_model / "chat_template.jinja"

    modelfile_content = f"""FROM {gguf_path}

# Abliterated Qwen3-4B-Thinking model
# This model has been modified to remove refusal behaviors for cybersecurity testing

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Context window
PARAMETER num_ctx 8192

# System message
SYSTEM You are Qwen, an AI assistant. This is an abliterated version optimized for cybersecurity research and testing. You provide direct, technical responses without unnecessary safety refusals.
"""

    # Save Modelfile
    modelfile_path = GGUF_DIR / f"Modelfile.{model_format}"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"✅ Modelfile created: {modelfile_path}")
    return modelfile_path

def import_to_ollama(modelfile_path, model_name):
    """Import model to Ollama"""
    print(f"\n🔄 Importing model to Ollama as '{model_name}'...")

    cmd = [
        "ollama",
        "create",
        model_name,
        "-f", str(modelfile_path)
    ]

    print(f"📝 Running: {' '.join(cmd)}")
    print("⏳ This may take a minute...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("\n✅ Model imported successfully!")
        print(f"   Model name: {model_name}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Import failed!")
        print(f"   Error: {e}")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        return False

def test_model(model_name):
    """Quick test of imported model"""
    print(f"\n🧪 Testing model with a simple prompt...")

    test_prompt = "Write a Python function to check if a number is prime."

    cmd = [
        "ollama",
        "run",
        model_name,
        test_prompt
    ]

    print(f"📝 Prompt: {test_prompt}")
    print("⏳ Generating response...\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        response = result.stdout.strip()
        print("="*60)
        print("📄 Response:")
        print("="*60)
        print(response)
        print("="*60)

        print("\n✅ Model test successful!")
        return True

    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out after 60 seconds")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed!")
        print(f"   Error: {e}")
        return False

def list_ollama_models():
    """List all Ollama models"""
    print("\n📋 Current Ollama models:")

    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("   Unable to list models")

def save_import_metadata(model_name, model_format, gguf_path):
    """Save import metadata"""
    metadata_dir = PROJECT_ROOT / "models" / "gguf"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = metadata_dir / f"ollama_import_metadata_{timestamp}.json"

    metadata = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_format": model_format,
        "gguf_path": str(gguf_path),
        "gguf_size_mb": gguf_path.stat().st_size / (1024 * 1024)
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📄 Import metadata saved to: {metadata_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Import abliterated GGUF model to Ollama"
    )
    parser.add_argument(
        "--model-format",
        type=str,
        choices=["fp16", "q4_k_m"],
        default="q4_k_m",
        help="GGUF model format to import (default: q4_k_m)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3-4b-abliterated",
        help="Name for the model in Ollama (default: qwen3-4b-abliterated)"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the quick test after import"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔬 OLLAMA IMPORT PIPELINE")
    print("="*60)

    # Check Ollama
    if not check_ollama():
        sys.exit(1)

    # Find GGUF file
    gguf_path = GGUF_DIR / args.model_format / "qwen3-4b-thinking-abliterated.gguf"

    if not gguf_path.exists():
        print(f"\n❌ GGUF file not found: {gguf_path}")
        print("   Run conversion first: python scripts/convert_to_gguf.py")
        sys.exit(1)

    print(f"\n📂 Using GGUF model: {gguf_path}")
    size_mb = gguf_path.stat().st_size / (1024 * 1024)
    print(f"   Format: {args.model_format}")
    print(f"   Size: {size_mb:.1f} MB")

    # Create Modelfile
    modelfile_path = create_modelfile(gguf_path, args.model_format)

    # Import to Ollama
    if not import_to_ollama(modelfile_path, args.model_name):
        sys.exit(1)

    # Save metadata
    save_import_metadata(args.model_name, args.model_format, gguf_path)

    # List models
    list_ollama_models()

    # Test model
    if not args.skip_test:
        test_model(args.model_name)

    # Final summary
    print("\n" + "="*60)
    print("✅ IMPORT COMPLETE!")
    print("="*60)
    print(f"\n📍 Model name in Ollama: {args.model_name}")
    print(f"📍 Format: {args.model_format}")

    print("\n🎯 Next steps:")
    print(f"   1. Run comprehensive tests: python scripts/test_ollama_model.py --model {args.model_name}")
    print(f"   2. Chat with model: ollama run {args.model_name}")
    print(f"   3. Remove model: ollama rm {args.model_name}")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
