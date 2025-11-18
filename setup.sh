#!/bin/bash

# LLM Abliteration Research - Automated Setup Script
# This script sets up the complete environment using UV for dependency management

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check if running on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_warning "This script is optimized for macOS but will continue anyway"
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.11 or later."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Found Python version: $PYTHON_VERSION"

    # Check if UV is installed
    if ! command -v uv &> /dev/null; then
        log_info "UV not found. Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        # Verify installation
        if ! command -v uv &> /dev/null; then
            log_error "UV installation failed. Please install manually: https://github.com/astral-sh/uv"
            exit 1
        fi
    fi

    UV_VERSION=$(uv --version)
    log_success "UV is installed: $UV_VERSION"

    # Check if Ollama is installed (optional but recommended)
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version)
        log_success "Ollama is installed: $OLLAMA_VERSION"
    else
        log_warning "Ollama is not installed. You can install it later with: brew install ollama"
    fi
}

# Create virtual environment with UV
create_venv() {
    print_header "Creating Virtual Environment"

    if [ -d ".venv" ]; then
        log_warning "Virtual environment already exists at .venv"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf .venv
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi

    log_info "Creating virtual environment with UV..."
    uv venv .venv --python python3.11

    log_success "Virtual environment created at .venv"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    log_info "Installing project dependencies with UV..."
    log_info "This may take several minutes (especially PyTorch)..."

    # Install dependencies directly with specific latest stable versions
    log_info "Installing core ML frameworks (PyTorch 2.8.0)..."
    uv pip install "torch>=2.8.0" "torchvision>=0.21.0" "torchaudio>=2.8.0"

    log_info "Installing HuggingFace ecosystem (Transformers 4.57.0)..."
    uv pip install "transformers>=4.57.0" "accelerate>=1.3.0" "sentencepiece>=0.2.0" "protobuf>=5.29.0" "safetensors>=0.5.0"

    log_info "Installing TransformerLens (2.14.0+)..."
    uv pip install "transformer-lens>=2.14.0"

    log_info "Installing Apple Silicon optimization (MLX 0.23.0+)..."
    uv pip install "mlx>=0.23.0" "mlx-lm>=0.23.0"

    log_info "Installing data science libraries..."
    uv pip install "numpy>=2.2.0" "pandas>=2.2.0" "matplotlib>=3.10.0" "seaborn>=0.13.0"

    log_info "Installing utilities..."
    uv pip install "einops>=0.8.0" "tqdm>=4.67.0" "datasets>=3.2.0" "jaxtyping>=0.2.0"

    log_info "Installing Jupyter..."
    uv pip install "jupyter>=1.1.0" "ipykernel>=6.30.0" "ipywidgets>=8.1.0"

    log_success "All dependencies installed successfully"
}

# Clone FailSpy abliterator
clone_abliterator() {
    print_header "Setting Up FailSpy Abliterator"

    if [ -d "abliterator" ]; then
        log_warning "Abliterator directory already exists"
        read -p "Do you want to update it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Updating abliterator repository..."
            cd abliterator
            git pull
            cd ..
            log_success "Abliterator updated"
        else
            log_info "Using existing abliterator"
        fi
    else
        log_info "Cloning FailSpy abliterator repository..."
        git clone https://github.com/FailSpy/abliterator.git
        log_success "Abliterator cloned successfully"
    fi

    # Install abliterator requirements if they exist
    if [ -f "abliterator/requirements.txt" ]; then
        log_info "Installing abliterator requirements..."
        uv pip install -r abliterator/requirements.txt
        log_success "Abliterator requirements installed"
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"

    log_info "Creating necessary directories..."

    mkdir -p data/prompts
    mkdir -p data/results/baseline
    mkdir -p data/results/abliterated
    mkdir -p data/vectors
    mkdir -p models/original
    mkdir -p models/abliterated
    mkdir -p notebooks
    mkdir -p scripts
    mkdir -p logs

    log_success "Directory structure created"
}

# Copy experiment scripts
setup_scripts() {
    print_header "Setting Up Experiment Scripts"

    log_info "Creating experiment scripts directory..."

    # Create scripts directory if it doesn't exist
    mkdir -p scripts

    log_info "Experiment scripts template directory created"
    log_info "You can now add your Python scripts to the scripts/ directory"
}

# Setup Ollama models (optional)
setup_ollama() {
    print_header "Ollama Model Setup (Optional)"

    if ! command -v ollama &> /dev/null; then
        log_warning "Ollama not installed. Skipping Ollama setup."
        log_info "To install Ollama: brew install ollama"
        return
    fi

    log_info "Checking if Ollama service is running..."

    # Check if ollama is running by trying to list models
    if ! ollama list &> /dev/null; then
        log_warning "Ollama service is not running"
        log_info "Starting Ollama service in background..."
        ollama serve &> logs/ollama.log &
        sleep 3
    fi

    log_info "Would you like to download Qwen2.5:3b-instruct model now?"
    echo "This will download ~2GB of data. You can do this later with: ollama pull qwen2.5:3b-instruct"
    read -p "Download now? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Downloading Qwen2.5:3b-instruct model..."
        ollama pull qwen2.5:3b-instruct
        log_success "Model downloaded successfully"
    else
        log_info "Skipping model download"
    fi
}

# Create verification script
create_verification_script() {
    print_header "Creating Verification Script"

    cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Verification script to test the installation
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 INSTALLATION VERIFICATION")
    print("="*60 + "\n")

    print("📦 Checking Python packages...\n")

    all_ok = True

    # Core packages
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("transformers", "HuggingFace Transformers")
    all_ok &= check_import("transformer_lens", "TransformerLens")
    all_ok &= check_import("accelerate", "Accelerate")

    # Data science
    all_ok &= check_import("numpy", "NumPy")
    all_ok &= check_import("pandas", "Pandas")
    all_ok &= check_import("matplotlib", "Matplotlib")

    # Utilities
    all_ok &= check_import("einops", "Einops")
    all_ok &= check_import("tqdm", "tqdm")
    all_ok &= check_import("datasets", "Datasets")

    # Optional: MLX (Apple Silicon)
    print("\n📱 Apple Silicon specific packages:")
    check_import("mlx", "MLX (optional)")
    check_import("mlx_lm", "MLX-LM (optional)")

    # Check PyTorch capabilities
    print("\n🔧 PyTorch Configuration:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   MPS (Metal) available: {torch.backends.mps.is_available()}")

        if torch.backends.mps.is_available():
            print("   ✅ Apple Silicon GPU acceleration enabled!")
        elif torch.cuda.is_available():
            print("   ✅ NVIDIA GPU acceleration enabled!")
        else:
            print("   ⚠️  GPU acceleration not available (CPU only)")
    except Exception as e:
        print(f"   ❌ Error checking PyTorch: {e}")
        all_ok = False

    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ All core packages installed successfully!")
        print("\n🎉 Setup complete! You can now run experiments.")
        print("\nNext steps:")
        print("  1. Activate environment: source .venv/bin/activate")
        print("  2. Run baseline test: python scripts/baseline_test.py")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("❌ Some packages failed to import")
        print("Try reinstalling with: uv pip install -e .")
        print("="*60 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

    chmod +x verify_setup.py
    log_success "Verification script created: verify_setup.py"
}

# Run verification
run_verification() {
    print_header "Running Verification"

    log_info "Testing installation..."

    source .venv/bin/activate
    python verify_setup.py
}

# Create activation script
create_activation_helper() {
    cat > activate.sh << 'EOF'
#!/bin/bash
# Helper script to activate the environment

source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""
echo "Available commands:"
echo "  python scripts/baseline_test.py       - Run baseline evaluation"
echo "  python scripts/extract_vectors.py     - Extract refusal vectors"
echo "  python scripts/abliterate_model.py    - Apply abliteration"
echo "  python scripts/evaluate_abliterated.py - Evaluate abliterated model"
echo "  jupyter notebook                      - Launch Jupyter"
echo ""
EOF

    chmod +x activate.sh
    log_success "Created activation helper: activate.sh"
}

# Create README for quick start
create_quickstart() {
    cat > QUICKSTART.md << 'EOF'
# Quick Start Guide

## Activate Environment

```bash
source .venv/bin/activate
# or use the helper:
source activate.sh
```

## Run Experiments

### 1. Baseline Test
```bash
python scripts/baseline_test.py
```

### 2. Extract Refusal Vectors
```bash
python scripts/extract_vectors.py
```

### 3. Abliterate Model
```bash
python scripts/abliterate_model.py
```

### 4. Evaluate Abliterated Model
```bash
python scripts/evaluate_abliterated.py baseline_summary_TIMESTAMP.json
```

## Directory Structure

```
.
├── data/
│   ├── prompts/          # Test prompts
│   ├── results/          # Experiment results
│   └── vectors/          # Extracted refusal vectors
├── models/
│   ├── original/         # Original models
│   └── abliterated/      # Abliterated models
├── scripts/              # Experiment scripts
├── notebooks/            # Jupyter notebooks
└── logs/                 # Log files
```

## Troubleshooting

### MPS (Metal) not working
If you see warnings about MPS, try:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Out of memory
Reduce model size or use quantization in your scripts.

### Import errors
Reinstall dependencies:
```bash
uv pip install -e . --reinstall
```

## Additional Resources

- Experiment plan: `first_experiment_plan.md`
- Sources overview: `sources_overview.md`
- Project README: `README.md`
EOF

    log_success "Created QUICKSTART.md"
}

# Main setup function
main() {
    clear
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                                                        ║"
    echo "║     LLM Abliteration Research - Setup Script          ║"
    echo "║     CrackenAGI Research Project                       ║"
    echo "║                                                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""

    log_info "Starting automated setup..."
    log_info "This will set up everything needed for abliteration experiments"
    echo ""

    # Run setup steps
    check_prerequisites
    create_venv
    install_dependencies
    clone_abliterator
    create_directories
    setup_scripts
    create_verification_script
    run_verification
    create_activation_helper
    create_quickstart

    # Optional Ollama setup
    setup_ollama

    # Final message
    print_header "Setup Complete! 🎉"

    echo "Your abliteration research environment is ready!"
    echo ""
    echo "📍 Project location: $(pwd)"
    echo ""
    echo "🚀 Quick Start:"
    echo "   1. Activate environment:"
    echo "      source .venv/bin/activate"
    echo ""
    echo "   2. See quick start guide:"
    echo "      cat QUICKSTART.md"
    echo ""
    echo "   3. Read detailed experiment plan:"
    echo "      cat first_experiment_plan.md"
    echo ""
    echo "📚 Documentation:"
    echo "   - QUICKSTART.md    - Quick reference"
    echo "   - first_experiment_plan.md - Detailed guide"
    echo "   - sources_overview.md - Research papers"
    echo ""
    echo "Happy researching! 🧪"
    echo ""
}

# Run main function
main "$@"
