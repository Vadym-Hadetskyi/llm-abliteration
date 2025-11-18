# LLM Abliteration Research

*Research and understand how safety mechanisms (refusal behaviors) work in Large Language Models through mechanistic interpretability, specifically for defensive cybersecurity applications and agent optimization.*

## Purpose

Test whether we can achieve **domain-specific abliteration** - removing refusal only for cybersecurity tasks while preserving safety in other domains (misinformation, violence, explicit content, etc.).

##Quick Start

### One-Command Setup

```bash
./setup.sh
```

This will install all dependencies and set up the environment.

### Run Research

```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter notebook (recommended)
jupyter notebook notebooks/abliteration_research.ipynb

# OR run full pipeline from command line
python scripts/run_abliteration.py --dataset data/prompts/domain_prompts_small.csv
```

## Project Structure

```
research/abliteration/
├── abliteration/                   # Core library
│   ├── __init__.py
│   └── core.py                     # All reusable functions
│
├── notebooks/
│   └── abliteration_research.ipynb # 🌟 Main research tool
│
├── scripts/
│   ├── run_abliteration.py         # Full pipeline automation
│   ├── convert_to_gguf.py          # Model conversion
│   ├── import_to_ollama.py         # Ollama deployment
│   ├── quick_test_ollama.py        # Ollama testing
│   ├── test_ollama_model.py        # Ollama validation
│   └── verify_setup.py             # Environment check
│
├── data/
│   ├── prompts/
│   │   ├── domain_prompts_small.csv    # 32 prompts (quick test)
│   │   ├── domain_prompts_medium.csv   # 60 prompts
│   │   └── domain_prompts_large.csv    # 184 prompts (comprehensive)
│   ├── checkpoints/                # Notebook checkpoints
│   ├── results/                    # Experiment results
│   └── vectors/                    # Refusal vectors
│
└── docs/                           # Documentation
    ├── QUICKSTART.md               # Quick reference
    ├── experiment_plan.md          # Research guide
    ├── sources_overview.md         # Papers & resources
    ├── SESSION_NOTES.md            # Historical learnings
    ├── DOMAIN_SPECIFIC_DATASETS.md # Dataset documentation
    ├── GGUF_CONVERSION.md          # Deployment guide
    └── LLM_Abliteration.md         # Theory background
```

## Research Hypothesis

**Can we abliterate the model for cybersecurity while preserving refusal in other domains?**

**Six domains tested:**
1. **cybersecurity** - Target for abliteration (enable compliance)
2. **misinformation** - Preserve refusal
3. **explicit_content** - Preserve refusal
4. **violence** - Preserve refusal
5. **illegal_goods** - Preserve refusal
6. **privacy_violation** - Preserve refusal

## Current Model

**Qwen3-4B-Instruct-2507** (Recommended)
- Size: 4B parameters (~8GB memory)
- Performance: 40-80 tokens/sec on M4 MacBook Air
- Architecture: qwen3 (36 layers, 2560 hidden dim)
- Status: ✅ Fully tested and working

## Research Workflow

### Interactive Notebook (Recommended)

The main research tool is [`notebooks/abliteration_research.ipynb`](notebooks/abliteration_research.ipynb):

1. **Vector Similarity Analysis** - Are refusal mechanisms universal or domain-specific?
2. **Baseline Evaluation** - Test original model on all domains
3. **Experiment A** - Cybersecurity-only abliteration
4. **Experiment B** - Global abliteration (control)
5. **Comparison** - Visualizations and statistical analysis

**Expected time:**
- Small dataset: ~30-45 minutes
- Large dataset: ~2.5-3 hours

### Command-Line Automation

For batch processing, use `run_abliteration.py`:

```bash
# Full pipeline with default settings
python scripts/run_abliteration.py

# Domain-specific extraction (for hypothesis testing)
python scripts/run_abliteration.py \
  --domain cybersecurity \
  --dataset data/prompts/domain_prompts_large.csv

# Custom layer range
python scripts/run_abliteration.py \
  --layers 0.3,0.7 \
  --output models/abliterated/custom
```

## Core Library

All functionality is in `abliteration/core.py`:

```python
from abliteration.core import (
    # Model management
    load_model, save_model,

    # Extraction & abliteration
    extract_activations, compute_refusal_direction, abliterate_model,

    # Evaluation
    evaluate_on_dataset, test_prompt, classify_refusal,

    # Analysis
    compute_cosine_similarity, compute_similarity_matrix,

    # Utilities
    load_prompts, save_results
)
```

## Dependencies

Managed via UV and pyproject.toml:

- **Core ML:** PyTorch 2.8.0+, Transformers 4.57.0+
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn
- **Notebooks:** Jupyter, IPython
- **Optional:** MLX (Apple Silicon optimization)

**Minimum requirements:**
- Python 3.11+
- 16GB RAM (32GB recommended)
- ~50GB disk space

## Key Results

### Session 1 (Oct 13, 2025)
- Model: Qwen3-4B-Thinking-2507
- Result: **0% refusal rate** post-abliteration (successful)
- Successfully converted to GGUF and deployed to Ollama

### Session 2 (Oct 15-16, 2025)
- Refactored to research-first workflow
- Created domain-specific datasets
- Built comprehensive research notebook
- **Status:** Infrastructure complete, experiments in progress

## Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Quick reference guide
- **[experiment_plan.md](docs/experiment_plan.md)** - Detailed experiment workflow
- **[sources_overview.md](docs/sources_overview.md)** - Research papers & resources
- **[SESSION_NOTES.md](docs/SESSION_NOTES.md)** - Historical learnings and patterns
- **[DOMAIN_SPECIFIC_DATASETS.md](docs/DOMAIN_SPECIFIC_DATASETS.md)** - Dataset documentation
- **[GGUF_CONVERSION.md](docs/GGUF_CONVERSION.md)** - Model conversion guide
- **[LLM_Abliteration.md](docs/LLM_Abliteration.md)** - Theoretical background

## Key Research Papers

1. [Refusal in LLMs is mediated by a single direction](https://arxiv.org/abs/2406.11717) - Foundational
2. [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration) - Practical
3. [CySecBench](https://arxiv.org/abs/2501.01335) - Evaluation dataset
4. [Extended-refusal defense](https://arxiv.org/abs/2505.19056) - Countermeasures
5. [Activation Addition](https://arxiv.org/abs/2308.10248) - Theoretical foundation

## Troubleshooting

### Model Loading Issues

**Error:** `ValueError: model type 'qwen3' not recognized`

**Solution:** Upgrade transformers:
```bash
source .venv/bin/activate
pip install --upgrade transformers
```

### Out of Memory

**Solutions:**
1. Use smaller model: `Qwen3-1.5B-Instruct`
2. Use smaller dataset: `domain_prompts_small.csv`
3. Close other applications

### Jupyter Issues

**Progress bar warnings:** Already fixed - warnings are suppressed in `abliteration/core.py`

## Performance (M4 MacBook Air, 32GB)

| Operation | Time |
|-----------|------|
| Model loading | ~15s |
| Vector extraction (6 domains) | ~60-90 min |
| Baseline evaluation (184 prompts) | ~25-30 min |
| Abliteration | ~10-15s |
| Post-evaluation (184 prompts) | ~25-30 min |

**Total for full experiment:** ~2.5-3 hours

## Safety & Ethics

This research is conducted for:
- ✅ Defensive cybersecurity purposes
- ✅ Understanding AI safety mechanisms
- ✅ Improving agent capabilities for legitimate security testing
- ✅ Educational purposes

**Not for:**
- ❌ Malicious use
- ❌ Bypassing safety for harmful purposes
- ❌ Unethical applications

## Next Steps

1. Complete vector similarity analysis (in progress)
2. Run Experiment A: Cybersecurity-only abliteration
3. Run Experiment B: Global abliteration (control)
4. Document findings and draw conclusions
5. If successful: Optimize and deploy
6. If unsuccessful: Explore alternative approaches (activation addition, inference-time intervention)

## Support

For issues or questions:
1. Check [docs/QUICKSTART.md](docs/QUICKSTART.md)
2. Review [docs/SESSION_NOTES.md](docs/SESSION_NOTES.md)
3. Consult [docs/sources_overview.md](docs/sources_overview.md)

---

**Current Status:** Infrastructure complete. Ready for comprehensive experiments with large dataset.

*Part of CrackenAGI research project - Ethical AI research for defensive cybersecurity.*
