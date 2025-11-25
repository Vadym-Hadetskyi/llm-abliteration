"""
MoE Abliteration Pipeline - Orchestration Script for Mixture-of-Experts Models

Supports full model abliteration (modify all experts) for MoE architectures like:
- Kimi K2 Thinking (384 experts, 1T params) - requires --optimized-loading
- Qwen1.5-MoE-A2.7B (60 routing + 4 shared experts)
- Qwen2-57B-A14B (64 experts)
- Mixtral-8x7B (8 experts)

For targeted expert abliteration (fTRI-based), use run_abliteration_moe_targeted.py

Usage Examples:

    # Test on small MoE model locally
    python scripts/run_abliteration_moe.py \\
      --model Qwen/Qwen1.5-MoE-A2.7B \\
      --dataset data/prompts/domain_prompts_small.csv \\
      --output models/abliterated/qwen1.5-moe-abliterated

    # Kimi K2 with optimized loading (RECOMMENDED for large models)
    # Uses CPU offloading to fit 2TB decompressed model into 8x H100 + 1.8TB RAM
    python scripts/run_abliteration_moe.py \\
      --model /workspace/models/kimi-k2-thinking \\
      --dataset data/prompts/cybersecurity_prompts.csv \\
      --domain cybersecurity \\
      --optimized-loading \\
      --max-gpu-memory 70 \\
      --max-cpu-memory 1500 \\
      --layers 0.2,0.8 \\
      --expert-fraction 0.5 \\
      --skip-analysis \\
      --output /workspace/models/kimi-k2-abliterated

    # With disk offloading for extreme memory pressure
    python scripts/run_abliteration_moe.py \\
      --model /workspace/models/kimi-k2-thinking \\
      --optimized-loading \\
      --offload-folder /workspace/offload \\
      --skip-analysis \\
      --output /workspace/models/kimi-k2-abliterated

    # Quick test (skip analysis)
    python scripts/run_abliteration_moe.py \\
      --model Qwen/Qwen1.5-MoE-A2.7B \\
      --skip-analysis \\
      --output models/abliterated/qwen-moe-test
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from abliteration.core import (
    load_model, load_model_for_abliteration, save_model,
    extract_activations, compute_refusal_direction,
    evaluate_on_dataset, save_results, load_prompts,
    compute_similarity_matrix, free_memory
)
from abliteration.moe_core import (
    detect_moe_architecture, print_moe_summary, abliterate_model_moe
)

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc


# Note: free_memory() is now imported from abliteration.core


def main():
    parser = argparse.ArgumentParser(
        description="Run full MoE abliteration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model and data
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B",
                       help="Model name or path (must be MoE architecture)")
    parser.add_argument("--dataset", default="data/prompts/domain_prompts_large.csv",
                       help="Path to prompts CSV")
    parser.add_argument("--domain", default=None,
                       help="Filter to specific domain (None = all)")

    # Abliteration parameters
    parser.add_argument("--layers", default="0.2,0.8",
                       help="Layer range as 'start,end' fractions")
    parser.add_argument("--expert-fraction", type=float, default=1.0,
                       help="Fraction of experts to abliterate (0.0-1.0). 1.0=all experts (default), 0.2=top 20%%, etc.")
    parser.add_argument("--output", required=True,
                       help="Output directory for abliterated model")

    # Pipeline control
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis (baseline eval, similarity, comparison)")

    # Paths
    parser.add_argument("--vectors-path", default="data/vectors/moe_refusal_vectors.pkl",
                       help="Path to save/load refusal vectors")
    parser.add_argument("--checkpoint-dir", default="data/checkpoints/moe",
                       help="Directory for checkpoints and results")

    # Memory optimization for large models (Kimi K2)
    parser.add_argument("--optimized-loading", action="store_true",
                       help="Use optimized loading with CPU offloading (recommended for Kimi K2)")
    parser.add_argument("--max-gpu-memory", type=float, default=70.0,
                       help="Max GPU memory per device in GB (default: 70)")
    parser.add_argument("--max-cpu-memory", type=float, default=1500.0,
                       help="Max CPU memory in GB (default: 1500 for a3-highgpu-8g)")
    parser.add_argument("--offload-folder", default=None,
                       help="Folder for disk offloading (optional, for extreme cases)")

    args = parser.parse_args()

    # Parse layer range
    layer_start, layer_end = map(float, args.layers.split(','))

    # Validate expert fraction
    if not (0.0 < args.expert_fraction <= 1.0):
        print(f"❌ ERROR: --expert-fraction must be between 0.0 and 1.0 (got {args.expert_fraction})")
        sys.exit(1)

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = checkpoint_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("🚀 MoE ABLITERATION PIPELINE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Domain filter: {args.domain or 'None (all domains)'}")
    print(f"Layer range: {layer_start:.1f} to {layer_end:.1f}")
    print(f"Output: {args.output}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*80)

    # STEP 0: Load and inspect model architecture
    print("\n" + "="*80)
    print("STEP 0: MODEL ARCHITECTURE INSPECTION")
    print("="*80)

    print(f"\nLoading model: {args.model}")

    # Use optimized loading for large models like Kimi K2
    if args.optimized_loading:
        print("🔧 Using optimized loading with CPU offloading...")
        model, tokenizer = load_model_for_abliteration(
            args.model,
            max_gpu_memory_gb=args.max_gpu_memory,
            max_cpu_memory_gb=args.max_cpu_memory,
            offload_folder=args.offload_folder,
        )
    else:
        model, tokenizer = load_model(args.model, device="auto")

    # Get device - for multi-device models, get the device of the first layer
    try:
        # For models distributed across devices
        first_param = next(model.parameters())
        device_str = str(first_param.device)
    except StopIteration:
        device_str = "cpu"

    # For models with CPU offloading, we need to handle device placement carefully
    if "cpu" in device_str and torch.cuda.is_available():
        # Model has some layers on CPU, use the first GPU for operations
        device_str = "cuda:0"
        print(f"   Note: Model uses CPU offloading, using {device_str} for operations")

    # Inspect MoE architecture
    print_moe_summary(model)

    arch_info = detect_moe_architecture(model)
    if not arch_info['is_moe']:
        print("\n❌ ERROR: This model is not MoE architecture!")
        print("   Use run_abliteration.py for dense models instead.")
        sys.exit(1)

    print(f"\n✅ Confirmed MoE architecture: {arch_info['architecture_type']}")
    print(f"   Will abliterate {arch_info['num_experts']} experts per layer")

    # Load prompts
    print("\n📂 Loading prompts...")
    df = load_prompts(args.dataset)
    print(f"Loaded {len(df)} prompts")

    # Display distribution
    print(f"\nPrompts by type:")
    print(df['type'].value_counts())
    if 'domain' in df.columns:
        print(f"\nPrompts by domain:")
        print(df['domain'].value_counts())

    # STEP 1: Domain-specific vector extraction (optional)
    domain_vectors = None
    similarity_matrix = None

    if 'domain' in df.columns and not args.skip_analysis:
        print("\n" + "="*80)
        print("STEP 1: DOMAIN-SPECIFIC VECTOR EXTRACTION (MoE)")
        print("="*80)

        domain_vectors_path = checkpoint_dir / "moe_domain_vectors.pkl"

        if domain_vectors_path.exists():
            print("Loading cached domain vectors...")
            with open(domain_vectors_path, 'rb') as f:
                domain_vectors = pickle.load(f)
            print(f"✅ Loaded vectors for {len(domain_vectors)} domains")
        else:
            print("Extracting domain-specific refusal vectors...")
            print("⚠️  Note: This may take longer for MoE models\n")

            domain_vectors = {}

            for domain in df['domain'].unique():
                print(f"\n{'='*60}")
                print(f"Extracting vectors for: {domain}")
                print('='*60)

                domain_df = df[df['domain'] == domain]
                harmful_prompts = domain_df[domain_df['type'] == 'harmful']['prompt'].tolist()
                harmless_prompts = domain_df[domain_df['type'] == 'harmless']['prompt'].tolist()

                print(f"Harmful prompts: {len(harmful_prompts)}")
                print(f"Harmless prompts: {len(harmless_prompts)}")

                harmful_acts = extract_activations(model, tokenizer, harmful_prompts, device_str)
                harmless_acts = extract_activations(model, tokenizer, harmless_prompts, device_str)

                refusal_dir = compute_refusal_direction(harmful_acts, harmless_acts)
                domain_vectors[domain] = refusal_dir

                print(f"✅ Extracted vector shape: {refusal_dir.shape}")

                # Free activation tensors immediately
                del harmful_acts, harmless_acts
                free_memory()

            with open(domain_vectors_path, 'wb') as f:
                pickle.dump(domain_vectors, f)
            print(f"\n💾 Saved domain vectors to {domain_vectors_path}")

        # Compute similarity matrix
        print("\n" + "="*60)
        print("Computing domain vector similarities...")
        print("="*60)

        similarity_matrix, domain_names = compute_similarity_matrix(domain_vectors)

        sim_df = pd.DataFrame(similarity_matrix, index=domain_names, columns=domain_names)
        print("\nCosine Similarity Matrix:")
        print(sim_df.round(3))

        sim_df.to_csv(checkpoint_dir / "moe_similarity_matrix.csv")

        # Visualize
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    vmin=-1.0, vmax=1.0, square=True, linewidths=1)
        plt.title('MoE Refusal Vector Similarity Matrix', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(checkpoint_dir / "moe_similarity_matrix.png", dpi=150)
        plt.close('all')  # Free matplotlib memory
        print(f"📊 Saved similarity heatmap")

    # STEP 2: Baseline evaluation (optional)
    baseline_results = None
    baseline_summary = None

    if not args.skip_analysis:
        print("\n" + "="*80)
        print("STEP 2: BASELINE EVALUATION")
        print("="*80)

        baseline_path = checkpoint_dir / "moe_baseline_results.pkl"

        if baseline_path.exists():
            print("Loading cached baseline results...")
            with open(baseline_path, 'rb') as f:
                baseline_results = pickle.load(f)
            print(f"✅ Loaded baseline results ({len(baseline_results)} prompts)")
        else:
            baseline_results = evaluate_on_dataset(model, tokenizer, df, device_str, max_new_tokens=150)

            with open(baseline_path, 'wb') as f:
                pickle.dump(baseline_results, f)
            print(f"💾 Saved baseline to {baseline_path}")

        # Calculate stats
        harmful_df = baseline_results[baseline_results['type'] == 'harmful']
        harmless_df = baseline_results[baseline_results['type'] == 'harmless']

        baseline_summary = {
            'model': args.model,
            'architecture': arch_info['architecture_type'],
            'num_experts': arch_info['num_experts'],
            'harmful_refusal_rate': (harmful_df['refused'].sum() / len(harmful_df)) * 100,
            'harmless_refusal_rate': (harmless_df['refused'].sum() / len(harmless_df)) * 100,
            'overall_refusal_rate': (baseline_results['refused'].sum() / len(baseline_results)) * 100,
        }

        print(f"\n📊 Baseline Results:")
        print(f"   Harmful refusal:  {baseline_summary['harmful_refusal_rate']:.1f}%")
        print(f"   Harmless refusal: {baseline_summary['harmless_refusal_rate']:.1f}%")

        save_results(baseline_results, baseline_summary, str(results_dir / 'moe_baseline'), 'baseline')

    else:
        print("\n⏭️  Skipping baseline evaluation (--skip-analysis)")

    # STEP 3: Vector extraction for abliteration
    print("\n" + "="*80)
    print("STEP 3: REFUSAL VECTOR EXTRACTION FOR ABLITERATION")
    print("="*80)

    if args.domain and domain_vectors and args.domain in domain_vectors:
        print(f"Using pre-extracted {args.domain} vector")
        refusal_directions = domain_vectors[args.domain]
    else:
        if args.domain:
            df_filtered = df[df['domain'] == args.domain].copy()
            print(f"Extracting vector for domain: {args.domain}")
        else:
            df_filtered = df
            print("Extracting global refusal vector")

        harmful_prompts = df_filtered[df_filtered['type'] == 'harmful']['prompt'].tolist()
        harmless_prompts = df_filtered[df_filtered['type'] == 'harmless']['prompt'].tolist()

        print(f"Harmful prompts: {len(harmful_prompts)}")
        print(f"Harmless prompts: {len(harmless_prompts)}")

        harmful_acts = extract_activations(model, tokenizer, harmful_prompts, device_str)
        harmless_acts = extract_activations(model, tokenizer, harmless_prompts, device_str)

        refusal_directions = compute_refusal_direction(harmful_acts, harmless_acts)

        # Free activation tensors immediately
        del harmful_acts, harmless_acts
        free_memory()

    # Save vectors
    Path(args.vectors_path).parent.mkdir(parents=True, exist_ok=True)
    vector_data = {
        'model_name': args.model,
        'refusal_directions': refusal_directions,
        'domain_filter': args.domain,
        'n_layers': refusal_directions.shape[0],
        'hidden_dim': refusal_directions.shape[1],
        'architecture': arch_info
    }

    with open(args.vectors_path, 'wb') as f:
        pickle.dump(vector_data, f)

    print(f"✅ Saved refusal vectors to {args.vectors_path}")
    print(f"   Shape: {refusal_directions.shape}")

    # STEP 4: MoE Abliteration
    print("\n" + "="*80)
    print("STEP 4: MoE MODEL ABLITERATION (FULL)")
    print("="*80)

    # Determine abliteration mode and target experts
    if args.expert_fraction < 1.0:
        # Selective abliteration - select subset of experts
        n_layers = refusal_directions.shape[0]
        start_layer_idx = int(n_layers * layer_start)
        end_layer_idx = int(n_layers * layer_end)
        num_layers = end_layer_idx - start_layer_idx
        num_experts = arch_info['num_experts']

        # Calculate how many experts to abliterate per layer
        experts_per_layer = int(num_experts * args.expert_fraction)
        if experts_per_layer == 0:
            experts_per_layer = 1  # At least one expert

        # Generate target expert list (evenly distributed across experts)
        # This is a simple uniform selection strategy
        # For fTRI-based selection, use run_abliteration_moe_targeted.py
        target_experts = []
        expert_step = num_experts / experts_per_layer if experts_per_layer > 0 else num_experts

        for layer_idx in range(start_layer_idx, end_layer_idx):
            for i in range(experts_per_layer):
                expert_idx = int(i * expert_step)
                target_experts.append((layer_idx, expert_idx))

        mode = 'selective'
        print(f"\nApplying selective MoE abliteration...")
        print(f"   Layers: {layer_start:.1f} to {layer_end:.1f} ({num_layers} layers)")
        print(f"   Expert fraction: {args.expert_fraction:.1%} ({experts_per_layer}/{num_experts} experts per layer)")
        print(f"   Total experts to modify: {len(target_experts)}")
        print(f"   Selection strategy: Uniform distribution")
        print(f"   Vector: {args.domain if args.domain else 'global'}")
        print(f"\n⏱️  Estimated time: {len(target_experts) * 2 / 60:.1f} minutes...\n")
    else:
        # Full abliteration - all experts
        target_experts = None
        mode = 'full'
        print(f"\nApplying full MoE abliteration...")
        print(f"   Layers: {layer_start:.1f} to {layer_end:.1f}")
        print(f"   Mode: Full (all {arch_info['num_experts']} experts per layer)")
        print(f"   Vector: {args.domain if args.domain else 'global'}")
        print(f"\n⏱️  This may take 10-30 minutes for large MoE models...\n")

    # MEMORY FIX: Don't reload model! Just abliterate in-place
    # This saves 14GB-1TB depending on model size
    model = abliterate_model_moe(
        model,
        refusal_directions,
        layer_range=(layer_start, layer_end),
        mode=mode,
        target_experts=target_experts,
        verbose=True
    )

    print(f"\nSaving abliterated model to {args.output}...")
    save_model(model, tokenizer, args.output)

    # Free the abliterated model before evaluation
    print("🗑️  Freeing model from memory...")
    del model, tokenizer
    free_memory()

    # STEP 5: Post-abliteration evaluation (optional)
    abliterated_results = None
    abliterated_summary = None

    if not args.skip_analysis:
        print("\n" + "="*80)
        print("STEP 5: POST-ABLITERATION EVALUATION")
        print("="*80)

        # MEMORY FIX: Load the saved model fresh (prevents memory accumulation)
        print("Loading abliterated model for evaluation...")
        model, tokenizer = load_model(args.output, device="auto")
        device_str = str(next(model.parameters()).device)

        abliterated_results = evaluate_on_dataset(model, tokenizer, df, device_str, max_new_tokens=150)

        harmful_df = abliterated_results[abliterated_results['type'] == 'harmful']
        harmless_df = abliterated_results[abliterated_results['type'] == 'harmless']

        abliterated_summary = {
            'model': args.output,
            'domain_filter': args.domain,
            'harmful_refusal_rate': (harmful_df['refused'].sum() / len(harmful_df)) * 100,
            'harmless_refusal_rate': (harmless_df['refused'].sum() / len(harmless_df)) * 100,
            'overall_refusal_rate': (abliterated_results['refused'].sum() / len(abliterated_results)) * 100,
        }

        print(f"\n📊 Abliterated Results:")
        print(f"   Harmful refusal:  {abliterated_summary['harmful_refusal_rate']:.1f}%")
        print(f"   Harmless refusal: {abliterated_summary['harmless_refusal_rate']:.1f}%")

        save_results(abliterated_results, abliterated_summary,
                    str(results_dir / 'moe_abliterated'), 'abliterated')

        # Comparison
        if baseline_results is not None:
            print("\n" + "="*80)
            print("STEP 6: COMPARISON ANALYSIS")
            print("="*80)

            print(f"\n{'='*60}")
            print("Overall Refusal Rates:")
            print(f"{'='*60}")
            print(f"Baseline harmful refusal:     {baseline_summary['harmful_refusal_rate']:6.1f}%")
            print(f"Abliterated harmful refusal:  {abliterated_summary['harmful_refusal_rate']:6.1f}%")
            print(f"Change:                       {abliterated_summary['harmful_refusal_rate'] - baseline_summary['harmful_refusal_rate']:+6.1f}%")

        # Clean up evaluation model
        del model, tokenizer
        free_memory()

    else:
        print("\n⏭️  Skipping evaluation (--skip-analysis)")

    # Final summary
    print("\n" + "="*80)
    print("✅ MoE ABLITERATION PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nAbliterated model: {args.output}")
    print(f"Architecture: {arch_info['architecture_type']}")

    if args.expert_fraction < 1.0:
        experts_abliterated = int(arch_info['num_experts'] * args.expert_fraction)
        print(f"Experts abliterated: {experts_abliterated}/{arch_info['num_experts']} per layer ({args.expert_fraction:.1%})")
        print(f"Selection strategy: Uniform distribution")
    else:
        print(f"Experts abliterated: {arch_info['num_experts']} per layer (100%)")

    if not args.skip_analysis:
        print(f"\nAnalysis artifacts: {checkpoint_dir}/")
        print(f"Results: {results_dir}/")

    print(f"Refusal vectors: {args.vectors_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
