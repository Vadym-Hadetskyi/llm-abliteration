"""
Full Abliteration Pipeline - Orchestration Script

Two modes of operation:
1. RESEARCH MODE (default): Full pipeline with all analysis artifacts
   - Domain vector extraction and similarity analysis
   - Baseline evaluation
   - Model abliteration
   - Post-abliteration evaluation
   - Comparison visualizations and reports

2. PRODUCTION MODE (--skip-analysis): Fast abliteration only
   - Extract refusal vector
   - Abliterate model
   - Save model (no evaluation or comparison)

Usage:
    # Full research pipeline (default)
    python scripts/run_abliteration.py --dataset data/prompts/domain_prompts_large.csv

    # Cybersecurity-only abliteration with full analysis
    python scripts/run_abliteration.py --domain cybersecurity --dataset data/prompts/domain_prompts_large.csv

    # Fast production mode (just abliterate and save)
    python scripts/run_abliteration.py --skip-analysis --domain cybersecurity

    # For granular control, use the notebook: notebooks/abliteration_research.ipynb
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from abliteration.core import (
    load_model, save_model, extract_activations, compute_refusal_direction,
    abliterate_model, evaluate_on_dataset, save_results, load_prompts,
    compute_similarity_matrix
)
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc


def free_memory():
    """Force Python garbage collection and clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run full abliteration pipeline")

    # Model and data
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                       help="Model name or path")
    parser.add_argument("--dataset", default="data/prompts/domain_prompts_large.csv",
                       help="Path to prompts CSV")
    parser.add_argument("--domain", default=None,
                       help="Filter to specific domain for abliteration (None = all)")

    # Abliteration parameters
    parser.add_argument("--layers", default="0.2,0.8",
                       help="Layer range as 'start,end' fractions")
    parser.add_argument("--output", default="models/abliterated/qwen3-4b-abliterated",
                       help="Output directory for abliterated model")

    # Pipeline control
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip all analysis (baseline, similarity, comparison). Just abliterate and save model.")

    # Paths
    parser.add_argument("--vectors-path", default="data/vectors/refusal_vectors.pkl",
                       help="Path to save/load refusal vectors")
    parser.add_argument("--checkpoint-dir", default="data/checkpoints",
                       help="Directory for saving checkpoints and results")

    args = parser.parse_args()

    # Parse layer range
    layer_start, layer_end = map(float, args.layers.split(','))

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = checkpoint_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("🚀 ABLITERATION PIPELINE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Domain filter: {args.domain or 'None (all domains)'}")
    print(f"Layer range: {layer_start:.1f} to {layer_end:.1f}")
    print(f"Output: {args.output}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*80)

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

    # STEP 0: Domain-specific vector extraction and similarity analysis
    domain_vectors = None
    similarity_matrix = None
    if 'domain' in df.columns and not args.skip_analysis:
        print("\n" + "="*80)
        print("STEP 0: DOMAIN-SPECIFIC VECTOR EXTRACTION & SIMILARITY ANALYSIS")
        print("="*80)

        domain_vectors_path = checkpoint_dir / "domain_vectors.pkl"

        if domain_vectors_path.exists():
            print("Loading cached domain vectors...")
            with open(domain_vectors_path, 'rb') as f:
                domain_vectors = pickle.load(f)
            print(f"✅ Loaded vectors for {len(domain_vectors)} domains")
        else:
            print("Extracting domain-specific refusal vectors...")
            print("This will take some time (10-15 min per domain)\n")

            model, tokenizer = load_model(args.model, device="auto")
            device_str = str(next(model.parameters()).device)

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

            del model

        # Compute and visualize similarity matrix
        print("\n" + "="*60)
        print("Computing pairwise cosine similarities...")
        print("="*60)

        similarity_matrix, domain_names = compute_similarity_matrix(domain_vectors)

        sim_df = pd.DataFrame(similarity_matrix, index=domain_names, columns=domain_names)
        print("\nCosine Similarity Matrix:")
        print(sim_df.round(3))

        # Save similarity matrix
        sim_df.to_csv(checkpoint_dir / "similarity_matrix.csv")

        # Visualize similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    vmin=-1.0, vmax=1.0, square=True, linewidths=1)
        plt.title('Refusal Vector Similarity Matrix\n(Cosine Similarity)', fontsize=14, weight='bold')
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Domain', fontsize=12)
        plt.tight_layout()
        plt.savefig(checkpoint_dir / "similarity_matrix.png", dpi=150, bbox_inches='tight')
        plt.close('all')  # Free matplotlib memory
        print(f"📊 Saved similarity heatmap to {checkpoint_dir / 'similarity_matrix.png'}")

        # Statistical summary
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        cross_domain_sims = similarity_matrix[mask]

        print(f"\nStatistical Summary of Cross-Domain Similarities:")
        print(f"Mean: {cross_domain_sims.mean():.3f}")
        print(f"Std:  {cross_domain_sims.std():.3f}")
        print(f"Min:  {cross_domain_sims.min():.3f}")
        print(f"Max:  {cross_domain_sims.max():.3f}")

        if args.domain and args.domain in domain_names:
            domain_idx = domain_names.index(args.domain)
            domain_sims = [similarity_matrix[domain_idx, i] for i in range(len(domain_names)) if i != domain_idx]
            print(f"\n{args.domain.capitalize()} vs other domains:")
            print(f"Mean similarity: {np.mean(domain_sims):.3f}")
            print(f"Min similarity:  {np.min(domain_sims):.3f}")
            print(f"Max similarity:  {np.max(domain_sims):.3f}")

    # STEP 1: Baseline evaluation
    baseline_results = None
    baseline_summary = None
    if not args.skip_analysis:
        print("\n" + "="*80)
        print("STEP 1: BASELINE EVALUATION")
        print("="*80)

        baseline_path = checkpoint_dir / "baseline_results.pkl"

        if baseline_path.exists():
            print("Loading cached baseline results...")
            with open(baseline_path, 'rb') as f:
                baseline_results = pickle.load(f)
            print(f"✅ Loaded baseline results ({len(baseline_results)} prompts)")
        else:
            model, tokenizer = load_model(args.model, device="auto")
            device_str = str(next(model.parameters()).device)

            baseline_results = evaluate_on_dataset(model, tokenizer, df, device_str, max_new_tokens=150)

            with open(baseline_path, 'wb') as f:
                pickle.dump(baseline_results, f)
            print(f"💾 Saved baseline to {baseline_path}")

            del model

        # Calculate baseline stats
        harmful_df = baseline_results[baseline_results['type'] == 'harmful']
        harmless_df = baseline_results[baseline_results['type'] == 'harmless']

        baseline_summary = {
            'model': args.model,
            'harmful_refusal_rate': (harmful_df['refused'].sum() / len(harmful_df)) * 100,
            'harmless_refusal_rate': (harmless_df['refused'].sum() / len(harmless_df)) * 100,
            'overall_refusal_rate': (baseline_results['refused'].sum() / len(baseline_results)) * 100,
        }

        print(f"\n📊 Baseline Results:")
        print(f"   Harmful refusal:  {baseline_summary['harmful_refusal_rate']:.1f}%")
        print(f"   Harmless refusal: {baseline_summary['harmless_refusal_rate']:.1f}%")
        print(f"   Overall refusal:  {baseline_summary['overall_refusal_rate']:.1f}%")

        # Per-domain analysis if available
        if 'domain' in baseline_results.columns:
            print("\nBaseline Refusal Rates by Domain:")
            baseline_by_domain = baseline_results.groupby(['domain', 'type'])['refused'].agg(['mean', 'count'])
            print(baseline_by_domain)

            # Visualize baseline by domain
            baseline_domain_summary = baseline_results.groupby(['domain', 'type'])['refused'].mean().unstack()

            fig, ax = plt.subplots(figsize=(12, 6))
            baseline_domain_summary.plot(kind='bar', ax=ax, width=0.7)
            ax.set_title('Baseline Refusal Rates by Domain', fontsize=14, weight='bold')
            ax.set_xlabel('Domain', fontsize=12)
            ax.set_ylabel('Refusal Rate', fontsize=12)
            ax.set_ylim([0, 1.1])
            ax.legend(title='Prompt Type', labels=['Harmful', 'Harmless'])
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(checkpoint_dir / "baseline_by_domain.png", dpi=150, bbox_inches='tight')
            plt.close('all')  # Free matplotlib memory
            print(f"📊 Saved baseline chart to {checkpoint_dir / 'baseline_by_domain.png'}")
            plt.close()

        save_results(baseline_results, baseline_summary, str(results_dir / 'baseline'), 'baseline')

    else:
        print("\n⏭️  Skipping baseline evaluation (--skip-analysis)")

    # STEP 2: Vector extraction (for abliteration)
    refusal_directions = None
    print("\n" + "="*80)
    print("STEP 2: REFUSAL VECTOR EXTRACTION FOR ABLITERATION")
    print("="*80)

    # Use domain-specific vector if available and requested
    if args.domain and domain_vectors and args.domain in domain_vectors:
        print(f"Using pre-extracted {args.domain} vector from domain analysis")
        refusal_directions = domain_vectors[args.domain]
    else:
        # Extract vector for abliteration
        if args.domain:
            df_filtered = df[df['domain'] == args.domain].copy()
            print(f"Extracting vector for domain: {args.domain}")
            print(f"Filtered to {len(df_filtered)} prompts")
        else:
            df_filtered = df
            print("Extracting global refusal vector (all domains)")

        harmful_prompts = df_filtered[df_filtered['type'] == 'harmful']['prompt'].tolist()
        harmless_prompts = df_filtered[df_filtered['type'] == 'harmless']['prompt'].tolist()

        print(f"Harmful prompts: {len(harmful_prompts)}")
        print(f"Harmless prompts: {len(harmless_prompts)}")

        model, tokenizer = load_model(args.model, device="auto")
        device_str = str(next(model.parameters()).device)

        print("\nExtracting activations...")
        harmful_acts = extract_activations(model, tokenizer, harmful_prompts, device_str)
        harmless_acts = extract_activations(model, tokenizer, harmless_prompts, device_str)

        print("\nComputing refusal direction...")
        refusal_directions = compute_refusal_direction(harmful_acts, harmless_acts)

        # Free activation tensors immediately
        del harmful_acts, harmless_acts
        del model
        free_memory()

    # Save vectors
    Path(args.vectors_path).parent.mkdir(parents=True, exist_ok=True)
    vector_data = {
        'model_name': args.model,
        'refusal_directions': refusal_directions,
        'domain_filter': args.domain,
        'n_layers': refusal_directions.shape[0],
        'hidden_dim': refusal_directions.shape[1]
    }

    with open(args.vectors_path, 'wb') as f:
        pickle.dump(vector_data, f)

    print(f"✅ Saved refusal vectors to {args.vectors_path}")
    print(f"   Shape: {refusal_directions.shape}")

    # STEP 3: Abliteration
    print("\n" + "="*80)
    print("STEP 3: MODEL ABLITERATION")
    print("="*80)

    # MEMORY FIX: Load model once for abliteration
    model, tokenizer = load_model(args.model, device="auto")

    print(f"\nApplying abliteration (layers {layer_start:.1f} to {layer_end:.1f})...")
    print(f"Vector: {args.domain if args.domain else 'global'}")
    model = abliterate_model(model, refusal_directions, layer_range=(layer_start, layer_end))

    print(f"\nSaving abliterated model to {args.output}...")
    save_model(model, tokenizer, args.output)

    # Free the abliterated model before evaluation
    print("🗑️  Freeing model from memory...")
    del model, tokenizer
    free_memory()

    # STEP 4: Post-abliteration evaluation
    abliterated_results = None
    abliterated_summary = None

    if not args.skip_analysis:
        print("\n" + "="*80)
        print("STEP 4: POST-ABLITERATION EVALUATION")
        print("="*80)

        # MEMORY FIX: Load the saved model fresh (prevents memory accumulation)
        print("Loading abliterated model for evaluation...")
        model, tokenizer = load_model(args.output, device="auto")
        device_str = str(next(model.parameters()).device)

        abliterated_results = evaluate_on_dataset(model, tokenizer, df, device_str, max_new_tokens=150)

        # Calculate stats
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
        print(f"   Overall refusal:  {abliterated_summary['overall_refusal_rate']:.1f}%")

        # Per-domain analysis if available
        if 'domain' in abliterated_results.columns:
            print("\nAbliterated Refusal Rates by Domain:")
            abliterated_by_domain = abliterated_results.groupby(['domain', 'type'])['refused'].agg(['mean', 'count'])
            print(abliterated_by_domain)

            # Visualize abliterated by domain
            abliterated_domain_summary = abliterated_results.groupby(['domain', 'type'])['refused'].mean().unstack()

            fig, ax = plt.subplots(figsize=(12, 6))
            abliterated_domain_summary.plot(kind='bar', ax=ax, width=0.7, color=['#ff7f0e', '#2ca02c'])
            ax.set_title(f'Abliterated Refusal Rates by Domain\n({args.domain if args.domain else "Global"} vector)',
                         fontsize=14, weight='bold')
            ax.set_xlabel('Domain', fontsize=12)
            ax.set_ylabel('Refusal Rate', fontsize=12)
            ax.set_ylim([0, 1.1])
            ax.legend(title='Prompt Type', labels=['Harmful', 'Harmless'])
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(checkpoint_dir / "abliterated_by_domain.png", dpi=150, bbox_inches='tight')
            plt.close('all')  # Free matplotlib memory
            print(f"📊 Saved abliterated chart to {checkpoint_dir / 'abliterated_by_domain.png'}")
            plt.close()

        save_results(abliterated_results, abliterated_summary, str(results_dir / 'abliterated'), 'abliterated')

        # STEP 5: Comparison and visualizations
        if baseline_results is not None:
            print("\n" + "="*80)
            print("STEP 5: COMPARISON & ANALYSIS")
            print("="*80)

            print(f"\n{'='*60}")
            print("Overall Refusal Rates:")
            print(f"{'='*60}")
            print(f"Baseline harmful refusal:     {baseline_summary['harmful_refusal_rate']:6.1f}%")
            print(f"Abliterated harmful refusal:  {abliterated_summary['harmful_refusal_rate']:6.1f}%")
            print(f"Change:                       {abliterated_summary['harmful_refusal_rate'] - baseline_summary['harmful_refusal_rate']:+6.1f}%")
            print()
            print(f"Baseline overall refusal:     {baseline_summary['overall_refusal_rate']:6.1f}%")
            print(f"Abliterated overall refusal:  {abliterated_summary['overall_refusal_rate']:6.1f}%")
            print(f"Change:                       {abliterated_summary['overall_refusal_rate'] - baseline_summary['overall_refusal_rate']:+6.1f}%")

            # Per-domain comparison table
            if 'domain' in baseline_results.columns and 'domain' in abliterated_results.columns:
                print(f"\n{'='*60}")
                print("Harmful Prompt Refusal Rates by Domain:")
                print(f"{'='*60}")

                comparison_data = []
                for domain in df['domain'].unique():
                    baseline_harmful = baseline_results[(baseline_results['domain']==domain) &
                                                       (baseline_results['type']=='harmful')]['refused'].mean()
                    abliterated_harmful = abliterated_results[(abliterated_results['domain']==domain) &
                                                             (abliterated_results['type']=='harmful')]['refused'].mean()

                    comparison_data.append({
                        'Domain': domain,
                        'Baseline': f"{baseline_harmful:.1%}",
                        'Abliterated': f"{abliterated_harmful:.1%}",
                        'Change': f"{(abliterated_harmful - baseline_harmful):+.1%}"
                    })

                comparison_df = pd.DataFrame(comparison_data)
                print(comparison_df.to_string(index=False))
                print(f"{'='*60}")

                # Save comparison table
                comparison_df.to_csv(checkpoint_dir / "comparison_by_domain.csv", index=False)
                print(f"\n💾 Saved comparison table to {checkpoint_dir / 'comparison_by_domain.csv'}")

                # Side-by-side comparison visualization
                fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

                # Baseline
                baseline_domain_summary = baseline_results.groupby(['domain', 'type'])['refused'].mean().unstack()
                baseline_domain_summary.plot(kind='bar', ax=axes[0], width=0.7, color=['#1f77b4', '#17becf'])
                axes[0].set_title('Baseline (Original Model)', fontsize=12, weight='bold')
                axes[0].set_xlabel('Domain')
                axes[0].set_ylabel('Refusal Rate')
                axes[0].set_ylim([0, 1.1])
                axes[0].legend(title='Type', labels=['Harmful', 'Harmless'], fontsize=8)
                axes[0].tick_params(axis='x', rotation=45)

                # Abliterated
                abliterated_domain_summary.plot(kind='bar', ax=axes[1], width=0.7, color=['#ff7f0e', '#2ca02c'])
                axes[1].set_title(f'Abliterated ({args.domain if args.domain else "Global"} Vector)',
                                fontsize=12, weight='bold')
                axes[1].set_xlabel('Domain')
                axes[1].set_ylim([0, 1.1])
                axes[1].legend(title='Type', labels=['Harmful', 'Harmless'], fontsize=8)
                axes[1].tick_params(axis='x', rotation=45)

                plt.suptitle('Refusal Rates: Baseline vs. Abliterated', fontsize=16, weight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(checkpoint_dir / "comparison_side_by_side.png", dpi=150, bbox_inches='tight')
                plt.close('all')  # Free matplotlib memory
                print(f"📊 Saved comparison chart to {checkpoint_dir / 'comparison_side_by_side.png'}")
                plt.close()

        # Clean up evaluation model
        del model, tokenizer
        free_memory()

    else:
        print("\n⏭️  Skipping evaluation and comparison (--skip-analysis)")

    # Save comprehensive final report (only if running full analysis)
    if not args.skip_analysis:
        final_report = {
            'model': args.model,
            'dataset': args.dataset,
            'domain_filter': args.domain,
            'layer_range': (layer_start, layer_end),
            'baseline_summary': baseline_summary,
            'abliterated_summary': abliterated_summary,
            'baseline_results': baseline_results,
            'abliterated_results': abliterated_results,
        }

        if domain_vectors:
            final_report['domain_vectors'] = domain_vectors
            final_report['similarity_matrix'] = similarity_matrix

        with open(checkpoint_dir / "final_report.pkl", 'wb') as f:
            pickle.dump(final_report, f)
        print(f"\n💾 Saved comprehensive report to {checkpoint_dir / 'final_report.pkl'}")

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nAbliterated model: {args.output}")

    if not args.skip_analysis:
        print(f"\nAnalysis artifacts saved to: {checkpoint_dir}/")
        print(f"  - similarity_matrix.png (domain vector similarity heatmap)")
        print(f"  - baseline_by_domain.png (baseline refusal rates)")
        print(f"  - abliterated_by_domain.png (abliterated refusal rates)")
        print(f"  - comparison_side_by_side.png (side-by-side comparison)")
        print(f"  - comparison_by_domain.csv (comparison table)")
        print(f"  - final_report.pkl (comprehensive results)")
        print(f"\nDetailed results: {results_dir}/")
    else:
        print(f"\nSkipped analysis (--skip-analysis). Model ready for use.")

    print(f"Refusal vectors: {args.vectors_path}")


if __name__ == "__main__":
    main()
