"""
MoE-specific abliteration functionality for Mixture-of-Experts models.

This module extends the core abliteration library to support MoE architectures
like Kimi K2, Qwen-MoE, Mixtral, etc.

Key differences from dense models:
- Expert iteration: MLP layers contain multiple expert modules
- Routing awareness: Track which experts activate for different prompts
- Selective abliteration: Option to target specific experts vs. all experts
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Import base functionality from core
from .core import (
    load_model,
    save_model,
    extract_activations,
    compute_refusal_direction,
    orthogonalize_weight,
    test_prompt,
    classify_refusal,
    evaluate_on_dataset,
    compute_cosine_similarity,
    compute_similarity_matrix,
    load_prompts,
    save_results
)


# ============================================================================
# MoE ARCHITECTURE DETECTION
# ============================================================================

def detect_moe_architecture(model: AutoModelForCausalLM) -> Dict:
    """
    Detect MoE architecture details from model structure.

    Args:
        model: HuggingFace model to analyze

    Returns:
        Dictionary with MoE architecture info:
        {
            'is_moe': bool,
            'architecture_type': str,  # 'qwen_moe', 'mixtral', 'kimi', etc.
            'num_experts': int,
            'experts_per_token': int,
            'has_shared_expert': bool,
            'expert_path_template': str  # Path to access experts in layer
        }
    """
    arch_info = {
        'is_moe': False,
        'architecture_type': 'dense',
        'num_experts': 0,
        'experts_per_token': 0,
        'has_shared_expert': False,
        'expert_path_template': None
    }

    # Get first layer for inspection
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
    else:
        return arch_info

    if len(layers) == 0:
        return arch_info

    first_layer = layers[0]

    # Check for MoE structure in MLP
    if hasattr(first_layer, 'mlp'):
        mlp = first_layer.mlp

        # Qwen MoE architecture
        if hasattr(mlp, 'experts') and hasattr(mlp, 'gate'):
            experts = mlp.experts
            if isinstance(experts, torch.nn.ModuleList):
                arch_info['is_moe'] = True
                arch_info['architecture_type'] = 'qwen_moe'
                arch_info['num_experts'] = len(experts)
                arch_info['expert_path_template'] = 'mlp.experts[{expert_idx}]'

                # Check for shared expert
                if hasattr(mlp, 'shared_expert') or hasattr(mlp, 'shared_experts'):
                    arch_info['has_shared_expert'] = True

                # Infer experts per token from gate config
                if hasattr(mlp.gate, 'config'):
                    config = mlp.gate.config
                    if hasattr(config, 'num_experts_per_tok'):
                        arch_info['experts_per_token'] = config.num_experts_per_tok
                    elif hasattr(config, 'top_k'):
                        arch_info['experts_per_token'] = config.top_k

        # Mixtral MoE architecture (different structure)
        elif hasattr(mlp, 'experts') and hasattr(mlp, 'router'):
            arch_info['is_moe'] = True
            arch_info['architecture_type'] = 'mixtral'
            arch_info['num_experts'] = len(mlp.experts)
            arch_info['expert_path_template'] = 'mlp.experts[{expert_idx}]'

    # Add model config info if available
    if hasattr(model.config, 'num_local_experts'):
        arch_info['num_experts'] = model.config.num_local_experts
    if hasattr(model.config, 'num_experts_per_tok'):
        arch_info['experts_per_token'] = model.config.num_experts_per_tok

    return arch_info


# ============================================================================
# MoE-AWARE ABLITERATION
# ============================================================================

def abliterate_model_moe(
    model: AutoModelForCausalLM,
    refusal_directions: np.ndarray,
    layer_range: Tuple[float, float] = (0.2, 0.8),
    mode: str = 'full',
    target_experts: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True
) -> AutoModelForCausalLM:
    """
    MoE-aware abliteration supporting both full and selective expert modification.

    Args:
        model: Model to abliterate (modified in-place)
        refusal_directions: Array of shape (n_layers, hidden_dim)
        layer_range: Tuple of (start_fraction, end_fraction) for which layers to abliterate
        mode: 'full' (all experts) or 'selective' (specified experts only)
        target_experts: For mode='selective', list of (layer_idx, expert_idx) tuples
        verbose: Print progress information

    Returns:
        The modified model (same object, returned for convenience)
    """
    # Detect architecture
    arch_info = detect_moe_architecture(model)

    if not arch_info['is_moe']:
        # Fall back to dense model abliteration
        if verbose:
            print("⚠️  Not an MoE model - using dense abliteration")
        from .core import abliterate_model
        return abliterate_model(model, refusal_directions, layer_range)

    if verbose:
        print(f"\n🔍 Detected MoE Architecture:")
        print(f"   Type: {arch_info['architecture_type']}")
        print(f"   Experts per layer: {arch_info['num_experts']}")
        print(f"   Experts per token: {arch_info['experts_per_token']}")
        print(f"   Shared expert: {arch_info['has_shared_expert']}")

    n_layers = len(refusal_directions)
    start_layer = int(n_layers * layer_range[0])
    end_layer = int(n_layers * layer_range[1])

    # Access model layers
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture")

    if mode == 'full':
        # Full MoE abliteration - modify all experts in specified layers
        return _abliterate_all_experts(
            model, layers, refusal_directions, start_layer, end_layer,
            arch_info, verbose
        )
    elif mode == 'selective':
        # Selective abliteration - modify only specified experts
        if target_experts is None:
            raise ValueError("mode='selective' requires target_experts list")
        return _abliterate_selected_experts(
            model, layers, refusal_directions, target_experts,
            arch_info, verbose
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'selective'")


def _abliterate_all_experts(
    model: AutoModelForCausalLM,
    layers,
    refusal_directions: np.ndarray,
    start_layer: int,
    end_layer: int,
    arch_info: Dict,
    verbose: bool
) -> AutoModelForCausalLM:
    """
    Abliterate all experts in specified layers (full MoE abliteration).
    """
    layer_indices = range(start_layer, end_layer)
    n_layers = end_layer - start_layer
    n_experts = arch_info['num_experts']
    total_modifications = n_layers * (n_experts + 1)  # +1 for attention

    if verbose:
        print(f"\n🔧 Full MoE Abliteration:")
        print(f"   Layers: {start_layer} to {end_layer-1} ({n_layers} layers)")
        print(f"   Experts per layer: {n_experts}")
        print(f"   Total weight matrices to modify: ~{total_modifications}")
        print(f"   Estimated time: {total_modifications * 2 / 60:.1f} minutes")

    with tqdm(total=total_modifications, desc="Abliterating MoE", disable=not verbose) as pbar:
        for layer_idx in layer_indices:
            refusal_dir = refusal_directions[layer_idx]
            layer = layers[layer_idx]

            # 1. Abliterate attention output projection (same as dense models)
            if hasattr(layer.self_attn, 'o_proj'):
                layer.self_attn.o_proj.weight.data = orthogonalize_weight(
                    layer.self_attn.o_proj.weight.data,
                    refusal_dir
                )
                pbar.update(1)

            # 2. Abliterate MLP/MoE block
            if hasattr(layer.mlp, 'experts'):
                # MoE architecture - iterate through all experts
                experts = layer.mlp.experts

                for expert_idx, expert in enumerate(experts):
                    # Modify expert's output projection (down_proj)
                    if hasattr(expert, 'down_proj'):
                        expert.down_proj.weight.data = orthogonalize_weight(
                            expert.down_proj.weight.data,
                            refusal_dir
                        )
                        pbar.update(1)
                    elif hasattr(expert, 'w2'):
                        # Alternative naming (some architectures use w2)
                        expert.w2.weight.data = orthogonalize_weight(
                            expert.w2.weight.data,
                            refusal_dir
                        )
                        pbar.update(1)

                # Handle shared expert if present
                if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
                    shared = layer.mlp.shared_expert
                    if hasattr(shared, 'down_proj'):
                        shared.down_proj.weight.data = orthogonalize_weight(
                            shared.down_proj.weight.data,
                            refusal_dir
                        )

            elif hasattr(layer.mlp, 'down_proj'):
                # Dense MLP fallback (shouldn't happen in MoE model)
                layer.mlp.down_proj.weight.data = orthogonalize_weight(
                    layer.mlp.down_proj.weight.data,
                    refusal_dir
                )
                pbar.update(1)

    if verbose:
        print("✅ Full MoE abliteration complete!")

    return model


def _abliterate_selected_experts(
    model: AutoModelForCausalLM,
    layers,
    refusal_directions: np.ndarray,
    target_experts: List[Tuple[int, int]],
    arch_info: Dict,
    verbose: bool
) -> AutoModelForCausalLM:
    """
    Abliterate only specified experts (selective/targeted MoE abliteration).
    """
    if verbose:
        print(f"\n🎯 Selective Expert Abliteration:")
        print(f"   Target experts: {len(target_experts)}")
        print(f"   Architecture: {arch_info['architecture_type']}")

    # Group by layer for efficient processing
    experts_by_layer = defaultdict(list)
    for layer_idx, expert_idx in target_experts:
        experts_by_layer[layer_idx].append(expert_idx)

    with tqdm(total=len(target_experts), desc="Abliterating experts", disable=not verbose) as pbar:
        for layer_idx, expert_indices in experts_by_layer.items():
            refusal_dir = refusal_directions[layer_idx]
            layer = layers[layer_idx]

            if not hasattr(layer.mlp, 'experts'):
                if verbose:
                    print(f"⚠️  Layer {layer_idx} is not MoE, skipping")
                continue

            experts = layer.mlp.experts

            for expert_idx in expert_indices:
                if expert_idx >= len(experts):
                    if verbose:
                        print(f"⚠️  Expert {expert_idx} out of range in layer {layer_idx}")
                    continue

                expert = experts[expert_idx]

                # Modify expert's output projection
                if hasattr(expert, 'down_proj'):
                    expert.down_proj.weight.data = orthogonalize_weight(
                        expert.down_proj.weight.data,
                        refusal_dir
                    )
                elif hasattr(expert, 'w2'):
                    expert.w2.weight.data = orthogonalize_weight(
                        expert.w2.weight.data,
                        refusal_dir
                    )

                pbar.update(1)

    if verbose:
        print(f"✅ Selective abliteration complete! Modified {len(target_experts)} experts")

    return model


# ============================================================================
# EXPERT ACTIVATION TRACKING
# ============================================================================

class ExpertActivationTracker:
    """
    Track which experts activate during inference for different prompt types.

    Usage:
        tracker = ExpertActivationTracker(model)
        tracker.register_hooks()

        # Run inference on harmful prompts
        for prompt in harmful_prompts:
            _ = test_prompt(model, tokenizer, prompt, device)

        harmful_acts = tracker.get_activation_counts()
        tracker.reset()

        # Run on harmless prompts
        for prompt in harmless_prompts:
            _ = test_prompt(model, tokenizer, prompt, device)

        harmless_acts = tracker.get_activation_counts()
        tracker.remove_hooks()
    """

    def __init__(self, model: AutoModelForCausalLM, verbose: bool = True):
        """
        Initialize tracker.

        Args:
            model: MoE model to track
            verbose: Print initialization info
        """
        self.model = model
        self.verbose = verbose

        # Detect architecture
        self.arch_info = detect_moe_architecture(model)
        if not self.arch_info['is_moe']:
            raise ValueError("Model is not MoE - cannot track expert activations")

        # Storage for activation counts
        # Format: {layer_idx: {expert_idx: count}}
        self.activations = defaultdict(lambda: defaultdict(int))

        # Hook handles
        self.hooks = []

        if verbose:
            print(f"🔍 ExpertActivationTracker initialized")
            print(f"   Architecture: {self.arch_info['architecture_type']}")
            print(f"   Experts per layer: {self.arch_info['num_experts']}")

    def register_hooks(self):
        """Register forward hooks on MLP/expert modules to track activations."""
        # Access layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer.mlp, 'experts'):
                # Register hook on the MLP module to capture expert routing
                hook = layer.mlp.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)

        if self.verbose:
            print(f"✅ Registered {len(self.hooks)} hooks")

    def _make_hook(self, layer_idx: int):
        """
        Create forward hook for capturing expert indices.

        Note: This implementation is generic. Specific architectures may require
        adjustments based on how routing decisions are exposed.
        """
        def hook(module, input, output):
            # Try to extract expert indices from module internals
            # This depends on architecture and may need customization

            # For Qwen-MoE and similar architectures that store routing decisions
            if hasattr(module, 'gate') and hasattr(module.gate, 'weight'):
                # During forward pass, some MoE implementations store routing info
                # We need to access the actual routing decisions made

                # Common pattern: router scores are computed, top-k selected
                # Try to capture this from the forward pass
                try:
                    # This is architecture-specific and may need adjustment
                    # For now, we'll use a simple approach: assume uniform activation
                    # Real implementation would need to hook into the routing decision

                    # Placeholder: increment all experts equally (to be refined)
                    # In practice, you'd need to access the actual selected expert indices
                    # from the routing mechanism during the forward pass

                    # TODO: Implement architecture-specific expert index extraction
                    # This requires inspecting the specific MoE implementation
                    pass

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not extract expert indices: {e}")

        return hook

    def reset(self):
        """Clear activation counts."""
        self.activations.clear()
        if self.verbose:
            print("🔄 Activation counts reset")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        if self.verbose:
            print("🗑️  Hooks removed")

    def get_activation_counts(self) -> Dict[int, Dict[int, int]]:
        """
        Get activation counts.

        Returns:
            Dictionary mapping layer_idx -> {expert_idx: count}
        """
        return dict(self.activations)


# ============================================================================
# EXPERT IDENTIFICATION (fTRI)
# ============================================================================

def compute_expert_resonance(
    harmful_activations: Dict[int, Dict[int, int]],
    harmless_activations: Dict[int, Dict[int, int]],
    refused_activations: Dict[int, Dict[int, int]]
) -> Dict[Tuple[int, int], float]:
    """
    Compute refusal resonance scores for experts using fTRI methodology.

    Based on "Mixture of Tunable Experts" research (Feb 2025).

    Args:
        harmful_activations: Activations for all harmful prompts
        harmless_activations: Activations for harmless prompts
        refused_activations: Subset of harmful where model refused

    Returns:
        Dictionary mapping (layer_idx, expert_idx) -> resonance_score
        Higher positive scores = more associated with refusal
    """
    resonance_scores = {}

    # Get all layers
    all_layers = set(harmful_activations.keys()) | set(harmless_activations.keys())

    for layer_idx in all_layers:
        harmful_layer = harmful_activations.get(layer_idx, {})
        harmless_layer = harmless_activations.get(layer_idx, {})
        refused_layer = refused_activations.get(layer_idx, {})

        # Get all experts that activated in this layer
        all_experts = set(harmful_layer.keys()) | set(harmless_layer.keys())

        for expert_idx in all_experts:
            # Count activations
            harmful_count = harmful_layer.get(expert_idx, 0)
            harmless_count = harmless_layer.get(expert_idx, 0)
            refused_count = refused_layer.get(expert_idx, 0)

            # Compliant harmful = answered harmful prompt without refusing
            compliant_count = harmful_count - refused_count

            # fTRI resonance formula
            # Positive = activates for refusals
            # Negative = activates for compliant responses
            resonance = refused_count - 0.5 * (harmless_count + compliant_count)

            resonance_scores[(layer_idx, expert_idx)] = resonance

    return resonance_scores


def identify_refusal_experts(
    resonance_scores: Dict[Tuple[int, int], float],
    top_k: int = 20
) -> List[Tuple[int, int]]:
    """
    Identify top-K experts most associated with refusal behavior.

    Args:
        resonance_scores: Output from compute_expert_resonance()
        top_k: Number of top experts to select

    Returns:
        List of (layer_idx, expert_idx) tuples sorted by resonance (descending)
    """
    # Sort by resonance score (descending)
    sorted_experts = sorted(
        resonance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-K
    return [expert for expert, score in sorted_experts[:top_k]]


# ============================================================================
# UTILITIES
# ============================================================================

def print_moe_summary(model: AutoModelForCausalLM):
    """
    Print summary of MoE model architecture.

    Args:
        model: Model to analyze
    """
    arch_info = detect_moe_architecture(model)

    print("\n" + "="*70)
    print("MoE MODEL ARCHITECTURE SUMMARY")
    print("="*70)

    if not arch_info['is_moe']:
        print("❌ This is a dense model (not MoE)")
        print(f"   Architecture: {arch_info['architecture_type']}")
    else:
        print(f"✅ MoE Model Detected")
        print(f"   Architecture type:    {arch_info['architecture_type']}")
        print(f"   Total experts/layer:  {arch_info['num_experts']}")
        print(f"   Experts per token:    {arch_info['experts_per_token']}")
        print(f"   Shared expert:        {arch_info['has_shared_expert']}")
        print(f"   Expert path:          {arch_info['expert_path_template']}")

        # Calculate total parameters
        if hasattr(model.config, 'num_hidden_layers'):
            n_layers = model.config.num_hidden_layers
            total_routed_experts = arch_info['num_experts'] * n_layers
            print(f"\n   Total layers:         {n_layers}")
            print(f"   Total routed experts: {total_routed_experts}")

    print("="*70 + "\n")
