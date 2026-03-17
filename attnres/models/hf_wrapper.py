"""
AttnResWrapper — attaches Block AttnRes to any HuggingFace causal LM.

Works by registering pre-forward hooks on each decoder layer so that the
hidden state entering each layer is replaced by the AttnRes-weighted
combination of previous layer outputs — with zero changes to the model's
weight files or architecture code.

Supported architectures (auto-detected by model class name):
    qwen3, qwen2, llama, mistral, gemma, phi, falcon, gpt_neox, bloom
    + generic fallback that introspects for the largest nn.ModuleList.

Usage:
    from transformers import AutoModelForCausalLM
    from attnres.models.hf_wrapper import AttnResWrapper

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    wrapper = AttnResWrapper(model, num_blocks=8)
    wrapper.apply()   # installs hooks; model.generate() now uses AttnRes

    # Save only the tiny AttnRes weights (~0.5 MB for 7B model)
    wrapper.save("./my-attnres-weights.pt")

    # Load back onto another instance
    wrapper2 = AttnResWrapper(model2, num_blocks=8)
    wrapper2.load("./my-attnres-weights.pt")
    wrapper2.apply()
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.block_attn_res import BlockAttnRes
from ..core.full_attn_res import FullAttnRes


# ---------------------------------------------------------------------------
# Architecture registry: maps model class-name substrings to layer list attrs
# ---------------------------------------------------------------------------

ARCH_CONFIG: Dict[str, Dict[str, str]] = {
    "qwen3":    {"layers_attr": "model.layers", "hidden_attr": "config.hidden_size"},
    "qwen2":    {"layers_attr": "model.layers", "hidden_attr": "config.hidden_size"},
    "llama":    {"layers_attr": "model.layers", "hidden_attr": "config.hidden_size"},
    "mistral":  {"layers_attr": "model.layers", "hidden_attr": "config.hidden_size"},
    "gemma":    {"layers_attr": "model.layers", "hidden_attr": "config.hidden_size"},
    "phi":      {"layers_attr": "model.layers", "hidden_attr": "config.hidden_size"},
    "falcon":   {"layers_attr": "transformer.h", "hidden_attr": "config.hidden_size"},
    "gpt_neox": {"layers_attr": "gpt_neox.layers", "hidden_attr": "config.hidden_size"},
    "bloom":    {"layers_attr": "transformer.h", "hidden_attr": "config.hidden_size"},
    "gpt2":     {"layers_attr": "transformer.h", "hidden_attr": "config.n_embd"},
}


def _resolve_attr(obj: Any, dotted: str) -> Any:
    """Resolve a dotted attribute path like 'model.layers'."""
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _detect_arch(model: nn.Module) -> Tuple[nn.ModuleList, int]:
    """Return (decoder_layers, hidden_dim) for a HuggingFace causal LM."""
    cls_name = type(model).__name__.lower()

    for key, cfg in ARCH_CONFIG.items():
        if key in cls_name:
            layers = _resolve_attr(model, cfg["layers_attr"])
            hidden = _resolve_attr(model, cfg["hidden_attr"])
            return layers, hidden

    # Generic fallback: find the largest nn.ModuleList
    best: Optional[nn.ModuleList] = None
    for _, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and (best is None or len(m) > len(best)):
            best = m

    if best is None:
        raise ValueError(
            f"Cannot auto-detect decoder layers for model type '{type(model).__name__}'. "
            "Pass layers_attr and hidden_dim explicitly."
        )

    # Try to get hidden dim from config
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        # Infer from first layer's first Linear weight
        for m in best[0].modules():
            if isinstance(m, nn.Linear):
                hidden = m.weight.shape[-1]
                break

    if hidden is None:
        raise ValueError("Cannot determine hidden_dim. Pass it explicitly.")

    return best, hidden


class AttnResWrapper:
    """Wraps a HuggingFace causal LM with Block AttnRes residuals via hooks.

    Args:
        model:       HF causal LM (AutoModelForCausalLM instance).
        num_blocks:  N for Block AttnRes (default 8).
        variant:     "block" (recommended) or "full".
        layers_attr: Dotted path to decoder layer list (auto-detected if None).
        hidden_dim:  Hidden dimension d (auto-detected if None).
        eps:         RMSNorm epsilon.
    """

    def __init__(
        self,
        model: nn.Module,
        num_blocks: int = 8,
        variant: str = "block",
        layers_attr: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        eps: float = 1e-6,
    ):
        self.model = model
        self.num_blocks = num_blocks
        self.variant = variant
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._applied = False

        # Resolve decoder layers and hidden dim
        if layers_attr is not None:
            self._layers: nn.ModuleList = _resolve_attr(model, layers_attr)
            self._hidden_dim: int = hidden_dim or self._infer_hidden_dim()
        else:
            self._layers, self._hidden_dim = _detect_arch(model)
            if hidden_dim is not None:
                self._hidden_dim = hidden_dim

        num_layers = len(self._layers)

        # Build the AttnRes module
        if variant == "block":
            self.attnres = BlockAttnRes(
                num_layers=num_layers,
                hidden_dim=self._hidden_dim,
                num_blocks=num_blocks,
                eps=eps,
            )
        elif variant == "full":
            self.attnres = FullAttnRes(
                num_layers=num_layers,
                hidden_dim=self._hidden_dim,
                eps=eps,
            )
        else:
            raise ValueError(f"variant must be 'block' or 'full', got '{variant}'")

        # Attach as a submodule so it's saved with the model
        self.model.attnres = self.attnres

        # State used across hook calls within one forward pass
        self._layer_outputs: List[torch.Tensor] = []
        self._embedding: Optional[torch.Tensor] = None
        self._attnres_inputs: Optional[List[torch.Tensor]] = None

    def _infer_hidden_dim(self) -> int:
        for m in self._layers[0].modules():
            if isinstance(m, nn.Linear):
                return m.weight.shape[-1]
        raise ValueError("Cannot infer hidden_dim from first decoder layer.")

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def apply(self):
        """Install forward hooks on all decoder layers.

        After calling apply(), self.model.forward() (and .generate()) will
        automatically use AttnRes residuals.
        """
        if self._applied:
            return
        self._remove_hooks()

        # Pre-forward hook on the whole model to capture embedding
        self._hooks.append(
            self.model.register_forward_pre_hook(self._model_pre_hook)
        )
        # Post-forward hook on the whole model to clean up state
        self._hooks.append(
            self.model.register_forward_hook(self._model_post_hook)
        )

        # Pre-forward hook on each decoder layer to inject AttnRes input
        for idx, layer in enumerate(self._layers):
            hook = layer.register_forward_pre_hook(
                self._make_layer_pre_hook(idx)
            )
            self._hooks.append(hook)

            # Post-forward hook to capture layer output
            hook = layer.register_forward_hook(
                self._make_layer_post_hook(idx)
            )
            self._hooks.append(hook)

        self._applied = True

    def remove(self):
        """Remove all installed hooks, restoring the original model."""
        self._remove_hooks()
        self._applied = False

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _model_pre_hook(self, module, args):
        """Called before model.forward() — reset layer output accumulator."""
        self._layer_outputs = []
        self._embedding = None
        self._attnres_inputs = None

    def _model_post_hook(self, module, args, output):
        """Called after model.forward() — clean up temporary state."""
        self._layer_outputs = []
        self._embedding = None
        self._attnres_inputs = None
        return output

    def _make_layer_pre_hook(self, layer_idx: int) -> Callable:
        """Returns a pre-forward hook for decoder layer `layer_idx`.

        On the first layer, captures the embedding (h_0) from the input.
        Once all layer outputs are available (second pass through via
        compute_all_inputs), replaces the hidden state with the AttnRes input.
        """
        def hook(module, args):
            # args[0] is the hidden_state tensor
            hidden_state = args[0] if isinstance(args, tuple) else args

            if layer_idx == 0:
                # Capture the embedding (input to the first layer)
                self._embedding = hidden_state.detach()

                # Compute all AttnRes inputs using outputs captured from
                # any previous forward pass. On the very first call of a new
                # sequence we won't have outputs yet, so skip.
                if len(self._layer_outputs) == len(self._layers):
                    self._attnres_inputs = self.attnres.compute_all_inputs(
                        self._layer_outputs, self._embedding
                    )
                    self._layer_outputs = []  # reset for this pass

            # If we have pre-computed AttnRes inputs, inject them
            if self._attnres_inputs is not None and layer_idx < len(self._attnres_inputs):
                new_hidden = self._attnres_inputs[layer_idx]
                if isinstance(args, tuple):
                    return (new_hidden,) + args[1:]
                return (new_hidden,)

            return args

        return hook

    def _make_layer_post_hook(self, layer_idx: int) -> Callable:
        """Returns a post-forward hook that captures each layer's output."""
        def hook(module, args, output):
            # output[0] is the hidden state after this layer
            hidden_out = output[0] if isinstance(output, tuple) else output
            # The layer output for AttnRes is the *delta* (f_l(h_l)),
            # which is hidden_out - args[0] (residual connection)
            hidden_in = args[0] if isinstance(args, tuple) else args
            layer_delta = hidden_out - hidden_in
            self._layer_outputs.append(layer_delta.detach())
            return output

        return hook

    # ------------------------------------------------------------------
    # Save / load AttnRes weights only
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save only the AttnRes pseudo-query weights (tiny checkpoint).

        Args:
            path: File path (e.g. "./attnres-weights.pt").
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "attnres_state_dict": self.attnres.state_dict(),
            "variant": self.variant,
            "num_blocks": self.num_blocks,
            "num_layers": len(self._layers),
            "hidden_dim": self._hidden_dim,
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load AttnRes weights from a checkpoint saved by save().

        Args:
            path: File path to load from.
        """
        state = torch.load(path, map_location="cpu")
        self.attnres.load_state_dict(state["attnres_state_dict"])

    # ------------------------------------------------------------------
    # Convenience: mark only AttnRes params as trainable
    # ------------------------------------------------------------------

    def freeze_base_model(self):
        """Freeze all base model parameters; only AttnRes queries are trainable."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.attnres.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the AttnRes trainable parameters."""
        return list(self.attnres.parameters())

    def param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        attnres = sum(p.numel() for p in self.attnres.parameters())
        return {"total": total, "attnres": attnres, "base": total - attnres}

    def __repr__(self) -> str:
        counts = self.param_count()
        return (
            f"AttnResWrapper(\n"
            f"  variant={self.variant}, num_blocks={self.num_blocks},\n"
            f"  num_layers={len(self._layers)}, hidden_dim={self._hidden_dim},\n"
            f"  attnres_params={counts['attnres']:,} "
            f"({counts['attnres']/counts['total']*100:.3f}% of total)\n"
            f")"
        )
