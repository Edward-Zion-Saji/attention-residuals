# Attention Residuals

A research implementation of **Block Attention Residuals** from the Kimi Team paper:

> *Attention Residuals* — arXiv:2603.15031  
> https://github.com/MoonshotAI/Attention-Residuals

---

## What this is

AttnRes replaces the standard residual connection

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

with learned softmax attention over all preceding layer outputs:

```
h_l = Σ α_{i→l} · v_i      where α are softmax weights from a single learned query w_l ∈ ℝ^d
```

**Block AttnRes** (the practical variant) groups layers into N≈8 blocks and attends over block-level summaries, reducing memory from O(L·d) to O(N·d) while recovering most of the gain.

---

## What this is NOT

This is a **research training library**. It is not an inference optimisation tool.

- ❌ You cannot take a pretrained model (Qwen3.5, LLaMA, etc.) and apply AttnRes to make it better or faster — the model must be **trained from scratch** with AttnRes replacing residual connections from step 1.
- ❌ The "<2% inference overhead" in the paper means an AttnRes-trained model costs 2% *more* than a baseline — it is not a speedup.
- ✅ Train small models from scratch and compare against a baseline (reproduces Fig 4 scaling curves).
- ✅ Study depth-wise attention routing patterns after training (reproduces Fig 8 heatmaps).

---

## Installation

```bash
git clone https://github.com/Edward-Zion-Saji/attention-residuals
cd attention-residuals
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2.

---

## Quick start

### 1. Train a small GPT with vs without AttnRes

```bash
python scripts/train_gpt_demo.py
```

Trains two ~10M-parameter GPTs (baseline and Block AttnRes with N=4 blocks) on synthetic data and saves:
- `outputs/training_dynamics.png` — loss curves + per-layer output magnitudes + gradient norms
- `outputs/scaling_laws.png` — log-log loss comparison

Options:
```bash
python scripts/train_gpt_demo.py \
    --n_layer 6 --n_embd 384 --n_head 6 \
    --num_attnres_blocks 4 \
    --steps 3000 --batch_size 32 \
    --dataset wikitext    # pip install datasets
```

### 2. Use the core modules directly

```python
from attnres.core import BlockAttnRes

ar = BlockAttnRes(num_layers=12, hidden_dim=768, num_blocks=4)

# Training: compute all layer inputs given all layer outputs
layer_outputs = [...]   # list of L tensors, each (B, T, d)
embedding = ...         # (B, T, d)
layer_inputs = ar.compute_all_inputs(layer_outputs, embedding)

# Stateful (sequential) forward
ar.reset_state()
ar.set_embedding(embedding)
h0 = ar.forward(0)    # input to layer 0
# ... run layer 0, push its output ...
ar.push_layer_output(f0_out)
h1 = ar.forward(1)    # input to layer 1 — attends over b_0 + partial sum
```

### 3. Visualise depth-wise attention weights

```python
from attnres.models.gpt_demo import GPTWithAttnRes, GPTConfig
from attnres.visualisation import plot_depth_attention

cfg = GPTConfig(n_layer=6, n_embd=384, use_attnres=True)
model = GPTWithAttnRes(cfg)
# ... load trained checkpoint ...

plot_depth_attention(model, save_path="depth_attention.png")
```

Produces a heatmap showing α_{i→l}: which source layers each consuming layer attends to most.

---

## Repository layout

```
attnres/
  core/
    utils.py              # RMSNorm (no learnable params), zero_init_
    online_softmax.py     # Milakov-Gimelshein merge + AttnWithStats
    full_attn_res.py      # Full AttnRes — attends over all L prior layer outputs
    block_attn_res.py     # Block AttnRes — attends over N block summaries (paper §3.2)
  models/
    gpt_demo.py           # Small GPT with pluggable AttnRes for training experiments
  inference/
    cache.py              # AttnResBlockCache — O(N·d) block representation store
  visualisation/
    attention_maps.py     # Depth-wise α_{i→l} heatmaps (Fig 8)
    training_dynamics.py  # Loss + output magnitude + gradient norm plots (Fig 5)
    scaling_laws.py       # Log-log power-law curves (Fig 4)
scripts/
  train_gpt_demo.py       # End-to-end: train baseline vs AttnRes, plot results
tests/
  test_core.py            # 20 tests — shapes, zero-init, online-softmax, two-phase
```

---

## Key design decisions from the paper

| Decision | Why |
|---|---|
| Zero-init pseudo-queries | Ensures uniform weights at init = standard residuals; avoids training instability |
| RMSNorm on keys | Prevents layers with large-magnitude outputs from dominating softmax |
| N≈8 blocks | Empirically recovers most of Full AttnRes gain; reduces memory O(L·d) → O(N·d) |
| Single d-dim query per layer | Multihead depth attention hurts (Table 4): optimal depth mixture is uniform across channels |
| Softmax not sigmoid | Competitive normalisation forces sharper selection; sigmoid hurts (Table 4) |

---

## Results from the paper

At 5.6 PFLOP/s-days of compute, Block AttnRes (N=8) achieves validation loss **1.692** vs baseline **1.714** — equivalent to a **1.25× compute advantage**.

On the 48B Kimi Linear model trained on 1.4T tokens:

| Benchmark | Baseline | AttnRes |
|---|---|---|
| GPQA-Diamond | 36.9 | **44.4** (+7.5) |
| MATH | 53.5 | **57.1** (+3.6) |
| HumanEval | 59.1 | **62.2** (+3.1) |
| MMLU | 73.5 | **74.6** (+1.1) |

---

## Citation

```bibtex
@article{kimi2026attnres,
  title   = {Attention Residuals},
  author  = {Kimi Team},
  journal = {arXiv preprint arXiv:2603.15031},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.15031}
}
```
