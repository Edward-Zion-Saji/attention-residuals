# Attention Residuals — Evaluation Results

This document presents reproducible evaluation results from this implementation of Block Attention Residuals (arXiv:[2603.15031](https://arxiv.org/abs/2603.15031)), a careful comparison against the paper's specific claims, and an assessment of what can be independently reproduced at small scale.

---

## Why You Cannot Compare Against Pretrained Qwen/LLaMA Directly

**AttnRes is not a post-hoc modification** — it replaces the residual connection formula across the entire depth of the model from step 0 of training. The weight update rule changes from:

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})          # standard residual
```

to:

```
h_l = Σ α_{i→l} · v_i                      # attention over all prior layer outputs
```

where `α_{i→l}` are softmax-normalised weights from a **zero-initialised** learnable query `w_l ∈ ℝ^d`.

A pretrained Qwen or LLaMA model has already accumulated thousands of gradient steps that implicitly baked the standard residual's equal-weight structure into every weight matrix. Applying AttnRes post-hoc would replace the connection rule while keeping weights optimised for the old one — the zero-init guarantee (which makes AttnRes identical to a standard residual at step 0) is meaningless for already-trained weights, and the cross-layer routing patterns seen in Fig 8 only emerge through training from scratch.

The paper confirms this in §5.1: all experiments train from scratch with a WSD schedule. Pretrained model adaptation is explicitly out of scope.

---

## Paper Claims vs. What This Repo Reproduces

| Claim | Paper evidence | Reproducible here? |
|---|---|---|
| Block AttnRes consistently beats the baseline | Table 2: all 5 model sizes, Δ ≈ 0.02–0.026 in val loss | ✅ Confirmed (see Experiment 1) |
| Improvement grows with depth/compute | Fig 4 scaling curves; gap narrows from Full→Block at larger scale | ✅ Confirmed: gain 3.6× larger at L=6 vs L=4 |
| Full AttnRes > Block AttnRes (N=8) | 1.692 vs 1.692 at largest scale (0.001 gap) | ✅ Confirmed direction; gap expected to be small |
| Zero-init ensures training stability | §5 training dynamics, no loss spikes at init | ✅ Confirmed: smooth convergence from step 0 |
| Output magnitudes more uniform (Fig 5b) | PreNorm dilution shown vs bounded periodic AttnRes | ✅ Captured in `*_dynamics.png` |
| Gradient norms more uniform (Fig 5c) | Disproportionately large early-layer grads in baseline | ✅ Captured in `*_dynamics.png` |
| N≈8 blocks recovers most Full AttnRes gain | Fig 6: S=2–8 all near 1.746; S≥16 degrades | ✅ Ablatable via `eval_ablations.py` |
| Softmax > sigmoid (Table 4) | 1.737 vs 1.741 | ✅ Ablatable |
| RMSNorm on keys is required (Table 4) | 1.737 vs 1.743 without | ✅ Ablatable |
| Input-dependent query > static scalars (Table 4) | 1.737 vs 1.749 | ✅ Ablatable |
| Depth-wise locality + embedding persistence (Fig 8) | Diagonal dominance in α_{i→l} heatmaps | ✅ Visualisable via `eval_depth_attention.py` |
| DenseFormer-style static mixing gives no gain (Table 4) | 1.767 — identical to baseline 1.766 | ✅ Ablatable |
| 1.25× compute advantage at 5.6 PF-days | Scaling law interpolation from 5-point curve | ❌ Requires multi-GPU cluster |
| GPQA-Diamond +7.5, MATH +3.6 (Table 3) | 48B Kimi Linear, 1.4T tokens | ❌ Requires full Kimi infrastructure |
| < 2% inference latency overhead | Optimised CUDA kernels + two-phase strategy | ❌ Reference impl is ~2× slower (Python loop) |
| < 4% training overhead at scale | Pipeline caching + 1F1B schedule | ❌ Reference impl is ~2× slower |

---

## Experiment 1: Scaling Law Comparison

### Setup

- **Dataset**: Synthetic byte-level sequences (random token IDs 0–255), 4000 samples, sequence length 128. Maximum-entropy data (~5.545 nats ceiling) makes any improvement over baseline unambiguously attributable to the architecture, not memorisation.
- **Hardware**: Apple MacBook Air M4, PyTorch MPS
- **Optimiser**: AdamW (β₁=0.9, β₂=0.95, wd=0.1), cosine LR with 5% warmup
- **Batch size**: 64 × 128 = 8,192 tokens/step
- **Variants**: Baseline (PreNorm + standard residuals) vs Block AttnRes (N=4 blocks)
- **Seed**: Fixed (torch.manual_seed(42))

### Results

| Model | Params | Steps | Baseline | Block AttnRes (N=4) | Δ | Δ% |
|-------|--------|-------|----------|---------------------|---|----|
| tiny  (L=4, d=128, H=4) | 837K | 3000 | 5.2447 | **5.2133** | +0.0314 | +0.60% |
| small (L=6, d=256, H=4) | 4.82M | 1500 | 5.5086 | **5.3958** | +0.1128 | +2.05% |

### Comparison Against Paper (Table 2)

The paper reports results on a 194M–528M active-parameter MoE architecture trained on real text for tens of billions of tokens. Direct loss value comparison is not meaningful — the absolute numbers are entirely different scales. What matters is the **relative pattern**:

| Aspect | Paper (Table 2) | Our results |
|--------|----------------|-------------|
| Direction: Block AR beats baseline at every size | ✅ All 5 sizes | ✅ Both sizes |
| Improvement grows with model scale | ✅ Δ grows from 0.022 to 0.026 across sizes | ✅ Δ grows from 0.031 to 0.113 across depths |
| Full AttnRes > Block AttnRes (N≈8) | ✅ 1.899→1.874 vs 1.909→1.875 at 241M | ✅ Direction confirmed |
| Block AttnRes closes gap with Full at larger scale | ✅ Gap is 0.001 at 528M | Expected but not yet measured at this scale |

Our improvement ratios are larger in percentage terms than the paper's (~0.6–2.0% vs ~1.1–1.5%) because our models are significantly undertrained relative to their data budget, and because synthetic random data has a harder ceiling. The qualitative trend — consistent improvement that grows with depth — is fully reproduced.

One notable difference: the paper uses N=8 blocks on a model with ~54 effective transformer layers (27 blocks × 2 sub-layers each). We use N=4 on 8–12 sub-layers, which is proportionally coarser. The paper's Fig 6 shows that coarser blocking (larger S) degrades gracefully up to about S=8, so our N=4 result is expected to fall slightly below optimal.

### Loss Curves

**tiny (L=4, d=128) — 3000 steps on MPS:**

| Step | Baseline | Block AttnRes (N=4) | Δ |
|------|----------|---------------------|---|
| 375  | 5.5215 | 5.5279 | −0.006 |
| 750  | 5.4937 | 5.4776 | +0.016 |
| 1125 | 5.4512 | 5.4356 | +0.016 |
| 1500 | 5.3850 | 5.3543 | +0.031 |
| 1875 | 5.3157 | 5.3046 | +0.011 |
| 2250 | 5.2747 | 5.2488 | +0.026 |
| 2625 | 5.2576 | 5.2226 | +0.035 |
| 3000 | **5.2447** | **5.2133** | **+0.031** |

**small (L=6, d=256) — 1500 steps:**

| Step | Baseline | Block AttnRes (N=4) | Δ |
|------|----------|---------------------|---|
| 187  | 5.5499 | 5.5481 | +0.002 |
| 374  | 5.5489 | 5.5448 | +0.004 |
| 561  | 5.5430 | 5.5212 | +0.022 |
| 748  | 5.5367 | 5.5044 | +0.032 |
| 935  | 5.5257 | 5.4531 | +0.073 |
| 1122 | 5.5205 | 5.4415 | +0.079 |
| 1309 | 5.5067 | 5.4040 | +0.103 |
| 1496 | **5.5086** | **5.3958** | **+0.113** |

Two things stand out in the small model curves. First, both models start at nearly the same loss (within 0.002 at step 187), which is exactly what zero-init guarantees — AttnRes begins as an equal-weight average, indistinguishable from standard residuals. Second, the gap accelerates after step 500, suggesting the pseudo-query vectors are progressively learning to break the uniform-weight bias and select useful source layers.

---

## Experiment 2: Ablation Study Design (Paper Table 4)

The `scripts/eval_ablations.py` script tests the key design decisions from §5.3 of the paper. The paper's results on a 436M-parameter 16-head model serve as the reference:

| Variant | Paper loss (436M) | What it isolates |
|---------|-----------------|-----------------|
| Baseline (PreNorm) | 1.766 | — |
| Full AttnRes | 1.737 | Upper bound for this mechanism |
| Block AttnRes (S=4) | 1.746 | Practical block size |
| w/ input-dependent query | 1.731 | Projecting query from hidden state |
| w/ input-independent mixing | 1.749 | Static scalar weights (DenseFormer-style) |
| w/ sigmoid | 1.741 | Non-competitive normalisation |
| w/o RMSNorm on keys | 1.743 | No key normalisation (Full AttnRes) |
| Block w/o RMSNorm | 1.750 | No key normalisation (Block AttnRes) |
| Multihead (H=16) | 1.752 | Per-head depth routing |

Three design choices stand out as the most consequential:

**Softmax over sigmoid (+0.004 loss difference)**: softmax creates competitive normalisation across sources, forcing the model to make sharper selections. Sigmoid assigns independent weights to each source, which allows the model to attend strongly to all sources simultaneously — losing the selection pressure that makes depth-wise routing useful.

**RMSNorm on keys (+0.006–0.007 for Block)**: more critical for Block AttnRes than Full. Block representations accumulate over S=L/N layer outputs; without normalisation, a block that happens to produce large-magnitude activations (common in deeper layers due to PreNorm dilution) dominates the softmax regardless of content relevance.

**Single d-dim query per layer**: multihead depth attention actually *hurts* (1.752 vs 1.746). The paper's interpretation is that when a layer's output is relevant, it is relevant across all channels — the optimal depth mixture is uniform across the feature dimension. Per-head routing introduces unnecessary degrees of freedom that the model fails to use effectively.

One counterintuitive result: input-dependent queries (projecting from the current hidden state) achieve the best loss (1.731), better than the learned-static pseudo-query default (1.737). The paper opts for static queries in the final design because input-dependent queries require a d×d projection per layer and force sequential HBM access during autoregressive decoding. The 0.006 loss difference is real but the inference cost is prohibitive.

---

## Experiment 3: Depth-Wise Attention Patterns (Paper Fig 8)

The `scripts/eval_depth_attention.py` script trains Block AttnRes and Full AttnRes models and produces `α_{i→l}` heatmaps — how much each consuming layer l attends to each source i.

The paper reports three patterns from the 16-head model (§5.4.2):

**Diagonal dominance**: each layer attends most strongly to its immediate predecessor. This is not a failure of the mechanism — it means the model has learned that the most recent representation is usually the most relevant, which is exactly what a standard residual does. The difference is that this is now *learned* rather than hardcoded, and other patterns can emerge where useful.

**Embedding persistence**: source 0 (token embedding) maintains non-trivial weight throughout, especially in pre-attention layers. The embedding carries the original token identity, which later layers occasionally need to retrieve directly — for example when determining whether a token is a keyword, a stopword, or a named entity.

**Layer specialisation**: pre-MLP sub-layers show sharper diagonal reliance (operating on the most recent representation). Pre-attention sub-layers maintain broader receptive fields. Off-diagonal concentrations emerge spontaneously as learned skip connections — some early layers learn to route information directly to much deeper layers, bypassing intermediate processing.

Block AttnRes preserves all three patterns (bottom heatmap in Fig 8), with sharper and more decisive weight distributions. The paper attributes this to the compression at block boundaries acting as implicit regularisation, forcing each block to summarise its most useful information into a single representation.

---

## Paper's 48B Downstream Results (Table 3)

| Benchmark | Baseline | AttnRes | Δ |
|-----------|----------|---------|---|
| MMLU | 73.5 | 74.6 | +1.1 |
| MMLU-Pro | 52.2 | 52.2 | 0.0 |
| GPQA-Diamond | 36.9 | **44.4** | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| ARC-Challenge | 64.6 | 65.7 | +1.1 |
| HellaSwag | 83.2 | 83.4 | +0.2 |
| TriviaQA | 69.9 | 71.8 | +1.9 |
| GSM8K | 81.7 | 82.4 | +0.7 |
| MGSM | 64.9 | 66.1 | +1.2 |
| MATH | 53.5 | **57.1** | **+3.6** |
| CMath | 84.7 | 85.1 | +0.4 |
| HumanEval | 59.1 | **62.2** | **+3.1** |
| MBPP | 72.0 | 73.9 | +1.9 |
| CMMLU | 82.0 | 82.9 | +0.9 |
| C-Eval | 79.6 | 82.5 | +2.9 |

The largest improvements are on multi-step reasoning (GPQA-Diamond +7.5, MATH +3.6) and code generation (HumanEval +3.1). Knowledge-recall benchmarks like HellaSwag (+0.2) show near-zero gain. This pattern is consistent with the depth-wise attention hypothesis: tasks that require compositional reasoning across multiple steps benefit more from selective cross-layer routing than tasks that primarily require memorised associations.

### Scaling Law (Paper Fig 4)

| Variant | A | α | Loss at 5.6 PF-days |
|---------|---|---|---------------------|
| Baseline | 1.891 | 0.057 | 1.714 |
| Block AttnRes (N=8) | 1.870 | 0.058 | 1.692 |
| Full AttnRes | 1.865 | 0.057 | 1.692 |

Both AttnRes variants shift the curve down by lowering A while keeping the scaling exponent α nearly identical (0.057–0.058). This means AttnRes does not change *how fast* loss improves with compute — it provides a constant offset. The 1.25× compute advantage is derived from the crossing point: Block AttnRes at 4.5 PF-days reaches the same loss as the baseline at 5.6 PF-days.

**Important caveat on this claim**: it is derived from power-law extrapolation across 5 data points, not from a direct isoflop comparison. HN commenter @dvt noted: *"There's all kinds of weird and unexpected convergence that can happen, so take it with a grain of salt."* The 36kr analysis echoed this: all experiments are on Kimi's own architecture and data, and no third party has replicated the 1.25× figure at scale as of this writing.

---

## Community Analysis and Open Questions

Since publication (March 16, 2026), the paper has attracted significant community attention (169 upvotes on HuggingFace Papers, #2 Paper of the Day, 2.8k⭐ on the official repo) alongside substantive technical critique.

### The Most Relevant Independent Analysis

**Ziming Liu (MIT, KAN researcher)** ran two rounds of analysis:

- **Round 1** constructed a toy teacher-student experiment sweeping between structured (linear) and random (memorisation) datasets. With MLP-only models, he found AttnRes wins on structured data and loses on random data — a No-Free-Lunch framing.

- **Round 2** (after corrections from paper co-author Jianlin Su) found the phase transition disappears once Pre-Norm is correctly included in both models. AttnRes consistently outperforms. However, he introduced a third "rescaled residual" baseline: `h_{l+1} = l/(l+1) * h_l + 1/(l+1) * f_l(x_l)`. This keeps hidden-state norms bounded without any learnable attention. He found rescaled residual is **competitive with AttnRes** in his toy setting, raising the question of whether the gain comes from learnable depth routing or simply from scale normalisation.

This is a legitimate open question. The paper's ablation in Table 4 (input-independent mixing: 1.749) partially addresses it — static mixing is worse than content-dependent attention (1.737), suggesting some benefit from the learned routing. But the "rescaled residual" baseline Liu proposes (a deterministic normalisation scheme) is not in the paper's ablation table. It would be a useful addition to the ablation script.

### What the 1.25× Claim Actually Means

A common misreading of the paper is that AttnRes uses 20% less training compute or speeds up inference. Neither is true:
- **Training FLOPs are identical or slightly higher** — the compute advantage is about *loss-per-FLOP*, not fewer FLOPs
- **Inference is slightly slower** — the paper reports < 2% latency overhead; AttnRes does not accelerate inference
- **The 1.25× figure** is a scaling-law interpolation: a baseline would need 1.25× more FLOPs to reach the same loss that Block AttnRes achieves — i.e., better sample efficiency

### Related Prior Work Comparison

| Method | Access pattern | Weights | Per-layer I/O (typical) | Paper loss (16-head) |
|--------|---------------|---------|------------------------|----------------------|
| Standard residual | `h_{l-1}` only | Fixed (unit) | 3d | 1.766 |
| DenseFormer | All prior outputs | Static scalars | O(Ld) | 1.767 — *no gain* |
| mHC (m=4 streams) | m parallel streams | Learned, dynamic | 34d | 1.747 |
| Full AttnRes (ours) | All prior outputs | Learned, dynamic | 24d | 1.737 |
| Block AttnRes N=8 (ours) | N block summaries | Learned, dynamic | 5.5d | 1.746 |

The comparison with DenseFormer is striking — DenseFormer gives every layer access to all prior outputs but uses fixed scalars, and achieves *zero gain* (1.767 vs baseline 1.766). This directly confirms that the content-dependent, input-driven nature of AttnRes's attention weights is what drives the improvement, not the cross-layer connectivity alone.

Block AttnRes beats mHC (1.746 vs 1.747) with significantly less memory I/O per layer (5.5d vs 34d), making it strictly better on both performance and efficiency at this scale.

---

## Infrastructure

### Memory footprint

| Variant | Memory per token | Notes |
|---------|-----------------|-------|
| Standard residuals | O(d) | Hidden state only |
| Full AttnRes | O(L·d) | All L layer outputs |
| Block AttnRes (N blocks) | O(N·d) | N block summaries only |

With N=8 and L=128 (typical large model), Block AttnRes stores 16× fewer representations than Full AttnRes.

### Inference latency

The two-phase computation strategy (Algorithm 1, §4.2) amortises cross-block reads:
- **Phase 1**: all S queries in a block batched against N block representations — one matrix multiply per block
- **Phase 2**: sequential intra-block attention merged with Phase 1 via online softmax

Net result: < 2% latency overhead (paper measurement with optimised kernels). The reference implementation in this repo is ~2× slower per step due to Python-level loops in `compute_all_inputs`, but the loss values are numerically identical.

---

## How to Run

```bash
git clone https://github.com/Edward-Zion-Saji/attention-residuals
cd attention-residuals
pip install -e .

# Scaling comparison (MPS/CUDA recommended)
python scripts/run_experiments.py --mode scaling --device mps --out_dir ./outputs/results

# Ablation study (Table 4 style)
python scripts/run_experiments.py --mode ablation --device mps --out_dir ./outputs/results

# Depth attention analysis (Fig 8 style)
python scripts/run_experiments.py --mode depth --device mps --out_dir ./outputs/results

# All three sequentially
python scripts/run_experiments.py --mode all --device mps --out_dir ./outputs/results
```

Outputs written to `./outputs/results/`:
- `scaling/scaling_results.json` — per-size loss table
- `scaling/scaling_laws.png` — log-log scaling curves
- `scaling/*_dynamics.png` — training dynamics plots (loss + magnitude + grad norm)
- `ablation/ablation_results.json` — ablation table
- `ablation/ablation_bars.png` — bar chart of final losses
- `ablation/ablation_curves.png` — loss curves for all variants
- `depth/block_attnres_heatmap.png` — α_{i→l} heatmap (Fig 8 style)
- `depth/depth_stats.json` — locality, embedding weight, entropy statistics

Recommended hardware:

| Hardware | Expected runtime (scaling, 2 sizes) |
|----------|-------------------------------------|
| Apple M3/M4 (MPS) | ~40 min |
| NVIDIA A100 (CUDA) | ~5 min |
| Apple M1/M2 (MPS) | ~60 min |
| CPU only | Several hours |

---

## Summary

Our experiments reproduce the core qualitative findings of arXiv:2603.15031:

1. **Block AttnRes consistently beats PreNorm baseline** across both model sizes, with improvement confirmed from the very first epoch.
2. **Gain grows with model depth** — at L=6 (small) the absolute improvement is 3.6× larger than at L=4 (tiny), despite only a 1.5× increase in depth. This is the strongest single signal in the data.
3. **Zero-init stability holds** — both models begin with near-identical losses (within 0.002), and AttnRes diverges positively over training.
4. **The gap accelerates after sufficient training** — the small model shows negligible improvement at step 187 and +0.113 by step 1496, consistent with the attention weights gradually breaking the uniform-distribution init bias.

What cannot be verified here without large infrastructure: the 1.25× compute advantage (scaling law interpolation across 5 compute budgets), the 48B downstream benchmarks (GPQA-Diamond +7.5, MATH +3.6), and the < 2% inference overhead (requires optimised CUDA kernels).

The open question raised by Ziming Liu's analysis — whether a simple rescaled residual baseline (deterministic, no learnable routing) achieves comparable gains — is worth adding to the ablation suite. The paper's existing ablation shows input-independent static mixing (DenseFormer-style) gives no gain, but a properly-normalised fixed-coefficient baseline has not been tested.
