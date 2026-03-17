# Depth is Poorly Utilized in Transformers

This is one of the most productive research threads of the last two years — a convergent body of work from empirical, theoretical, and architectural angles all pointing at the same structural flaw. Here is a rigorous breakdown.

---

## 1. The Hypothesis: What "Dead Layers" Actually Means

The claim is precise, not vague. In a transformer with $L$ layers, the hidden state evolves as:

$$h_\ell = h_{\ell-1} + f_\ell(\text{LN}(h_{\ell-1}))$$

A layer is "dead" or redundant when $f_\ell(\cdot) \approx 0$ — i.e., the residual update is negligible relative to the running stream. The canonical measure is:

$$\text{Block Influence}_\ell = 1 - \mathbb{E}\left[\cos\!\text{sim}(h_{\ell-1},\ h_\ell)\right]$$

A value near 0 means the layer barely changed anything — the input and output are nearly identical vectors. Multiple groups independently arrived at this same metric.

---

## 2. The Diagnosis Papers (Measuring the Problem)

### **ShortGPT** — *Men et al., 2024* | ACL Findings 2025
[arXiv:2403.03853](https://arxiv.org/abs/2403.03853)

The first large-scale empirical confirmation. They compute Block Influence (BI) for each layer across LLaMA-2, Mistral, Baichuan, etc., and find a consistent pattern: **middle-to-late layers have near-zero BI**. Critically, they also derive *why*: in Pre-LN architectures, the residual stream norm grows as $O(\sqrt{L})$ while each layer's update norm is $O(1)$, so cosine similarity of $h_\ell$ and $h_{\ell+1}$ approaches 1 as depth increases — mathematically forced redundancy.

**Result:** Removing the bottom 25% of layers by BI (usually layers ~60–90% through the network) on LLaMA-2-13B drops MMLU by only 2.8 points while cutting 25% of parameters. Against four competing compression methods, ShortGPT retains 86–91% of performance where others retain 72–80%.

### **The Unreasonable Ineffectiveness of the Deeper Layers** — *Gromov et al., 2024* | ICLR 2025
[arXiv:2403.17887](https://arxiv.org/abs/2403.17887)

Uses **angular distance** between hidden states across a block of $n$ layers: $d(h^{(\ell)}, h^{(\ell+n)}) = \frac{1}{\pi}\arccos(\text{cosim})$. They find the optimal block to remove is consistently the deepest contiguous block, and the damage is repairable. The key finding: **up to 50% of layers can be pruned from LLaMA-2-70B** before QA benchmark collapse, and a small QLoRA fine-tuning pass ("healing") on ~164M tokens restores most performance — achievable on a single A100.

One sharp dissociation: **reasoning tasks (GSM8k, HellaSwag) degrade immediately** with any pruning, while knowledge QA survives heavy pruning. This implies depth matters for computation but not for fact storage.

### **The Curse of Depth** — *Sun et al., 2025* | NeurIPS 2025
[arXiv:2502.05795](https://arxiv.org/abs/2502.05795)

Provides the **mechanistic explanation** for why deep layers don't train properly in the first place. Pre-LN causes output variance to grow **exponentially** with depth. This forces the Jacobian $\partial h_L / \partial \theta_\ell \approx I$ for deep layers — the gradient through a deep layer is effectively the identity, meaning those layers receive near-zero *effective* gradient signal. They never learn.

Fix: **LayerNorm Scaling (LNS)** — scale each LayerNorm's output by $1/\sqrt{\ell}$, counteracting variance explosion. This makes deep layers train properly and consistently outperforms Pre-LN baselines from 130M to 7B parameters.

### **Do Language Models Use Their Depth Efficiently?** — *Csordás et al., 2025* | NeurIPS 2025
[arXiv:2505.13898](https://arxiv.org/abs/2505.13898)

The most rigorous *functional* analysis. They probe Llama 3.1, Qwen 3, and OLMo 2, asking whether depth creates **higher-order computation** or just spreads the same work thinner. Their findings:

- There is a sharp **phase transition at the network midpoint**: first-half layers build representations, second-half layers refine the output distribution for the *current* token.
- Second-half layers contribute much smaller norms to the residual stream.
- Skipping second-half layers has minimal effect on future computations.
- For multi-hop reasoning, there is **zero evidence of compositional use of depth** — computation depth doesn't scale with problem complexity.
- Linear mapping from shallow to deep model residual streams shows a diagonal pattern: deeper models just **"stretch" the same computations** over more layers, not deepen them.

This is perhaps the most damning finding. It's not just that layers are removable — it's that the model never learned to use them compositionally.

---

## 3. Why This Happens: The Pre-LN Dilution Chain

The three diagnosis papers converge on a single root cause:

```
Pre-LN architecture
  → residual stream norm grows O(√L) or faster
    → each new layer's contribution is proportionally diluted
      → deep layers need exponentially larger updates to matter
        → gradients through deep layers ≈ identity
          → deep layers don't receive effective training signal
            → they converge to near-identity transformations
              → cosine similarity ≈ 1, Block Influence ≈ 0
```

The structural irony: Pre-LN was adopted for training stability (it prevents gradient explosion), but the exact mechanism that stabilizes training also lobotomizes deep layers.

---

## 4. Measuring Layer Contribution: A Toolkit

| Metric | Computation | What it Captures |
|---|---|---|
| **Block Influence (BI)** | $1 - \mathbb{E}[\cos\text{sim}(h_{\ell-1}, h_\ell)]$ | Input-output vector similarity |
| **Angular Distance** | $\frac{1}{\pi}\arccos(\cos\text{sim})$ | Same, metrically nicer |
| **Relative Norm** | $\|a_\ell + m_\ell\|_2 / \|h_\ell\|_2$ | Update magnitude vs. stream magnitude |
| **CKA** | Kernel alignment between layer representations | Layer-level representational similarity |
| **LogitLens KL** | KL($p_\ell \| p_L$) where $p_\ell$ is intermediate prediction | How far each layer's prediction is from final |
| **Causal skip** | Drop layer $\ell$; measure downstream norm change | Actual causal influence on later computation |

These metrics all agree on the qualitative result. The cleanest for practical pruning is BI — cheap to compute over a small calibration set.

---

## 5. Prune / Reweight Dynamically — The Solution Landscape

### 5a. Static Pruning (Post-Training)

**ShortGPT:** Rank layers by BI, remove lowest-BI layers. No retraining. 25% compression with minimal loss.

**Gromov et al. + QLoRA healing:** Prune deepest contiguous block, then run ~164M tokens of QLoRA on a single GPU. Recovers most performance, even at 40–50% pruning for large models.

**LaCo** — *Yang et al., 2024* | EMNLP 2024 Findings: Rather than removing layers, **merge** similar adjacent layers by weight averaging. Information is consolidated rather than discarded, outperforming simple removal.

**LASER** — *Sharma et al., 2023* | ICLR 2024 ([arXiv:2312.13558](https://arxiv.org/abs/2312.13558)): Instead of removing whole layers, applies **SVD rank reduction** to specific weight matrices in specific MLP layers. The unintuitive finding: this can *improve* reasoning by up to +20 percentage points on QA benchmarks — the high-rank noise components in deep MLP weight matrices apparently interfere with fact retrieval.

### 5b. Training-Time Layer Dropout

**LayerDrop** — *Fan et al., 2019* | ICLR 2020 ([arXiv:1909.11556](https://arxiv.org/abs/1909.11556)): Drop entire layers stochastically during training. At inference, prune to any desired depth without retraining. A 24-layer model trained with LayerDrop pruned to 12 layers matches a model trained at 12 layers from scratch.

**LayerSkip** — *Elhoushi et al., 2024* | ACL 2024 ([arXiv:2404.16710](https://arxiv.org/abs/2404.16710)): Apply increasing dropout rates for deeper layers (low for early, high for late) + shared early-exit loss on a single LM head across all depths. At inference, use **self-speculative decoding**: generate drafts from early exits (e.g., layer 12 of 32), verify/correct with the full model using a shared KV cache. Speedups of **1.82–2.16×** on real tasks.

### 5c. Dynamic Inference-Time Skipping

**SkipDecode** — *Del Corro et al., 2023* ([arXiv:2307.02628](https://arxiv.org/abs/2307.02628)): Later tokens in a sequence need less processing (earlier tokens already did the heavy lifting). Skip middle layers for later tokens while always running the final layers. A monotonic exit-point constraint makes this compatible with batching and KV caching. **2–5× speedup** on OPT models.

**LazyLLM** — *Fu et al., 2024* ([arXiv:2407.14057](https://arxiv.org/abs/2407.14057)): Not all tokens need to go through all layers at every step. Select only important tokens per layer per generation step using attention scores. Pruned tokens can be revived in later steps. **2.34–3× speedup** in time-to-first-token on long-context tasks, no fine-tuning required.

### 5d. Dense Cross-Layer Aggregation

**DenseFormer** — *Pagliardini et al., 2024* | NeurIPS 2024 ([arXiv:2402.02622](https://arxiv.org/abs/2402.02622)): Instead of a fixed `h_ℓ = h_{ℓ-1} + f_ℓ(...)`, compute a **Depth-Weighted Average** across all prior layer outputs: $h_\ell = \sum_{k \leq \ell} \alpha_{k \to \ell} \cdot h_k$ with globally learned scalar weights. The learned weight patterns reveal that specific distant layers are highly reused — non-trivial long-range depth dependencies emerge. Reaches the same perplexity as deeper models with fewer parameters.

**Value Residual Learning / ResFormer** — *Zhou et al., 2024* | ACL 2025 ([arXiv:2410.17897](https://arxiv.org/abs/2410.17897)): A simpler version — add the **first layer's value vector** as a residual to every layer's attention output. Achieves equivalent loss with **16% fewer parameters and 20% less training data**. SVFormer (full shared K,V) halves the KV cache.

---

## 6. The Attention Residuals Paper — The Synthesis

**"Attention Residuals"** — *Kimi Team (MoonshotAI), March 2026*
[arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | GitHub: [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals) (937 ★)

This is the most conceptually complete response to the entire body of work above. Its central observation:

> *Just as Transformers replaced fixed recurrence over the time dimension with attention, Attention Residuals applies the same idea over the depth dimension.*

Standard residual connection:
$$h_\ell = h_{\ell-1} + f_\ell(\text{LN}(h_{\ell-1})) \quad \Leftrightarrow \quad h_\ell = \sum_{i<\ell} \text{(uniform weight)} \cdot \Delta h_i$$

This is depth-wise recurrence with fixed uniform aggregation — exactly the architecture Transformers defeated in the time dimension by replacing RNNs. The fix is the same one:

**AttnRes:**
$$h_\ell = \sum_{i=0}^{\ell-1} \alpha_{i \to \ell} \cdot v_i, \quad \alpha_{i \to \ell} = \text{softmax}\!\left(\frac{w_\ell^\top h_i}{\sqrt{d}}\right)$$

A single learned pseudo-query vector $w_\ell \in \mathbb{R}^d$ per layer computes input-dependent attention weights over all prior layer outputs. The model can now selectively "look back" to any earlier representation and de-emphasize uninformative intermediate layers.

### Block AttnRes (Practical Version)

Full AttnRes is $O(Ld)$ memory — incompatible with pipeline parallelism. The solution: partition $L$ layers into $N$ blocks ($N \approx 8$). Standard residuals run *within* each block; AttnRes attention operates *over block-level representations* only:

$$h_\ell = f_\ell\!\left(\sum_{j \leq \lfloor \ell / B \rfloor} \alpha_{j \to \ell} \cdot h_{jB}\right)$$

Memory cost: $O(Nd)$ instead of $O(Ld)$. Inference overhead: **< 2% latency**. This is what ships in production.

### Results

| Benchmark | Baseline | AttnRes | Δ |
|---|---|---|---|
| MMLU | 73.5 | 74.6 | +1.1 |
| **GPQA-Diamond** | 36.9 | **44.4** | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| Math | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |
| CMMLU | 82.0 | 82.9 | +0.9 |
| C-Eval | 79.6 | 82.5 | +2.9 |

Deployed in **Kimi Linear 48B** (MoE, 3B activated params, 1.4T tokens). The gains are largest on **multi-step reasoning and code** — exactly the tasks Csordás et al. showed require genuine depth utilization. Scaling law analysis: Block AttnRes matches a baseline trained with **1.25× more compute**.

The training dynamics directly confirm the diagnosis: output magnitudes stay bounded across depth (Pre-LN dilution is suppressed), and gradients distribute uniformly across all layers (the "curse of depth" is broken).

---

## 7. Unified Picture: What the Field Now Knows

```
PROBLEM STRUCTURE
─────────────────────────────────────────────────────────────
Pre-LN architecture
├── Causes residual stream norm growth → O(√L) or exponential
├── Each layer's ∆h contributes proportionally less with depth
│   └── Measured by: BI, angular distance, relative norm
├── Deep layers receive near-identity gradients (Curse of Depth)
│   └── They barely train → they become near-identity transforms
└── Deep models "stretch" not "deepen" computations (Csordás et al.)
    └── No compositional depth utilization observed

LAYER REDUNDANCY HIERARCHY (stronger → weaker effect of removal)
─────────────────────────────────────────────────────────────
Depth redundancy > Width/channel redundancy
Within depth: Attention sublayers > MLP sublayers
By position: Middle-to-late layers > Early and final layers
By task sensitivity: Reasoning > Knowledge QA > Perplexity

SOLUTION LANDSCAPE
─────────────────────────────────────────────────────────────
Post-training pruning
├── Static removal: ShortGPT (BI), Gromov+QLoRA healing
├── Weight merging: LaCo
└── Rank reduction: LASER (SVD on MLP weights)

Training-time solutions
├── Structured dropout: LayerDrop, LayerSkip
└── Normalization fix: LayerNorm Scaling (Curse of Depth paper)

Dynamic inference
├── Token-level: LazyLLM (per-layer token selection)
├── Layer-level: SkipDecode (monotonic exit points)
└── Self-speculative: LayerSkip (early-exit drafts)

Architectural redesign
├── DenseFormer: learned scalar weights over all prior layers
├── ResFormer: first-layer value residual
└── Attention Residuals (AttnRes): softmax attention over depth
    ← this is the most principled and complete solution
```

---

## 8. Open Questions and Implications

**Why is depth still useful at all if it's so inefficient?**
The Csordás et al. finding is key: second-half layers are doing *output distribution refinement*, not representation building. This is real work, just not compositional work. The question is whether the right architecture could do it in far fewer layers.

**Does AttnRes break the recurrence analogy entirely?**
Standard residuals are depth-wise RNNs with weight-tied transitions. AttnRes is depth-wise Transformer attention. The formal duality is exact: hidden state at depth $\ell$ playing the role of position, layer outputs playing the role of value vectors. The success of AttnRes suggests the Transformer inductive bias — "attend selectively over a sequence" — may be the right inductive bias for depth as well as sequence length.

**What's the right amount of depth?**
The pruning results suggest existing models are grossly over-deepened relative to their effective utilization. A 70B model with 50% of its layers prunable implies we are using roughly twice as many layers as needed — or equivalently, with a better architecture (AttnRes), the same model count could achieve much more.

**Will AttnRes change pretraining?**
The 1.25× compute efficiency improvement is significant — it means the same FLOP budget produces a better model. Given that frontier pretraining runs cost hundreds of millions of dollars, this is not a minor finding.

---

## Key Papers at a Glance

| Paper | Year | Venue | Core Contribution |
|---|---|---|---|
| [ShortGPT](https://arxiv.org/abs/2403.03853) | 2024 | ACL Findings 2025 | Block Influence metric; 25% pruning with 86-91% perf retained |
| [Unreasonable Ineffectiveness](https://arxiv.org/abs/2403.17887) | 2024 | ICLR 2025 | Up to 50% layer removal + QLoRA healing; task dissociation |
| [Curse of Depth](https://arxiv.org/abs/2502.05795) | 2025 | NeurIPS 2025 | Pre-LN variance explosion causes deep-layer gradient collapse; LNS fix |
| [Do LMs Use Depth?](https://arxiv.org/abs/2505.13898) | 2025 | NeurIPS 2025 | No compositional depth use; "stretch not deepen" finding |
| [LayerDrop](https://arxiv.org/abs/1909.11556) | 2019 | ICLR 2020 | Structured layer dropout for on-demand depth at inference |
| [LayerSkip](https://arxiv.org/abs/2404.16710) | 2024 | ACL 2024 | Early-exit training + self-speculative decoding; 2.16× speedup |
| [SkipDecode](https://arxiv.org/abs/2307.02628) | 2023 | preprint | Batch-compatible monotonic layer skipping; 2-5× speedup |
| [LASER](https://arxiv.org/abs/2312.13558) | 2023 | ICLR 2024 | SVD rank reduction improves reasoning without training |
| [DenseFormer](https://arxiv.org/abs/2402.02622) | 2024 | NeurIPS 2024 | Learned depth-weighted averaging over all prior layers |
| [ResFormer](https://arxiv.org/abs/2410.17897) | 2024 | ACL 2025 | First-layer value residual; 16% fewer params for same loss |
| [**Attention Residuals**](https://arxiv.org/abs/2603.15031) | **2026** | arXiv | **Softmax attention over depth replaces fixed residuals; +7.5 GPQA, 1.25× compute efficiency** |
