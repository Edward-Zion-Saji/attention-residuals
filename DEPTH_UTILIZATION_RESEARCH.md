# Are Transformer Layers Actually Doing Anything?

You've probably heard that scaling laws are about to hit a wall. One underappreciated reason why: a meaningful fraction of the layers in your favorite LLM might be doing almost nothing at all.

This isn't speculation. Over the last two years, a cluster of papers has converged on the same uncomfortable finding from different angles: depth is systematically underutilized in modern transformers. Layers that were trained, that cost compute to run, that sit in your model eating memory, are in many cases barely perturbing the hidden state at all. Some of the most interesting recent work in ML is about understanding why, measuring how bad it is, and finding ways to actually fix it.

Here is a walkthrough of the full research thread, ending with the most complete solution to date.

---

## What a "Dead" Layer Actually Means

To be precise about this, consider how a transformer layer works. The hidden state evolves as:

$$h_\ell = h_{\ell-1} + f_\ell(\text{LN}(h_{\ell-1}))$$

A layer is "dead" when $f_\ell(\cdot) \approx 0$, meaning its residual update is negligible relative to the stream it's writing into. The most intuitive way to measure this is cosine similarity between a layer's input and output. If the two vectors are nearly identical, the layer did almost nothing. Several groups independently landed on formalizing this as:

$$\text{Block Influence}_\ell = 1 - \mathbb{E}\left[\cos\!\text{sim}(h_{\ell-1},\ h_\ell)\right]$$

A Block Influence near zero is a dead layer. A value near one means the layer actually transformed the representation. When you plot this across a 32-layer model, what you see is not pretty.

---

## Four Papers That Nail the Diagnosis

### ShortGPT (Men et al., 2024) - [arXiv:2403.03853](https://arxiv.org/abs/2403.03853)

ShortGPT was the first large-scale confirmation. They computed Block Influence for every layer across LLaMA-2, Mistral, and Baichuan and found that **middle-to-late layers have near-zero BI** consistently across architectures. Then they did the obvious thing: removed them.

Cutting the bottom 25% of layers by BI from LLaMA-2-13B dropped MMLU by 2.8 points while removing 25% of parameters. Against four competing compression methods, ShortGPT retained 86-91% of performance where others managed 72-80%. The key layers being removed are usually from around 60% to 90% of network depth.

They also derived *why* this happens mathematically. In Pre-LN architectures, the residual stream norm grows as $O(\sqrt{L})$ while each layer's update norm stays $O(1)$. As depth grows, cosine similarity between consecutive hidden states is forced toward 1. The redundancy is not accidental. It is architecturally baked in.

### The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al., 2024) - [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)

Gromov et al. used **angular distance** between hidden states across a block of $n$ layers as their metric:

$$d(h^{(\ell)}, h^{(\ell+n)}) = \frac{1}{\pi}\arccos(\text{cosim})$$

Their finding was even more striking: **up to 50% of layers can be pruned from LLaMA-2-70B** before QA benchmark collapse. And a small QLoRA fine-tuning pass ("healing") on around 164M tokens on a single A100 restores most of the damage. The optimal block to remove is consistently the deepest contiguous block.

The dissociation they uncovered is the most important empirical result in the paper: reasoning tasks like GSM8k degrade immediately with any pruning, while factual QA survives enormous pruning. Depth matters for computation. It does not matter for fact storage.

### The Curse of Depth (Sun et al., 2025) - [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)

This paper provides the mechanistic explanation for why dead layers form during training in the first place. Pre-LN causes output variance to grow **exponentially** with depth. That forces the Jacobian $\partial h_L / \partial \theta_\ell \approx I$ for deep layers, meaning the gradient through a deep layer is effectively the identity. Deep layers receive near-zero effective gradient signal. They never had a chance to learn anything meaningful.

The fix proposed is **LayerNorm Scaling (LNS)**: scale each LayerNorm output by $1/\sqrt{\ell}$ to counteract the variance explosion. This consistently outperforms standard Pre-LN from 130M to 7B parameter models and results in deep layers that actually learn diverse, non-trivial representations.

### Do Language Models Use Their Depth Efficiently? (Csordás et al., 2025) - [arXiv:2505.13898](https://arxiv.org/abs/2505.13898)

The most rigorous functional analysis of the four. Csordás et al. probed Llama 3.1, Qwen 3, and OLMo 2, asking not just whether layers are removable but whether the model is doing anything compositionally sophisticated with depth.

The answer is no. There is a sharp **phase transition at the network midpoint**: first-half layers build representations, second-half layers refine the output distribution for the current token and basically nothing else. Skipping second-half layers has minimal effect on future computations. For multi-hop reasoning problems, there is zero evidence that computation depth scales with problem complexity. When you linearly map representations between a shallow and a deep model, you get a diagonal pattern. Deeper models **stretch** the same computations across more layers. They do not deepen them.

This is perhaps the sharpest version of the indictment. It is not just that layers are removable. It is that the model never learned to use depth compositionally in the first place.

---

## Why It Happens: The Pre-LN Dilution Chain

All four papers point at the same root cause:

```
Pre-LN architecture
  -> residual stream norm grows O(sqrt(L)) or faster
    -> each new layer's contribution is proportionally diluted
      -> deep layers need exponentially larger updates to matter
        -> gradients through deep layers approach identity
          -> deep layers receive near-zero effective training signal
            -> they converge to near-identity transformations
              -> cosine similarity approaches 1, Block Influence approaches 0
```

The structural irony is that Pre-LN was adopted precisely for training stability. It prevents gradient explosion. But the same mechanism that keeps gradients from exploding also starves deep layers of the signal they need to learn.

---

## How to Measure Layer Contribution

If you want to audit your own model, here are the metrics the literature has converged on:

| Metric | Computation | What It Captures |
|---|---|---|
| **Block Influence (BI)** | $1 - \mathbb{E}[\cos\text{sim}(h_{\ell-1}, h_\ell)]$ | Input-output vector similarity |
| **Angular Distance** | $\frac{1}{\pi}\arccos(\cos\text{sim})$ | Same thing, metrically cleaner |
| **Relative Norm** | $\|a_\ell + m_\ell\|_2 / \|h_\ell\|_2$ | Update magnitude vs. stream magnitude |
| **CKA** | Kernel alignment between layer representations | Layer-level representational similarity |
| **LogitLens KL** | KL($p_\ell \| p_L$) | How far each layer's prediction is from the final |
| **Causal skip** | Drop layer $\ell$; measure downstream norm change | Actual causal influence on later computation |

They all agree qualitatively. For practical pruning, Block Influence is the most convenient since it is cheap to compute over a small calibration set.

---

## The Solution Landscape

Once you know which layers are dead, what do you do about it? The literature has explored four distinct directions.

### Static Post-Training Pruning

The simplest approach: rank layers by Block Influence, remove the worst ones, optionally heal. ShortGPT does this with no retraining at all. Gromov et al. prune up to 50% of layers and recover with a QLoRA pass.

**LaCo** (Yang et al., 2024, EMNLP 2024) takes a softer approach: instead of removing layers, it *merges* similar adjacent layers by weight averaging. Information is consolidated rather than discarded, which outperforms simple removal in their experiments.

**LASER** (Sharma et al., 2023, ICLR 2024, [arXiv:2312.13558](https://arxiv.org/abs/2312.13558)) goes even further in a counterintuitive direction. Rather than removing whole layers, it applies SVD rank reduction to specific weight matrices inside specific MLP layers. The unintuitive result: this can *improve* reasoning by up to 20 percentage points on QA benchmarks. The high-rank noise components in deep MLP weight matrices apparently interfere with fact retrieval. Removing them exposes the low-rank signal underneath.

### Training-Time Layer Dropout

**LayerDrop** (Fan et al., 2019, ICLR 2020, [arXiv:1909.11556](https://arxiv.org/abs/1909.11556)) drops entire layers stochastically during training. At inference you can prune to any desired depth without retraining. A 24-layer model trained with LayerDrop and then pruned to 12 layers matches a model trained at 12 layers from scratch.

**LayerSkip** (Elhoushi et al., 2024, ACL 2024, [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)) applies increasing dropout rates to deeper layers and attaches a shared early-exit loss to a single LM head across all depths. At inference it enables **self-speculative decoding**: generate drafts from early exits, then verify and correct with the full model using a shared KV cache. This achieves 1.82-2.16x speedups on real tasks.

### Dynamic Inference-Time Skipping

**SkipDecode** (Del Corro et al., 2023, [arXiv:2307.02628](https://arxiv.org/abs/2307.02628)) exploits the observation that later tokens in a sequence need less processing than earlier ones. It skips middle layers for later tokens while always running the final layers, with a monotonic exit-point constraint that maintains compatibility with batching and KV caching. Result: 2-5x speedup on OPT models.

**LazyLLM** (Fu et al., 2024, [arXiv:2407.14057](https://arxiv.org/abs/2407.14057)) selects only important tokens per layer per generation step using attention scores. Pruned tokens can be revived in later steps. No fine-tuning required, and it achieves 2.34-3x speedup in time-to-first-token on long-context tasks.

### Dense Cross-Layer Aggregation

**DenseFormer** (Pagliardini et al., 2024, NeurIPS 2024, [arXiv:2402.02622](https://arxiv.org/abs/2402.02622)) replaces the standard residual with a **Depth-Weighted Average** over all prior layer outputs:

$$h_\ell = \sum_{k \leq \ell} \alpha_{k \to \ell} \cdot h_k$$

The weights are globally learned scalars. Analyzing them reveals non-trivial long-range depth dependencies: specific distant layers are heavily reused in structured patterns. DenseFormer reaches the same perplexity as deeper models with fewer parameters.

**ResFormer** (Zhou et al., 2024, ACL 2025, [arXiv:2410.17897](https://arxiv.org/abs/2410.17897)) is a simpler version of the same idea: add the first layer's value vector as a residual to every layer's attention output. This single change achieves equivalent loss with 16% fewer parameters and 20% less training data. Its sibling SVFormer shares a single K,V across all layers, cutting the KV cache roughly in half.

---

## Attention Residuals: The Synthesis

**"Attention Residuals"** by the Kimi Team at MoonshotAI (March 2026, [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) is the most conceptually complete answer to the entire body of work above.

The central observation is an elegant duality:

> Just as Transformers replaced fixed recurrence over the time dimension with attention, Attention Residuals applies the same shift over the depth dimension.

Standard residual connections are depth-wise RNNs with weight-tied transitions. Each layer just adds its output to a running sum with a fixed weight of one. This is exactly the architecture transformers defeated in the sequence dimension by replacing RNNs. The fix is the same one:

**AttnRes:**
$$h_\ell = \sum_{i=0}^{\ell-1} \alpha_{i \to \ell} \cdot v_i, \quad \alpha_{i \to \ell} = \text{softmax}\!\left(\frac{w_\ell^\top h_i}{\sqrt{d}}\right)$$

A single learned pseudo-query vector $w_\ell \in \mathbb{R}^d$ per layer computes input-dependent attention weights over all prior layer outputs. The model can now selectively look back to any earlier representation and down-weight uninformative intermediate layers. Standard residual connections are a special case where all weights are equal.

### Making It Practical: Block AttnRes

Full AttnRes has $O(Ld)$ memory, which is incompatible with pipeline parallelism at scale. Block AttnRes partitions the $L$ layers into $N$ blocks (around 8 works well). Standard residuals run within each block; attention operates only over the block-level representations:

$$h_\ell = f_\ell\!\left(\sum_{j \leq \lfloor \ell / B \rfloor} \alpha_{j \to \ell} \cdot h_{jB}\right)$$

Memory drops to $O(Nd)$ instead of $O(Ld)$. Inference overhead is under 2% latency. This is what ships in production at Kimi.

### Results

Block AttnRes was deployed in Kimi Linear 48B, a MoE model with 3B activated parameters trained on 1.4T tokens.

| Benchmark | Baseline | AttnRes | Gain |
|---|---|---|---|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | **44.4** | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| Math | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |
| CMMLU | 82.0 | 82.9 | +0.9 |
| C-Eval | 79.6 | 82.5 | +2.9 |

The gains are largest on multi-step reasoning and code. These are exactly the tasks that Csordás et al. showed require genuine depth utilization, and exactly where a fixed residual architecture fails to deliver it. Scaling law analysis puts Block AttnRes at matching a baseline trained with **1.25x more compute**.

Training dynamics confirm the diagnosis: output magnitudes stay bounded across depth, and gradients distribute uniformly across all layers. The Pre-LN dilution problem and the curse of depth are both directly addressed.

---

## The Full Picture

```
PROBLEM STRUCTURE
Pre-LN architecture
  -> Residual stream norm grows O(sqrt(L)) or exponential
  -> Each layer's contribution is proportionally diluted with depth
     (Measured by: BI, angular distance, relative norm)
  -> Deep layers receive near-identity gradients
     (They barely train and become near-identity transforms)
  -> Deep models "stretch" not "deepen" computations
     (No compositional depth utilization observed)

LAYER REDUNDANCY HIERARCHY
Depth redundancy > Width/channel redundancy
Within depth: Attention sublayers > MLP sublayers
By position: Middle-to-late layers > Early and final layers
By task sensitivity: Reasoning > Knowledge QA > Perplexity

SOLUTION LANDSCAPE
Post-training pruning
  -> Static removal: ShortGPT (BI), Gromov + QLoRA healing
  -> Weight merging: LaCo
  -> Rank reduction: LASER (SVD on MLP weights)

Training-time solutions
  -> Structured dropout: LayerDrop, LayerSkip
  -> Normalization fix: LayerNorm Scaling (Curse of Depth)

Dynamic inference
  -> Token-level: LazyLLM (per-layer token selection)
  -> Layer-level: SkipDecode (monotonic exit points)
  -> Self-speculative: LayerSkip (early-exit drafts)

Architectural redesign
  -> DenseFormer: learned scalar weights over all prior layers
  -> ResFormer: first-layer value residual
  -> Attention Residuals: softmax attention over depth  <-- most principled
```

---

## Open Questions

**Why is depth still useful at all if it is so inefficient?** Csordás et al. show that second-half layers do *output distribution refinement*, not representation building. That is real work, just not compositional work. The open question is whether the right architecture could do it in far fewer layers.

**Should depth routing be hard or soft?** Mixture of Depths ([arXiv:2404.02258](https://arxiv.org/abs/2404.02258)) does hard per-token routing: a router decides at each layer whether to compute it or bypass it entirely. AttnRes does soft routing: all layers run, but aggregation weights are learned and input-dependent. Hard routing saves compute at training. Soft routing is always differentiable. The likely ideal is training with soft routing until weights stabilize, then discretizing for inference.

**What is the right amount of depth?** A 70B model where 50% of layers are prunable implies something close to a 2x over-parameterization in the depth dimension. With AttnRes, those same parameter counts could be doing substantially more. We do not yet know what the Pareto frontier looks like for depth vs. width under a fixed compute budget once the architecture actually uses depth properly.

**Will AttnRes change pretraining norms?** 1.25x compute efficiency is not a minor finding at frontier scale. Pretraining runs at the frontier cost hundreds of millions of dollars. If AttnRes becomes standard, the baseline efficiency of every new run improves.

---

## Papers Referenced

| Paper | Year | Venue | Core Contribution |
|---|---|---|---|
| [ShortGPT](https://arxiv.org/abs/2403.03853) | 2024 | ACL Findings 2025 | Block Influence metric; 25% pruning with 86-91% performance retained |
| [Unreasonable Ineffectiveness](https://arxiv.org/abs/2403.17887) | 2024 | ICLR 2025 | Up to 50% layer removal + QLoRA healing; task dissociation |
| [Curse of Depth](https://arxiv.org/abs/2502.05795) | 2025 | NeurIPS 2025 | Pre-LN variance explosion causes deep-layer gradient collapse; LNS fix |
| [Do LMs Use Depth?](https://arxiv.org/abs/2505.13898) | 2025 | NeurIPS 2025 | No compositional depth use; stretch not deepen |
| [LayerDrop](https://arxiv.org/abs/1909.11556) | 2019 | ICLR 2020 | Structured layer dropout for on-demand depth at inference |
| [LayerSkip](https://arxiv.org/abs/2404.16710) | 2024 | ACL 2024 | Early-exit training + self-speculative decoding; 2.16x speedup |
| [SkipDecode](https://arxiv.org/abs/2307.02628) | 2023 | preprint | Batch-compatible monotonic layer skipping; 2-5x speedup |
| [LASER](https://arxiv.org/abs/2312.13558) | 2023 | ICLR 2024 | SVD rank reduction improves reasoning without training |
| [DenseFormer](https://arxiv.org/abs/2402.02622) | 2024 | NeurIPS 2024 | Learned depth-weighted averaging over all prior layers |
| [ResFormer](https://arxiv.org/abs/2410.17897) | 2024 | ACL 2025 | First-layer value residual; 16% fewer params for same loss |
| [**Attention Residuals**](https://arxiv.org/abs/2603.15031) | **2026** | arXiv | **Softmax attention over depth replaces fixed residuals; +7.5 GPQA, 1.25x compute efficiency** |
