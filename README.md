# Speculative Decoding from Scratch

A from-scratch implementation of speculative decoding in Python, based on:

> **Fast Inference from Transformers via Speculative Decoding**
> Leviathan, Kalman, Matias — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

No existing speculative decoding libraries were used. The algorithm is implemented directly using PyTorch and HuggingFace Transformers, including the probabilistic rejection criterion and residual resampling described in the paper.

---

## What this is

Standard autoregressive generation calls the target model once per token, which is slow for large models. Speculative decoding takes advantage of the fact that a transformer can score multiple tokens in a single forward pass just as cheaply as it scores one. The way this works is by running a small, fast **draft model** autoregressively to propose `k` candidate tokens, then run the **target model** once over the full extended sequence. In one target forward pass you get probability scores for all `k` positions simultaneously. A rejection sampling step then decides which tokens to keep based on the probability scores of both models, in a way that preserves the exact output distribution of the target model and no quality loss.

The expected tokens produced per target call is `α·k + 1`, where `α` is the per-token acceptance rate. When the draft model is much faster than the target, this translates directly into wall-clock speedup.

---

## How it works

The implementation has three core functions and a main loop.

**`generate_draft(draft_model, input_ids, k)`**
Runs the draft model autoregressively `k` times. For each step, it samples a token via `torch.multinomial` and records both the scalar probability of the chosen token and the full vocabulary distribution. Both are needed later: the scalar for the acceptance test, the full distribution for residual resampling on rejection.

**`verify(target_model, input_ids_with_draft, prompt_len)`**
Runs the target model once on the prompt concatenated with all `k` draft tokens. Slices `logits[:, prompt_len-1:-1, :]` to extract the target's predicted distribution at each of the `k` draft positions in one shot. Also captures `logits[:, -1, :]` as the bonus distribution used when all `k` tokens are accepted.

**`rejection_sample(target_probs, draft_probs, draft_probs_full, draft_tokens)`**
Iterates through the `k` draft tokens in order. For each token `x_i`:

- Compute `α = min(1, p(x_i) / q(x_i))` where `p` is the target probability and `q` is the draft probability for the sampled token.
- Accept with probability `α` using Bernoulli Sampling (`random.random() < α`). This is the approach from the paper and was used because a deterministic threshold would bias the output distribution.
- On the first rejection, resample from the residual distribution: `normalize(clamp(p - q, 0))`. This corrects for the mass already covered by the draft, ensuring the final distribution matches the target exactly.
- Break as all tokens after the first rejection are discarded.

**Main loop in `speculative_decoding`**
Each iteration: generate `k` draft tokens → verify with target → rejection sample → append accepted tokens + one resampled or bonus token → repeat until `max_tokens` reached. The bonus token (sampled from the target's distribution at position `k+1`) is what enables `k+1` tokens from a single target pass in the best case.

---

## Experiments

Hardware: Apple M4 MacBook Air (10-Core CPU/10-Core GPU) 16GB RAM, MPS. 50 tokens generated per run, 2 runs per configuration.

### Acceptance rate by prompt type (k=4)

Acceptance rate measures how often the target model agrees with the draft model's token choices. Higher is better as it means more tokens per target call.

| Prompt type | Example | GPT-2 → GPT-2 M | Qwen2-0.5B → 1.5B | Qwen3.5-0.8B → 2B | Avg |
|---|---|---|---|---|---|
| code | `def fibonacci(n):` | 44.3% | 77.4% | 83.7% | **68.5%** |
| repetitive | `1, 2, 3, 4, 5, 6, 7, 8, 9,` | 59.9% | 68.5% | 60.8% | **63.1%** |
| factual | `The theory of relativity states that` | 43.7% | 39.5% | 50.4% | **44.5%** |
| conversational | `Why did the chicken cross the road` | 52.2% | 36.6% | 40.3% | **43.0%** |

Code prompts have the highest acceptance because they constrain the output heavily. The next token after `def fibonacci(n):` is almost certainly `\n` or a tab, so draft and target converge. Repetitive sequences (numerical lists) are similarly constrained: the target and draft both assign most of their probability mass to the same small set of next tokens. Conversational prompts have a wide range of plausible continuations, so the draft diverges from the target more often.

The Qwen model pairs significantly outperform GPT-2 on code and repetitive prompts, likely because the 0.5B and 1.5B (or 0.8B and 2B) Qwen checkpoints are trained on the same data distribution and share a more consistent style of output. GPT-2 and GPT-2 Medium were trained independently and differ more in their distributions.

### k ablation (conversational prompt: "Why did the chicken cross the road", averaged across all three model pairs)

| k | Avg acceptance | Actual speedup | Theoretical speedup (c=10) |
|---|---|---|---|
| 1 | 73.2% | 0.68x | 1.57x |
| 2 | 70.6% | 0.55x | 2.01x |
| 4 | 50.7% | 0.59x | 2.16x |
| 6 | 36.3% | 0.44x | 1.99x |
| 8 | 24.6% | 0.33x | 1.65x |

Theoretical speedup assumes the draft model is 10× faster than the target (see section below). Formula: `(α·k + 1) / (k/c + 1)` where `α` is acceptance rate, `k` is draft length, and `c` is the draft-to-target speed ratio.

The tradeoff is clear: larger `k` means more tokens per round if everything is accepted, but acceptance rate declines as `k` increases because each additional draft token is a new chance to diverge. This compounding effect is visible in the data: k=1 achieves 73.2% acceptance because the draft only needs to predict one token in isolation, while k=8 drops to 24.6% because each token is conditioned on previous draft tokens the target may have already disagreed with. The optimum given these results is around `k=4`, which balances the reduction against the cost of wasted draft work. In practice, the right `k` depends on the model pair's alignment and the speed ratio.

---

## Why there is no wall-clock speedup

All runs measured actual speedup below 1.0x, meaning speculative decoding was slower than running the target model alone. This is expected given the hardware setup.

**The draft model is not faster.** On Apple Silicon with MPS, a 0.5B and 1.5B model running on unified memory produce tokens at similar rates. There is no separate VRAM pool, no bandwidth bottleneck between devices, and no speed gap that speculative decoding can exploit. Both models access the same shared memory at similar throughput, so `k` draft passes add overhead rather than saving time.

**Both models run sequentially on the same device.** Speculative decoding is designed for setups where the draft runs on a separate, faster device (or on CPU) while the large target runs on GPU. In that configuration, draft latency is partially hidden. Here, they share one device and run one after the other.

**What the numbers would look like on appropriate hardware.** To get a sense of what these acceptance rates mean in practice, assume a 10× speed ratio between draft and target which is roughly what you'd see with a 1B draft and 70B target on a multi-GPU server. Plugging the actual acceptance rates from these runs into `(α·k + 1) / (k/c + 1)` with `k=4` and `c=10`:

| Model pair | Avg acceptance (k=4, conversational) | Theoretical speedup at c=10 |
|---|---|---|
| GPT-2 → GPT-2 Medium | 52.2% | 2.21x |
| Qwen2-0.5B → Qwen2-1.5B | 36.6% | 1.76x |
| Qwen3.5-0.8B → Qwen3.5-2B | 40.3% | 1.87x |

The acceptance rates indicate that the issues comes from the hardware and not the algorithm or implementation. The missing piece is a target model large enough that each forward pass is the bottleneck.

---

## How to run

```bash
git clone https://github.com/ammarhassona/speculative-decoding-from-scratch
cd speculative-decoding
python -m venv venv && source venv/bin/activate
pip install torch transformers accelerate
```

Edit [main.py](main.py) to select a model pair (three options are commented in), then:

```bash
python main.py
```

The script runs acceptance rate by prompt type (k=4) and a k ablation study, writing results to a text file named after the model pair.

To switch model pairs, uncomment the relevant block in [main.py](main.py):

```python
# GPT-2 → GPT-2 Medium
draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
target_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Qwen2-0.5B → Qwen2-1.5B
# draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)
# target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B").to(device)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Qwen3.5-0.8B → Qwen3.5-2B
# draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B").to(device)
# target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B").to(device)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
```

You can also use any model you like if it can be imported from transformers library and from huggingface. 

**Draft and target models must share the same tokenizer vocabulary.**

Change `device = "mps"` to `"cuda"` or `"cpu"` as needed.

---

## Reference

```
@misc{leviathan2022fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Yaniv Leviathan and Matan Kalman and Yossi Matias},
  year={2022},
  eprint={2211.17192},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## Study Resources I Used
- https://www.youtube.com/watch?v=VkWlLSTdHs8
- https://www.adaptive-ml.com/post/speculative-decoding-visualized
- https://www.youtube.com/watch?v=p23SblAIoXc
- https://arxiv.org/pdf/2211.17192
