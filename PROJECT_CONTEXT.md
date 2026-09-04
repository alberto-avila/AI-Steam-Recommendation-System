# PROJECT_CONTEXT.md — steam-genrec

Context document for Claude Code. Read this fully before writing any code.

---

## 1. What we're building

A scaled-down reimplementation of Netflix's **GenRec** paper (arXiv 2608.10257, "GenRec:
An LLM-Backed Recommendation Ranker at Netflix"), applied to **Steam game recommendations**.

The core idea from the paper: instead of hand-engineering hundreds of numeric features
about a user, **verbalize** their interaction history as natural-language text, feed it
through an LLM, and attach a lightweight **scoring head** that scores the entire catalog
in a single forward pass — not token-by-token generation.

We are replicating that architecture at small scale on a single consumer GPU.

**This is a portfolio project** for Data Science / AI roles. Code quality, a clear README,
and an honest evaluation table matter as much as model performance. Prefer readable,
well-commented code over clever code.

### Explicit non-goals for v1
- No production serving infra, no vLLM, no KV-cache optimization.
- No reinforcement learning / reward models (paper's Section on GRPO). Deferred.
- No cold-start handling for unseen games. **Documented as a known limitation.**
- No multimodal input. Gemma 4 is multimodal; we use text only.

---

## 2. Hardware & environment

- **Local NVIDIA GPU** (single card). Ask me for the exact VRAM before picking a model
  size, or read it from `nvidia-smi` and pick from the table in §3.
- Linux/WSL assumed. Python 3.11+.
- Use `uv` or plain `venv` — your call, but pin versions in `pyproject.toml` or
  `requirements.txt` so this is reproducible.
- Ollama is installed locally but is **NOT used for training** (see §3).

---

## 3. Model choice

**Backbone: Gemma 4, `E2B` or `E4B` variant, instruction-tuned.**

Gemma 4 is Apache 2.0 licensed, so this repo is safe to publish publicly.

### CRITICAL: do not try to fine-tune the Ollama copy
I have Gemma 4 pulled in Ollama. That copy is a quantized **GGUF** file intended for
inference only. QLoRA training requires:
- the original HuggingFace weights (`google/gemma-4-E2B-it` or `google/gemma-4-E4B-it`), and
- access to hidden states, which Ollama does not expose.

Download HF weights separately for training. Ollama can stay around for quick
manual sanity-checking of prompts, nothing else.

### Size selection by VRAM

| VRAM | Pick | Notes |
|---|---|---|
| 8–12 GB | Gemma 4 **E2B** | 2.3B effective params. Safe default. Short seq len. |
| 16 GB | Gemma 4 **E2B** or **E4B** | E4B needs careful batch/seq tuning. |
| 24 GB+ | Gemma 4 **E4B** | 4.5B effective params. Comfortable. |

Start with **E2B regardless** to get the pipeline working end-to-end, then scale up if
VRAM allows. Do not debug the pipeline on the bigger model.

### Training method
QLoRA (4-bit base + LoRA adapters) via `peft` + `bitsandbytes`, or **Unsloth** if it
supports the Gemma 4 architecture at the time of writing — check first, don't assume.
If Unsloth doesn't support it yet, fall back to plain `transformers` + `peft`.

Starting hyperparameters (tune from here):
- LoRA rank 16, alpha 32, dropout 0.05
- Target modules: attention q/k/v/o projections
- lr 2e-4, cosine schedule, bf16
- Gradient checkpointing ON, gradient accumulation to reach effective batch ~32
- Max sequence length: start at 512 tokens. Our verbalized histories are short.

---

## 4. Data

### 4.1 Primary: Steam-200k (user interactions)
Kaggle: `tamber/steam-video-games`

- ~200,000 rows, 12,393 unique users, 5,155 unique games.
- Columns: `user-id`, `game-title`, `behavior-name`, `value`
- `behavior-name` is either `purchase` or `play`
- For `purchase` rows, `value` is always 1 (carries no signal — drop these)
- For `play` rows, `value` = **hours played**

**Hours played is our engagement signal.** It's the analogue of Netflix's watch-duration
signal in the paper. This is the single most important field in the project.

### 4.2 Deferred to v2 (do NOT wire up now, just note in README)
- `nikdavis/steam-store-games` — ~27k games with genres, tags, categories. For enriching
  verbalization with item metadata.
- `forgemaster/steam-reviews-dataset` — has `voted_up` (explicit thumbs up/down) and real
  timestamps. Would enable proper recency-based context engineering.

### 4.3 Preprocessing rules
1. Keep only `behavior-name == "play"` rows with `value > 0`.
2. Drop users with fewer than **5** played games (too little signal to verbalize).
3. Build a stable `game_title -> game_id` integer mapping. **Persist this to disk** —
   the item embedding table is indexed by it and it must not change between runs.
4. Normalize game titles (strip whitespace, consistent casing) before mapping, since
   this dataset has some title inconsistencies.
5. Log dataset stats after filtering (n_users, n_games, n_interactions, hours
   distribution) and save to `data/processed/stats.json`.

### 4.4 Train/test split — IMPORTANT CAVEAT
Steam-200k has **no timestamps**. A proper recsys eval would use a temporal split
(train on past, predict future). We cannot. We use **random leave-one-out**: for each
user, hold out one played game as the test positive.

This is a real methodological weakness and it **must be stated plainly in the README**,
not hidden. It's the kind of thing an interviewer will probe, and knowing it is a
strength. Note that the v2 reviews dataset would fix this.

---

## 5. Architecture

Two-tower design mirroring the paper's scoring head:

```
verbalized user history  ──> Gemma 4 (QLoRA) ──> pooled hidden state  h   [d]
candidate game ids       ──> item embedding table ──> item vectors    e_i [n, d]

scores = h @ e_i.T          # dot product
loss   = cross_entropy(softmax(scores), index_of_true_game)
```

Key points:
- **Pooling**: mean-pool the last hidden layer over non-padding tokens, or use the final
  token's hidden state. Try mean-pooling first; make it a config flag so we can ablate.
- **Item tower is a plain learned `nn.Embedding`** over game IDs, trained from scratch
  jointly with the LoRA adapters. It is NOT text-encoded. This mirrors the paper and is
  what makes single-pass full-catalog scoring possible.
- **Projection**: if the LLM hidden dim doesn't match the chosen item embedding dim, add
  a single linear projection on `h`. Item embedding dim of 128–256 is plenty here.
- The LM head / token generation is **unused**. We never sample tokens. This is a
  prefill-only, encoder-style use of a decoder model.

### Negative sampling
For each training example: 1 held-out positive + **N sampled negatives** (start N=16)
drawn from games the user has not played. Sample negatives in proportion to popularity^0.75
rather than uniformly — uniform negatives make the task too easy and inflate metrics.

---

## 6. Verbalization (the "context engineering" step)

This is the conceptual heart of the project. Keep it in **one clearly-named module**
(`src/verbalize.py`) with the strategy swappable via config, because we will ablate it later.

**v1 strategy:** sort the user's games by hours descending, take top K (default 15),
render as text. Something like:

```
This player's most-played games:
Dota 2 (413 hrs), Team Fortress 2 (208 hrs), Counter-Strike (95 hrs), ...
```

Requirements:
- K must be a config parameter, not a magic number — we ablate it in v2.
- The held-out test game must NEVER appear in the verbalized history. **Write a unit
  test for this.** Leakage here would silently invalidate every result.
- Log the token-length distribution of rendered prompts. Prompt length is the cost axis
  in the paper and we'll want this data later.

---

## 7. Evaluation

Metrics on the held-out set:
- **Recall@10**
- **MRR** (mean reciprocal rank)

Evaluate by ranking the true positive against a fixed set of ~100 sampled negatives per
user (keep the negative set fixed across models with a seed, so comparisons are fair).

### Baselines — build these FIRST, before any LLM code
Both are short and they anchor whether the LLM is actually earning its complexity:
1. **Popularity**: rank by global play count. Ignores the user entirely.
2. **Implicit ALS matrix factorization** on the user × log(1+hours) matrix
   (`implicit` library). This is the real baseline to beat.

Output a comparison table to `results/comparison.md`. If GenRec loses to ALS, that's a
finding to report honestly, not a bug to hide. Say so in the README.

---

## 8. Repo structure

```
steam-genrec/
├── README.md
├── PROJECT_CONTEXT.md        # this file
├── pyproject.toml
├── config/
│   └── default.yaml          # all hyperparams, paths, K, N_negatives
├── data/
│   ├── raw/                  # gitignored
│   └── processed/            # gitignored
├── src/
│   ├── data.py               # load, filter, id mapping, splits
│   ├── verbalize.py          # history -> text
│   ├── model.py              # Gemma4 + pooling + item embedding + scoring head
│   ├── train.py              # QLoRA training loop
│   ├── evaluate.py           # Recall@10, MRR
│   └── baselines.py          # popularity, ALS
├── notebooks/
│   └── 01_eda.ipynb          # dataset exploration, hours distribution
├── tests/
│   └── test_leakage.py       # held-out game not in prompt
└── results/
    └── comparison.md
```

---

## 9. Build order

Work in this order and **stop for my review at each checkpoint**. Don't run ahead.

1. **Data pipeline** — load, filter, ID mapping, leave-one-out split. Print stats.
   *Checkpoint: stats look sane, ID mapping persisted.*
2. **Baselines** — popularity + ALS, with metrics printed.
   *Checkpoint: I have real numbers to beat.*
3. **Verbalizer** + leakage unit test + token length distribution.
   *Checkpoint: I read ~10 sample prompts myself and confirm they look right.*
4. **Model definition** — forward pass on a single dummy batch, shapes correct, no OOM.
   *Checkpoint: shapes and VRAM headroom confirmed.*
5. **Training loop** — overfit deliberately on 100 examples first to prove the loss goes
   down and the plumbing works. Only then train for real.
   *Checkpoint: loss decreases on the tiny set.*
6. **Full train + eval** — populate `results/comparison.md`.
7. **README** — architecture diagram, results table, limitations section.

---

## 10. Working style

- **Ask before installing anything heavy** or restructuring directories.
- Prefer standard, boring tools. No exotic dependencies.
- Every non-obvious design decision gets a one-line comment explaining *why*, tied back
  to the paper where relevant. This repo is a teaching artifact for me and a signal to
  interviewers.
- Seed everything (`torch`, `numpy`, `random`) and log the seed.
- If something in this document turns out to be wrong or impossible (e.g. a library
  doesn't support Gemma 4 yet), **stop and tell me** rather than silently substituting.
- I'm learning this material, not just shipping it. When you make a meaningful modeling
  choice, explain the tradeoff briefly in chat — don't just write the code.

---

## 11. Known limitations (must appear in README)

1. **Cold start**: the item embedding table is indexed by game ID and cannot score a game
   it never saw in training. Steam adds titles constantly. Deferred to v2 — planned
   research into content-based item towers and hybrid ID+text embeddings.
2. **No timestamps** in steam-200k, so the split is random leave-one-out rather than
   temporal. Results are optimistic relative to a realistic deployment.
3. **Hours played is a noisy proxy** for enjoyment — idle time, AFK farming, and games
   left running all inflate it. The paper has explicit feedback signals; we don't.
4. **Dataset is from ~2016** and covers 5,155 games, a small slice of Steam's catalog.
5. **No online evaluation.** The paper's headline claim rests on an A/B test; we only
   have offline metrics, which historically correlate imperfectly with online results.
