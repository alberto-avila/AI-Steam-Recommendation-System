# steam-genrec

A small Steam recommendation research project exploring an LLM ranker over verbalized
play histories. The planned architecture and build checkpoints are in
[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

**Current status: data, baseline, verbalizer, and model forward-pass checkpoints
implemented.** Popularity and validation-tuned CPU ALS have measured results. Prompts
were reviewed. The Gemma 4 E2B ranker runs forward and backward on an 8 GB GPU. The
next checkpoint is the training loop; no LLM result exists yet.

Open [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) for the executed EDA walkthrough:
every filtering step, real removed/merged records, user-history and hours distributions,
persistent IDs, a user's exact holdout selection, and checks against the saved splits.

Open [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb) for the executed
baseline walkthrough, including confidence weights, validation search, fixed candidates,
reproduced test metrics, and recommendations for three deterministically selected users.
The complete comparison is in [`results/comparison.md`](results/comparison.md).

Open [`notebooks/03_verbalization.ipynb`](notebooks/03_verbalization.ipynb) for the
executed history-to-text walkthrough: source splits, each exclusion for one user,
top-K omissions, real token counts, and ten review prompts. The same prompts are in
[`results/verbalization/samples.md`](results/verbalization/samples.md).

## Set up and run

Python 3.11+ is required. Data preparation needs only PyYAML; it does not need a GPU,
PyTorch, Hugging Face weights, or Ollama. Dependencies are pinned in `pyproject.toml`.

From the repository root, create a local environment and install the project.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux / WSL:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Download [Steam Video Games by Tamber](https://www.kaggle.com/datasets/tamber/steam-video-games)
and extract `steam-200k.csv` to `data/raw/steam-200k.csv`. The initial run uses
[dataset version 3](https://www.kaggle.com/api/v1/datasets/download/tamber/steam-video-games?datasetVersionNumber=3).
The pipeline never downloads data implicitly. Raw data and generated outputs are
gitignored; the source dataset has its own license, listed on Kaggle.

Run preparation and tests on Windows:

```powershell
.\.venv\Scripts\python.exe -m src.data --config config/default.yaml
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_data.py" -v
```

With the environment activated on Linux / WSL:

```bash
python -m src.data --config config/default.yaml
python -m unittest discover -s tests -p "test_data.py" -v
```

Configure the seed, minimum distinct games per user, and paths in
[`config/default.yaml`](config/default.yaml). Paths are resolved relative to the
parent of the config directory, rather than the shell's current directory. Absolute
paths also work. The `baselines` and `evaluation` sections configure the new checkpoint;
the data pipeline continues to accept the original two-section configuration.

### Re-run the EDA notebook

The notebook includes saved tables and figures, so it can be reviewed immediately.
To execute it again, install the optional analysis dependencies into the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[eda]"
```

On Linux / WSL with the environment activated, use `python -m pip install -e ".[eda]"`.
Open `notebooks/01_eda.ipynb` in your notebook editor, select the project's `.venv`
interpreter, and run all cells. The notebook works from either the repository root or
the `notebooks/` directory. It reads the existing raw and processed data without
rewriting them, and fails if the recomputed rows do not match the saved artifacts.

## Train and evaluate the baselines

Windows PowerShell, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[baselines,eda]"
.\.venv\Scripts\python.exe -m src.baselines --config config/default.yaml
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_baselines.py" -v
```

On Linux / WSL, activate `.venv` and use `python` in place of the Windows executable.
The optional `baselines` group pins NumPy 2.2.6, SciPy 1.15.3, implicit 0.7.3, and
threadpoolctl 3.6.0. Training and scoring use CPU, float32, seed 42, and one numerical
thread. The `eda` group is only needed to execute the notebooks.

The runner performs a deterministic additional leave-one-out split of the original
training rows using seed 43: **52,902 development-fit rows** and **2,436 validation
positives**. It compares eight ALS configurations (32/64 factors, 0.01/0.1
regularization, 10/40 confidence alpha; 30 iterations), selecting full-catalog
validation Recall@10, then MRR, then fixed grid order. After selection it refits on
all **55,338 original training rows** and evaluates the original test split.

Popularity counts distinct training users per game. ALS uses binary preferences and
confidence **`1 + alpha * log1p(hours)`** for observed entries; missing interactions
have preference zero and confidence one in the solver. The library's own alpha is
fixed at one to avoid applying the confidence multiplier twice. Scores are dot
products, not predicted hours or calibrated probabilities.

### Test results

Headline results include all **2,436 users**, including 25 unsupported targets:

| Evaluation | Model | Recall@10 | MRR |
| --- | --- | ---: | ---: |
| Full eligible catalog | Popularity | 0.220033 | 0.124391 |
| Full eligible catalog | Tuned ALS | **0.323892** | **0.174186** |
| 100 sampled alternatives | Popularity | 0.347291 | 0.181018 |
| 100 sampled alternatives | Tuned ALS | **0.555008** | **0.299101** |

Validation selected **32 factors, regularization 0.01, alpha 10**, with 30 iterations.
ALS improves full-catalog Recall@10 by **10.39 percentage points** over popularity in
this run. These are the numbers the later LLM should be compared against using the
same evaluator; no LLM result is claimed yet.

The full eligible catalog contains **3,519 games with training interactions**, with
known played games removed for each user except the evaluated target. Sampled
evaluation uses 100 unique alternatives weighted by fitting popularity^0.75, fixed
across models. Other held-out game identities are used only to exclude known positives
from competitors; their hours and scores do not enter fitting or selection. Unobserved
alternatives are not explicit dislikes.

Coverage is **2,411/2,436 = 98.97%**. Unsupported targets receive zero Recall@10 and
reciprocal rank, with an unavailable rank rather than artificial credit for last
place. The complete report also shows supported-user metrics. MRR is untruncated;
score ties use ascending game ID. Sampled and full-catalog scores describe different
candidate universes and must not be mixed.

### Saved baseline artifacts

`data/baselines/` is gitignored and contains:

- `validation_fit.csv` and `validation.csv`: the additional split, preserving hours
  and original game IDs. The six original processed files are never rewritten.
- `validation_candidates.jsonl` and `test_candidates.jsonl`: per-user sampled game
  IDs, target ID, and support flag. Unsupported targets are recorded but are not
  included among the eligible candidate IDs.
- `user_mapping.json`, `popularity.npy`, and `als.npz`: stable user indices and
  pickle-free final model arrays. `validation_popularity.npy` records fit-only
  frequencies for auditing the validation candidates.
- `validation_search.csv`, `per_user.csv`, and `metrics.json`: every trial, the selected
  models' validation/test ranks under both protocols, and aggregate results.
- `manifest.json`: input/code/artifact SHA-256 hashes, configuration, random seeds,
  dependency versions, and numerical-library details.

Portable aggregate results are also written to `results/metrics.json` and the readable
comparison to `results/comparison.md`. Repeating the command verifies and reuses a
compatible completed run. Changed inputs, settings, code, environment, or modified
artifacts are rejected; choose a new `baselines.artifacts_dir` to make a fresh run.
The run publishes its artifact directory only after training and verification succeed.

Both models implement `score(user_id, game_ids)`. `src.evaluate` owns candidate
generation, common tie handling, Recall@10, and MRR so future models can use the same
protocol. The baseline notebook loads these modules and saved factors, verifies every
test rank, and does not retrain models. The original EDA notebook remains compatible.

## Prepare and review the verbalizer

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[verbalizer,eda]"
.\.venv\Scripts\python.exe -m src.verbalize --config config/verbalize.yaml --download-tokenizer
```

On Linux / WSL, use `python` in place of the Windows executable. The optional
`verbalizer` group pins `tokenizers==0.23.2`. The explicit download flag fetches only
`tokenizer.json` and `tokenizer_config.json` (about 32 MB), with pinned checksums, from
[Google's Gemma 4 E2B tokenizer](https://huggingface.co/google/gemma-4-E2B-it/tree/3e22461f65e89153144f8adb70e3b8c2cc9845a7).
No model weights or PyTorch are required. Omit `--download-tokenizer` on subsequent,
offline runs; missing files produce an error rather than an implicit download.

[`config/verbalize.yaml`](config/verbalize.yaml) references the existing pipeline
configuration and holds the strategy, **K=15**, 512-token budget, review sample count,
paths, and tokenizer revision. Keeping this checkpoint's settings separate preserves
the verified baseline configuration and code identity. Paths use the same project-root
resolution as the earlier configs.

`src.verbalize.render_history` removes supplied held-out game IDs, sorts remaining
games by hours descending (ties by persistent game ID), takes K, and renders readable
titles plus raw hours. The strategy is registered by name so later ablations can reuse
the split, tokenizer, and reporting code. Short sessions keep their fractional hours.
Game IDs and user IDs are audit metadata, not prompt text.

- **Validation contexts:** use the saved 52,902 development-fit rows; exclude both
  validation and test targets.
- **Test contexts:** use the original 55,338 training rows; exclude test targets.
  Validation interactions are allowed again after refitting, as in the ALS checkpoint.

The saved tokenizer does not automatically add BOS. The explicit future model-input
contract is **one BOS token followed by plain history text**, with no EOS, chat template,
generation prefix, padding, or token truncation. These are real tokenizer counts,
not word-count estimates. Over-budget prompts are flagged; the notebook reports them.
Changing this encoding contract requires remeasuring lengths at the model checkpoint.

| Context | Users | Median input tokens | p99 | Maximum | Over 512 | Histories capped by K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 2,436 | 116 | 188.65 | 236 | 0 | 886 |
| Test | 2,436 | 126.5 | 189 | 236 | 0 | 940 |

At test time, K=15 retains **25,628** of 55,338 history interactions and omits **29,710**.
It caps **38.59%** of users' histories. This prioritizes sustained engagement but can
lose lower-hour interests; K=15 is a starting choice, not a demonstrated optimum.
Any later K selection should use validation recommendation metrics. Token headroom
alone does not establish better recommendation quality.

A second registered strategy, `top_hours_summary`
([`config/verbalize_summary.yaml`](config/verbalize_summary.yaml)), prepends three
lines that restore whole-history signal lost to top-K: the number of games, total
hours, and the share of playtime in the single most-played game (breadth vs. loyalty).
All three are computed after held-out games are removed, and a leakage test covers
this. It writes to `data/verbalized_summary/` and `results/verbalization_summary/`.
Median input length rises from 126.5 to 158 test tokens (max 277; none over 512).
Which strategy ranks better is an ablation to settle on validation metrics.
Genre lines are deferred: Steam-200k has no genres, and joining store metadata is v2.

`data/verbalized/` stores validation/test JSONL contexts, selected IDs, audit-only targets,
source and omission counts, token lengths, budget flags, statistics, and the sample report.
Its manifest records input and code hashes, configuration, seed, dependency versions,
tokenizer identity, and artifact checksums. Existing mismatched caches are rejected;
choose a new artifact directory for changed settings. Original data and baseline files
remain read-only. The small review report and statistics are copied to
`results/verbalization/`; large generated files and tokenizer assets are gitignored.

The ten review users span history lengths, including the smallest and largest, with
seed-derived tie ordering. They are not selected by model success or test scores.
**Review their prompts before moving to the model forward-pass checkpoint.** These
artifacts are evaluation contexts; supervised training examples will additionally
need their own positive removed from the context before rendering.

To run the complete suite after installing all checkpoint dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[baselines,verbalizer,eda]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The 49 tests include explicit target-text leakage, summary-statistic leakage, normalized identity exclusion,
equal-hour ordering, fractional hours, deterministic review samples, a hand-calculated
token encoding, no silent truncation, both context populations, fresh-run consistency,
cache rejection, and the earlier data/baseline checks. The executed third notebook
also reconstructs all **4,872 prompts** with the actual pinned tokenizer and checks
that all 20 existing data, baseline, and result files remain byte-identical.

## Model definition (forward-pass checkpoint)

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[model]"
.\.venv\Scripts\python.exe -m src.model --config config/model.yaml --smoke
```

The CUDA 12.8 build of PyTorch is required for RTX 50-series (Blackwell, sm_120) GPUs.
The first run downloads the Gemma 4 E2B weights (10.2 GB) at the commit pinned for the
tokenizer, so token IDs and weights always match.

[`src/model.py`](src/model.py) mirrors the paper's scoring head:

```
verbalized history -> Gemma 4 E2B (4-bit NF4 + LoRA) -> mean-pooled h [1536] -> Linear -> u [128]
game IDs           -> nn.Embedding(3542, 128)                                  -> e [3542, 128]
scores = u @ e.T   (whole catalog, one prefill pass, no token generation)
```

- **Text-only backbone.** Gemma 4 is multimodal; only `model.language_model.*` is loaded.
  Vision and audio towers are dropped.
- **Pruned vocabulary.** E2B keeps about 2.3B of its parameters in embedding tables
  (262,144 tokens × 35 per-layer embeddings). 4-bit quantization does not compress
  embeddings, and PEFT's k-bit preparation upcasts them to fp32: the standard load needs
  **11.15 GB** of weights on an 8 GB card. We never generate text, so only rows for
  tokens our prompts use are kept (**5,968 tokens**, from every prompt plus each title
  in list context). Weights drop to **1.17 GB**. Hidden states are **bit-identical** to the full-vocabulary
  model (max absolute difference 0.0 on real prompts). Unknown tokens raise an error
  rather than being silently mapped.
- **Trainable parameters:** 5.36M LoRA (rank 16, q/k/v/o), 0.20M projection, 0.45M item
  embeddings; 1.94B frozen. The item tower is an ID embedding trained from scratch, as in
  the paper, so it cannot score unseen games (see limitations).
- **Pooling** is a config flag (`mean` | `last`) for later ablation; right padding, fp32 head.

Smoke test on an RTX 5060 Laptop (8 GB): longest prompts (277 tokens), forward + backward
with gradient checkpointing. Gradients are nonzero for LoRA, projection, and item
embeddings, and absent from all frozen parameters. Full report: [`results/model/smoke.json`](results/model/smoke.json).

| Batch size | Peak GPU memory | Seconds / step |
| ---: | ---: | ---: |
| 4 | 1.91 GB | 1.7 |
| 8 | 2.63 GB | 2.1 |
| 16 | 4.07 GB | 4.4 |
| 32 | 6.95 GB | 8.8 |

Planned training setting: batch 16 × 2 accumulation steps (effective 32), leaving
headroom for the display and longer training prompts. `tests/test_model.py` adds
CPU-only checks for pooling, padding invariance, vocabulary remapping, candidate/catalog
score agreement, and frozen-parameter gradients (59 tests total).

## Preparation rules

1. Read the original headerless CSV as `user-id, game-title, behavior-name, value`.
   Ignore its unused fifth field. Four-column files and an optional header using
   those exact field names are also accepted. User IDs remain strings to avoid
   precision loss when loading large identifiers.
2. Keep `play` rows with finite hours greater than zero. Drop purchases and nonpositive
   playtime; fail with an error on malformed rows, unknown behaviors, or nonfinite hours.
3. Normalize titles with Unicode NFKC, collapsed whitespace, and case folding. Preserve
   punctuation and retain a readable display title separately.
4. Merge repeated normalized `(user_id, game_title)` pairs using **maximum hours**.
   These are accumulated playtime snapshots; summing would double-count engagement.
   Deduplication happens before filtering and splitting so a game cannot leak through
   a duplicate row. Conflicting snapshots have no timestamps to identify the latest one.
5. Retain users with at least **5 distinct played games**, measured before holding out
   a test game. Thus each retained user has at least four training interactions.
6. Assign zero-based game IDs in sorted normalized-title order on the first run. Reuse
   `game_mapping.json` on subsequent runs, appending new titles in sorted order and
   preserving IDs for missing titles. `mapping_size` can consequently exceed `n_games`
   on a changed dataset. `games.csv` marks currently present titles with `is_active=1`.
   Preserve the mapping with model checkpoints: deleting it after catalog changes can
   invalidate the correspondence between IDs and embedding rows.
7. Select one test game uniformly per user, using a seed derived from the configured
   seed and user ID over sorted titles. The split is independent of input row order
   and of unrelated users being added or removed. This stage only uses Python's local
   random generator; it does not import NumPy or PyTorch.

The item mapping covers the entire filtered catalog, including held-out items. This
defines item identity, not a fitted popularity or engagement feature. **Fit popularity,
ALS weights, negative-sampling frequencies, and user histories from `train.csv` only.**
Do not use `interactions.csv` as training input. Later negative sampling must exclude
all known played games, including the test positive, from the negative pool.

## Output contract

All artifacts are written under `data/processed/` by default.

| File | Contents |
| --- | --- |
| `interactions.csv` | All retained positive interactions; inspection and split auditing only |
| `train.csv` | Remaining user histories after holding out one game per user |
| `test.csv` | Exactly one held-out positive per retained user |
| `game_mapping.json` | Persistent normalized `game_title -> game_id` mapping |
| `games.csv` | `game_id, game_title, display_title, is_active` lookup |
| `stats.json` | Counts, filtering audit, hours and history-length distributions, seed, source and mapping SHA-256 hashes, split counts, and games absent from training |

The three interaction CSVs have the same schema:
`user_id, game_id, game_title, hours`. Hours remain in their original units; the ALS
baseline computes its logarithmic confidence weights from fitting data itself.

The tests check duplicate handling, distinct-game filtering, stable mappings, invalid
inputs, split disjointness and completeness, exactly one test row per user, and
identical output bytes on repeated runs with unchanged data, mapping, and config.
Prompt-level leakage and tokenizer-length checks are implemented in the verbalizer stage.

Baseline tests add hand-calculated popularity/confidence checks, validation isolation,
candidate reproducibility and exclusion, ranking ties and metric boundaries,
unsupported targets, factor persistence, fresh-run reproducibility, and cache rejection.

## Initial data checkpoint

Measured on version 3 with seed **42** and minimum games **5**:

| Statistic | Value |
| --- | ---: |
| Raw rows | 200,000 |
| Purchase rows dropped | 129,511 |
| Play rows before deduplication | 70,489 |
| Duplicate play rows merged | 15 |
| Retained users | 2,436 |
| Retained games / persisted game IDs | 3,542 |
| Retained interactions | 57,774 |
| Training interactions | 55,338 |
| Test positives | 2,436 |
| Median hours per retained interaction | 4.4 |
| Mean hours per retained interaction | 40.59 |
| 99th percentile hours | 730.54 |
| Maximum hours | 10,442 |
| Test positives for games absent from training | 25 users / 23 games |

The mean far exceeds the median: engagement is strongly skewed. The data pipeline
keeps raw hours; the ALS baseline applies its logarithmic confidence transform.

The 25 positives with no training interactions remain in the test set and are explicitly
reported in `stats.json`. Resampling them would change the specified random evaluation.
They count as misses in the baseline headline metrics, with the supported-user subset
reported alongside it; the mapping alone does not provide a learned preference.

## Known limitations and deferred work

- **No timestamps:** random leave-one-out is not temporal evaluation. It can use games
  played after the held-out game, so results are optimistic relative to predicting
  future behavior in deployment.
- **Cold start:** the planned ID embedding cannot represent a new game outside its
  catalog, and games with no positive training interactions lack that preference
  signal. Content-based and hybrid item towers are deferred to v2.
- **Noisy engagement:** idle time, AFK farming, and games left open inflate hours. Hours
  are an engagement proxy, not explicit enjoyment labels.
- **History compression:** the verbalizer keeps the top 15 games by hours, so lower-hour
  interests can disappear. Titles without metadata can be ambiguous; this checkpoint
  measures input cost and isolation, not whether language-model ranking improves.
- **Historical, small catalog:** Steam-200k is the historical dataset described in the
  project context as circa 2016. The raw catalog has about 5,155 titles, with fewer
  remaining after play and user filtering; it is not a current Steam catalog.
- **Offline evaluation only:** the reported Recall@10 and MRR do not establish online
  recommendation quality. A small validation search is not exhaustive tuning, and
  results from one random split do not establish statistical significance.
- **Deferred datasets:** `nikdavis/steam-store-games` could add genres/tags, and
  `forgemaster/steam-reviews-dataset` could add explicit feedback and timestamps.
  Neither is connected in v1. Timestamped reviews would enable a temporal protocol.
