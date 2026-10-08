# Popularity and tuned ALS baselines

Both models use the same saved split and candidates. Test scores were evaluated only after validation selected ALS settings.

Seed: **42**. Selected ALS: **{'alpha': 10, 'factors': 32, 'iterations': 30, 'regularization': 0.01}**.

Full-catalog validation Recall@10 selects the winner; ties use MRR, then the fixed grid order.

| Evaluation | Population | Model | Users | Recall@10 | MRR |
| --- | --- | --- | ---: | ---: | ---: |
| full_catalog | all_users | popularity | 2,436 | 0.220033 | 0.124391 |
| full_catalog | all_users | als | 2,436 | 0.323892 | 0.174186 |
| full_catalog | supported_users | popularity | 2,411 | 0.222314 | 0.125681 |
| full_catalog | supported_users | als | 2,411 | 0.327250 | 0.175993 |
| sampled | all_users | popularity | 2,436 | 0.347291 | 0.181018 |
| sampled | all_users | als | 2,436 | 0.555008 | 0.299101 |
| sampled | supported_users | popularity | 2,411 | 0.350892 | 0.182895 |
| sampled | supported_users | als | 2,411 | 0.560763 | 0.302203 |

Coverage: **2,411/2,436 (98.97%)** test targets have training support.
Unsupported targets stay in all-user metrics with Recall@10=0 and reciprocal rank=0; their rank is unavailable, not last place.

## Protocol

- Full catalog means games represented in fitting data, excluding all of the user's known positives except the evaluated target.
- Sampled evaluation uses up to 100 unique alternatives weighted by fit-only popularity^0.75; candidates are fixed across models.
- Other holdout identities are used only to exclude known positives from candidate competitors; their hours and scores are not used to fit or select models.
- Score ties use ascending game ID. MRR uses the complete ranking, with no cutoff at ten.
- These are two different candidate universes: sampled scores must not be presented as full-catalog scores.

## Validation search

| Grid order | Factors | Regularization | Alpha | Recall@10 | MRR | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 32 | 0.01 | 10 | 0.298440 | 0.162375 | True |
| 1 | 32 | 0.01 | 40 | 0.271346 | 0.138179 | False |
| 2 | 32 | 0.1 | 10 | 0.296388 | 0.162120 | False |
| 3 | 32 | 0.1 | 40 | 0.280378 | 0.140861 | False |
| 4 | 64 | 0.01 | 10 | 0.270115 | 0.152923 | False |
| 5 | 64 | 0.01 | 40 | 0.256568 | 0.141662 | False |
| 6 | 64 | 0.1 | 10 | 0.274631 | 0.155788 | False |
| 7 | 64 | 0.1 | 40 | 0.257800 | 0.143366 | False |

## Interpretation and limits

ALS's full-catalog all-user Recall@10 differs from popularity by **+10.39 percentage points**.
A higher sampled score alone does not establish strong catalog-wide recommendations. No LLM result has been produced at this checkpoint.
Steam-200k has no timestamps: this random evaluation is not future-play prediction. Hours are noisy engagement proxies; the catalog is historical, short-history users are excluded, and no online evaluation is available.

## Reproducibility

Run `python -m src.baselines --config config/default.yaml`. Identical completed artifacts are reused; incompatible or modified artifacts are rejected.
Dependencies: `{"PyYAML": "6.0.3", "implicit": "0.7.3", "numpy": "2.2.6", "scipy": "1.15.3", "threadpoolctl": "3.6.0"}`.
Run identity SHA-256: `ada7fd069b3a33673b13523c7caa2ffb4ea464d717d635cb843bd358ae50ee78`.
Machine-readable aggregate results are in `metrics.json`; model files, candidate sets, validation membership, and per-user ranks are in the configured baseline artifacts directory.
