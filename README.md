# Few-shot prompting and fine-tuning for intent classification under label scarcity

Code and data for the master's thesis *Few-Shot Prompting and Fine-Tuning for Intent Classification of Support Emails under Label Scarcity: Accuracy, Calibration and Selective Prediction*, SRH Wilhelm Löhe University, 2026.

The study compares five systems on Banking77 under a frozen protocol in which every system receives the same labelled pool at each budget and is evaluated on the same test instances, and reports accuracy, calibration, selective prediction and out-of-scope discrimination together rather than accuracy alone.

## What is compared

| System | Paradigm | Use of the labelled pool |
|---|---|---|
| TF-IDF + logistic regression | Classical supervised baseline | Trains a linear classifier on sparse lexical features |
| SetFit (`paraphrase-mpnet-base-v2`) | Few-shot efficient fine-tuning | Contrastive fine-tuning on generated example pairs, then a classification head |
| RoBERTa-base | Conventional supervised fine-tuning | Updates all model weights |
| kNN, similarity-weighted | Non-parametric retrieval | Weighted vote over the 10 nearest neighbours |
| GPT-4o | Retrieval-based in-context learning | The same 10 neighbours placed in the prompt as demonstrations |

The budget defines the labelled **pool**, not the prompt. At every budget the prompted model and the fine-tuned models draw on exactly the same labelled examples and differ only in how they use them.

## Datasets

Neither dataset is included here; both are downloaded by the notebooks.

- **Banking77** — 13,083 customer-service queries over 77 intents, from the [PolyAI repository](https://github.com/PolyAI-LDN/task-specific-datasets). CC BY 4.0.
- **CLINC150** — 150 in-scope intents plus a dedicated out-of-scope split, from [clinc/oos-eval](https://github.com/clinc/oos-eval). CC BY 3.0.

## Repository layout

```
Google colab notebooks/     the nine notebooks, in run order
splits/        frozen split indices (splits.json, clinc_splits.json)
results/       result tables as CSV
figures/       the figures used in the thesis
probs/         saved per-query predictions and confidences (.npz)
gpt4o/         per-query API responses (.jsonl)
```

## Notebooks, in run order

| # | Notebook | What it does |
|---|---|---|
| 1 | `baseline_banking77.ipynb` | Loads Banking77, defines `sample_budget`, runs the TF-IDF baseline |
| 2 | `setfit_banking77.ipynb` | Creates the frozen splits (`splits.json`), reruns TF-IDF under the frozen protocol, runs the SetFit grid |
| 3 | `gpt4o_smoke_test.ipynb` | 50-query smoke test: retrieval, parse rate, latency and cost extrapolation |
| 4 | `gpt4o_full_grid.ipynb` | Zero-shot and few-shot GPT-4o across all budgets, with per-query resumability and a spend guard |
| 5 | `roberta_grid.ipynb` | RoBERTa under the fixed 1,500-step schedule |
| 6 | `Sanity check & KNN baseline.ipynb` | RoBERTa with validation-based early stopping; kNN over the same retrieved neighbours |
| 7 | `rq2_calibration_stage1.ipynb` | ECE, reliability diagrams, risk–coverage curves |
| 8 | `rq2_stage2_oos.ipynb` | Length-normalisation check; CLINC150 out-of-scope analysis and mixture sweep |
| 9 | `split_half_threshold_check.ipynb` | Split-half threshold selection, realised risk, split stability, temperature scaling |

Notebooks 1 and 2 must run before the others, because they create the frozen splits everything else depends on. Notebook 9 runs on the saved predictions in `probs/` and `gpt4o/` and needs no retraining or API calls.

## Reproducing the results

Notebooks 1, 2, 5, 6, 7 and 9 run without an API key. Notebooks 3, 4 and part of 8 call the OpenAI API and expect a key in the Colab secret `OPENAI_API_KEY`. Total API spend for the reported experiments was approximately 32 USD.

Fine-tuning was run on a T4 GPU in Google Colab. RoBERTa takes roughly 155 seconds per configuration on GPU; without one it takes several hours, so notebook 5 asserts that CUDA is available before starting.

Every sampling operation uses a fixed seed and the split indices are stored, so the labelled pools and the evaluation subsample can be regenerated exactly. The commercial model is a different case: requests were issued at temperature 0 against a pinned snapshot, but decoding from a hosted service is not deterministic and the snapshot may be withdrawn by the provider, so the prompting results are replicable in protocol rather than reproducible bit-for-bit.

## Key result files

| File | Contents |
|---|---|
| `results/results_all.csv` | Macro-F1 and accuracy per model, budget and seed |
| `results/split_half_coverage.csv` | Coverage at 1%, 2% and 5% risk — split-half, realised risk and oracle |
| `results/temperature_scaling_ece.csv` | ECE before and after temperature scaling, fitted on held-out predictions |
| `results/rq2_length_normalisation.csv` | Length-normalised against unnormalised confidence: AUROC, ECE, coverage |
| `results/rq2_oos_results.csv` | CLINC150 in-scope accuracy, AUROC and escalation share |

## Environment

Python 3.x on Google Colab, with `transformers`, `sentence-transformers`, `setfit`, `scikit-learn`, `openai`, `tiktoken`, `pandas`, `numpy` and `matplotlib`. The notebooks install what they need in their first cell.

## Licence
 
Code in this repository is released under the MIT Licence. The datasets retain their own licences, as noted above.
