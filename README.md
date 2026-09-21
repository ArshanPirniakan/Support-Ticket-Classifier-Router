# Support Ticket Classifier & Router

A single-file NLP project that classifies customer-support messages into intents with a fine-tuned
transformer, compares that transformer against a classical TF-IDF baseline on the same test set, and
then routes each ticket to a department and priority using a transparent deterministic rule layer.

Example input:

```
I've been charged twice for my subscription and nobody has responded.
```

Example output (intent is predicted by the model, department and priority come from rules):

```
predicted intent    : bill_balance (learned)
confidence          : 0.9241
suggested department: Billing (rule-based)
suggested priority  : P1 - High (rule-based)
routing reasons     : intent 'bill_balance' is in the payment/logistics intent list; payment risk keyword; unanswered follow-up
```

The confidence value above is illustrative formatting only; real values come from your own trained model.

---

## Why this project exists

Support desks need two different things from an automated triage system:

1. **Understanding what the customer is asking about.** This is a genuine NLP problem and is solved
   here with supervised multiclass text classification.
2. **Deciding who handles it and how urgent it is.** In most real organizations this is a policy
   decision, not a statistical one. It changes when the org chart or the SLA changes.

Many portfolio projects blur these two together and present rule outputs as if a model learned them.
This project keeps them explicitly separate:

- **Learned:** the intent/category label, trained on a real public dataset with real ground truth.
- **Not learned:** department and priority, produced by a small, readable, auditable rule layer.

The dataset used here does not contain department or priority ground truth, so no fake labels are
invented and no rule output is presented as a model prediction.

This is a real training and evaluation pipeline, not a wrapper around a hosted LLM API. No API keys
are required and no paid service is used.

---

## Architecture

```
                         +-----------------------------+
                         |  CLINC150 data_full.json    |
                         |  downloaded automatically   |
                         +--------------+--------------+
                                        |
                                        v
                  +---------------------------------------------+
                  | Data pipeline                               |
                  |  clean -> intent filter -> dedup + leakage  |
                  |  removal -> label encoding -> official      |
                  |  train / val / test splits (seeded)         |
                  +----------------+----------------------------+
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
 +-------------------------+                 +-----------------------------+
 | Baseline                |                 | Transformer                 |
 | TF-IDF (1-2 grams)      |                 | distilbert-base-uncased     |
 | + Logistic Regression   |                 | fine-tuned for sequence     |
 | C selected on val set   |                 | classification, early stop  |
 +-----------+-------------+                 +--------------+--------------+
             |                                              |
             +---------------------+------------------------+
                                   v
                    +--------------------------------+
                    | Evaluation on the SAME test set|
                    | accuracy / macro P,R,F1 /      |
                    | weighted F1 / per-class /      |
                    | confusion matrix / latency /   |
                    | error analysis                 |
                    +---------------+----------------+
                                    |
                                    v
                    +--------------------------------+
                    | Inference                      |
                    | load saved artifacts, predict  |
                    | intent + confidence            |
                    +---------------+----------------+
                                    |
                                    v
                    +--------------------------------+
                    | Deterministic routing layer    |
                    | intent -> department           |
                    | intent + keywords -> priority  |
                    | (RULES, NOT LEARNED)           |
                    +--------------------------------+
```

---

## Dataset

- **Source:** CLINC150, from the official research repository `clinc/oos-eval`
  (`data/data_full.json`), downloaded automatically at first run from
  `https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json`.
- **Content:** short real-world user requests labeled with 150 intents, with predefined
  `train` / `val` / `test` splits.
- **What this project uses:** by default a curated subset of support-relevant intents (billing,
  cards, account, orders, security, device/settings). The subset is defined by the
  `DEPARTMENT_BY_INTENT` map in the script. Use `--all-intents` to train on all 150 intents instead.
  If a curated intent name is not present in the downloaded file it is simply skipped, and if too few
  match the script falls back to the full label set.
- **Out-of-scope splits:** the `oos_*` splits are not used.
- **No fabricated data.** Department and priority labels do **not** exist in this dataset and are
  never treated as ground truth.

### Why the predefined splits

CLINC150 ships with official train/validation/test splits, so those are used directly rather than
re-splitting. This keeps results comparable to published work on the dataset and avoids accidental
leakage from re-shuffling. The script additionally:

- removes exact duplicate texts inside each split,
- removes any train/val example whose text already appears in a later-priority split
  (test is processed first, then val, then train), so no test text can leak into training,
- shuffles each split with a fixed seed.

---

## ML / NLP methodology

Core task: **multiclass single-label text classification** over intent labels.

Pipeline steps implemented in the script:

1. Dataset download with error handling and JSON validation.
2. Light text cleaning only (whitespace normalization, non-printable character removal). No
   stemming or stopword removal, because the transformer tokenizer and TF-IDF both benefit from the
   original wording.
3. Intent filtering to the curated support subset (or all intents).
4. Duplicate and leakage removal.
5. Label encoding into a stable, sorted `label_mapping.json`.
6. Tokenization with the DistilBERT tokenizer, dynamic padding per batch.
7. PyTorch `Dataset` / `DataLoader` with a custom padding collator.
8. Manual PyTorch training loop with linear warmup + decay, gradient clipping, per-epoch
   validation, best-checkpoint selection, and early stopping.
9. Final evaluation of both models on the identical test split.
10. Artifact saving, then re-loading from disk for evaluation and inference.

### Baseline model

- `TfidfVectorizer` with word unigrams and bigrams, sublinear term frequency, unicode accent
  stripping, lowercasing.
- `LogisticRegression` (lbfgs, `max_iter=2000`, fixed `random_state`).
- The regularization strength `C` is selected over a small grid `[0.5, 1.0, 4.0, 10.0]` using
  **validation macro F1 only**. The test set is never used for model selection.
- Saved as a single scikit-learn `Pipeline` so the vectorizer and classifier can never drift apart.

This is a genuinely strong baseline on short intent text, which is the point: the transformer has to
earn its place.

### Transformer model

- `distilbert-base-uncased` with `AutoModelForSequenceClassification`.
- Max sequence length 64 (support queries in this dataset are short; longer input is truncated).
- Batch size 16, AdamW, learning rate 5e-5, weight decay 0.01, 10% linear warmup, gradient clipping
  at 1.0.
- 3 epochs by default with early stopping (patience 2) on validation macro F1.
- The best epoch's weights are restored before saving, so the saved model is the best validation
  checkpoint, not simply the last epoch.
- CUDA is used automatically when available, otherwise CPU. Nothing in the defaults assumes a GPU.

---

## Training process

```
python support_ticket_classifier.py train
```

This will:

1. Print the environment report (Python, torch, transformers, scikit-learn, numpy, pandas versions,
   device, CUDA availability, seed).
2. Download the dataset if it is not already cached in `artifacts/data/`.
3. Build and save the splits and label mapping.
4. Train and tune the baseline, printing validation macro F1 per candidate `C`.
5. Fine-tune DistilBERT with per-epoch validation logging.
6. Save all artifacts.
7. Re-load the saved artifacts from disk and run the full evaluation, so the evaluation numbers come
   from exactly the files that inference will later use.

Useful flags for a slow laptop:

```
python support_ticket_classifier.py train --max-train-per-class 40 --epochs 2
python support_ticket_classifier.py train --skip-transformer
```

Rough expectation: with the default curated intent subset, CPU fine-tuning typically takes on the
order of tens of minutes; a CUDA GPU takes a few minutes. Exact time depends entirely on your
hardware.

---

## Evaluation methodology

Both models are evaluated on the **same saved test split**, loaded from
`artifacts/data/prepared_splits.json`. Reported metrics:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- full per-class precision / recall / F1 / support (the 15 weakest classes are printed when the
  label set is large; the complete per-class report is always written to `evaluation.json`)
- confusion matrix, saved as CSV with named rows and columns
- inference latency: mean and p95 single-example latency, and batch-of-16 latency, for each model

The script prints a direct comparison table:

```
Model                             Accuracy  Macro F1  Weighted F1
TF-IDF + Logistic Regression        ...       ...         ...
DistilBERT (fine-tuned)             ...       ...         ...
```

followed by a factual statement of the measured macro F1 difference in whichever direction it
actually goes.

**This README deliberately contains no performance numbers.**

```
Accuracy    : measured after training
Macro F1    : measured after training
Weighted F1 : measured after training
Latency     : measured after training, on your hardware
```

To generate the real numbers, run `train` (or `evaluate` on existing artifacts) and read the printed
comparison table or open `artifacts/results/evaluation.json`. No claim that the transformer beats the
baseline is made anywhere in this project unless your own measured results show it.

---

## Error analysis

After evaluation, the script collects every misclassified test example for each model, sorts them by
the model's confidence in the wrong answer (most confident mistakes first), and prints a limited
number of them (default 10, configurable with `--error-examples`). For each printed error:

- the original ticket text (truncated for terminal readability)
- the true label
- the predicted label
- the model's confidence in the prediction
- the probability the model assigned to the correct label

The full collected error list for both models is written to `artifacts/results/error_analysis.json`.
Confident mistakes are the ones worth looking at: they usually reveal genuinely overlapping intents
rather than random noise.

---

## Routing and priority logic

**This layer is not machine learning.** It is a small set of explicit rules applied on top of the
model's predicted intent. It is implemented in `DEPARTMENT_BY_INTENT`, `HIGH_PRIORITY_INTENTS`,
`MEDIUM_PRIORITY_INTENTS`, and `PRIORITY_RULES` in the script.

### Departments

Predicted intents map to one of:

| Department        | Example intents |
|-------------------|-----------------|
| Billing           | `bill_balance`, `bill_due`, `pay_bill`, `transfer`, `transactions`, `international_fees` |
| Account Support   | `account_blocked`, `pin_change`, `change_user_name`, `credit_limit`, `application_status` |
| Security          | `report_fraud`, `report_lost_card`, `card_declined`, `freeze_account` |
| Orders            | `new_card`, `order_checks`, `damaged_card`, `order_status`, `cancel` |
| Technical Support | `sync_device`, `reset_settings`, `change_language`, `change_volume` |
| General Support   | fallback for any intent not in the map (including all unmapped intents under `--all-intents`) |

These department names come from this project's routing policy, **not** from the dataset. The dataset
contains no department labels.

### Priority

Priority starts at `P3 - Low` and is escalated by rules:

| Condition | Resulting level |
|-----------|-----------------|
| Intent in the high-risk set (fraud, lost card, blocked account, declined card, frozen account) | P1 |
| Intent in the payment/logistics set (bills, payments, transfers, order status, cancellations) | P2 |
| Text matches a security-risk keyword (`fraud`, `stolen`, `hacked`, `unauthorized`, `scam`, ...) | P1 |
| Text matches a payment-risk keyword (`charged twice`, `double charge`, `overcharged`, ...) | P1 |
| Text matches an urgency keyword (`urgent`, `asap`, `immediately`, `emergency`, ...) | P2 |
| Text matches an escalation keyword (`lawyer`, `legal action`, `chargeback`, `complaint`, ...) | P2 |
| Text indicates an unanswered follow-up (`nobody has responded`, `still waiting`, ...) | P2 |

Escalations only move priority upward, never downward. Every rule that fired is returned in
`routing_reasons`, so any routing decision can be explained without inspecting model internals.

### Summary of what is learned and what is not

| Output | Source |
|--------|--------|
| `predicted_intent` | Learned classifier (DistilBERT or TF-IDF + LogReg) |
| `confidence`, `top_k` | Learned classifier softmax / predicted probabilities |
| `suggested_department` | Deterministic rules |
| `suggested_priority` | Deterministic rules |
| `routing_reasons` | Deterministic rules |

---

## Example prediction

```
python support_ticket_classifier.py predict --text "I was charged twice for my subscription" --json
```

Output shape (values depend on your trained model):

```json
{
  "text": "I was charged twice for my subscription",
  "model": "DistilBERT (fine-tuned)",
  "predicted_intent": "<model output>",
  "confidence": 0.0,
  "top_k": [
    {"intent": "<model output>", "probability": 0.0},
    {"intent": "<model output>", "probability": 0.0},
    {"intent": "<model output>", "probability": 0.0}
  ],
  "suggested_department": "Billing",
  "suggested_priority": "P1 - High",
  "priority_level": 1,
  "routing_reasons": ["payment risk keyword"],
  "routing_source": "deterministic rules (not learned)",
  "intent_source": "learned classifier"
}
```

---

## Installation

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`:

```
torch==2.3.1
transformers==4.41.2
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
```

Notes:

- For CPU-only machines, install the CPU build of PyTorch to keep the download small:
  `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu`
- For CUDA, install the PyTorch build matching your CUDA version from pytorch.org. Nothing else
  changes; the script detects CUDA automatically.
- `datasets`, `accelerate`, FastAPI, Streamlit, Docker, MLflow and W&B are intentionally **not**
  dependencies. The dataset is a single JSON file fetched over HTTPS, and the training loop is plain
  PyTorch, so those layers add nothing here.
- The only network access needed is the dataset download plus the Hugging Face Hub download of
  `distilbert-base-uncased` on first run. Both are cached locally afterwards.

---

## Usage commands

```bash
# train both models, evaluate, save all artifacts
python support_ticket_classifier.py train

# lighter training run for a slow CPU
python support_ticket_classifier.py train --max-train-per-class 40 --epochs 2

# train on all 150 CLINC intents instead of the curated support subset
python support_ticket_classifier.py train --all-intents

# baseline only (fast sanity check of the whole pipeline)
python support_ticket_classifier.py train --skip-transformer

# re-evaluate saved artifacts on the saved test split (no retraining)
python support_ticket_classifier.py evaluate
python support_ticket_classifier.py evaluate --error-examples 20

# single prediction
python support_ticket_classifier.py predict --text "I've been charged twice and nobody replied"
python support_ticket_classifier.py predict --text "my card was stolen" --json
python support_ticket_classifier.py predict --text "reset my pin" --model baseline

# interactive mode
python support_ticket_classifier.py interactive

# shorthand forms (no subcommand)
python support_ticket_classifier.py --text "I was charged twice for my subscription"
python support_ticket_classifier.py --interactive

# custom artifacts directory
python support_ticket_classifier.py --artifacts ./run1 train
```

The script is directly executable on Unix-like systems (`chmod +x support_ticket_classifier.py`,
then `./support_ticket_classifier.py train`).

Model selection at inference time: `--model auto` (default) prefers the fine-tuned transformer and
falls back to the baseline if transformer artifacts are missing; `--model transformer` and
`--model baseline` force a specific model, with a clear message if that model is unavailable.

---

## Project structure

```
.
├── support_ticket_classifier.py     # the entire project: data, training, evaluation, routing, CLI
├── requirements.txt                 # created by you from the block above
├── README.md
└── artifacts/                       # created automatically by the script
    ├── config.json                  # run config, environment, label set, baseline/transformer info
    ├── label_mapping.json           # label -> id, stable and sorted
    ├── data/
    │   ├── clinc150_data_full.json  # cached raw dataset download
    │   └── prepared_splits.json     # cleaned, deduplicated train/val/test splits
    ├── baseline/
    │   └── tfidf_logreg.joblib      # TF-IDF vectorizer + LogisticRegression in one pipeline
    ├── transformer/
    │   ├── config.json              # saved via save_pretrained
    │   ├── model.safetensors
    │   ├── tokenizer.json / vocab.txt / tokenizer_config.json / special_tokens_map.json
    │   └── ...
    └── results/
        ├── evaluation.json          # all metrics, per-class reports, latency
        ├── error_analysis.json      # collected misclassifications for both models
        ├── confusion_matrix_baseline.csv
        └── confusion_matrix_transformer.csv
```

All paths are relative and built with `pathlib`. No absolute paths are hard-coded. Directories are
created automatically.

---

## Reproducibility

- A single seed (default 42) is applied to `random`, `numpy`, `torch`, CUDA, and the DataLoader
  shuffling generator.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False`.
- Splits are the dataset's official ones; the deduplication and shuffling are deterministic given the
  seed.
- The test split is written to disk and re-loaded for evaluation, so the baseline and transformer are
  always scored on byte-identical data.
- Every run prints Python version, platform, torch / transformers / scikit-learn / numpy / pandas
  versions, the selected device, and whether CUDA is available. The same report is stored in
  `artifacts/config.json`.
- Model selection uses the validation split only; the test split is touched once, at final
  evaluation.

Exact bitwise reproducibility across different hardware, CUDA versions, or thread counts is not
guaranteed — that is a property of floating-point kernels, not of this script. Results on identical
environments are reproducible.

---

## Robustness

Handled explicitly:

- Empty or whitespace-only input text — rejected with a clear message and a non-zero exit code.
- Very long input — capped at 2000 characters before tokenization, then truncated to the model's max
  sequence length.
- Missing model artifacts — clear instruction to run `train`, with automatic fallback from
  transformer to baseline where sensible.
- Corrupted or unreadable artifacts — load failures are caught and reported instead of crashing with
  a stack trace.
- Label mapping / saved model mismatch — detected and reported rather than silently producing
  nonsense labels.
- Dataset download failure — explains the failure and points to the exact local path where the file
  can be placed manually.
- Missing or malformed saved config — falls back to defaults with a warning.
- Invalid CLI arguments — handled by `argparse`; running with no arguments prints help and exits
  with code 2.
- No CUDA — CPU is selected automatically; nothing in the code path requires a GPU.
- `Ctrl+C` — exits cleanly, including inside interactive mode.

---

## Limitations

- **Department and priority are not validated against real ground truth.** They are policy rules. No
  accuracy claim can be or is made about them.
- CLINC150 queries are short, clean, single-intent utterances. Real support tickets are longer,
  messier, often multi-intent, and frequently contain log dumps or screenshots. Performance on real
  production tickets will be lower.
- The curated intent subset is skewed toward banking and card-related intents, because that is what
  CLINC150 actually covers. It is not a full representation of a generic SaaS support desk.
- Single-label classification only. A ticket that is both a billing dispute and a security concern
  gets one intent.
- No out-of-scope / unknown-intent detection is enabled, even though CLINC150 contains `oos` data.
  The model will always return its best in-vocabulary guess.
- Priority keyword rules are English-only and regex-based. They will miss paraphrases.
- Latency is measured on whatever hardware you run it on and is not a benchmark of anything else.
- No calibration is applied, so confidence values are raw softmax outputs and should not be read as
  true probabilities.

---

## Possible future improvements

- Out-of-scope detection using the `oos_*` splits, or a confidence threshold that routes low-confidence
  tickets to `General Support` for human triage.
- Probability calibration (temperature scaling on the validation set) so confidence thresholds mean
  something.
- Multi-label classification for tickets that genuinely carry more than one intent.
- Training on a longer-form support dataset, or on real anonymized tickets, with `max_length`
  increased accordingly.
- Replacing the priority keyword rules with a small learned urgency model, once labeled urgency data
  actually exists.
- Cross-validation over the training set for the baseline instead of a single validation split.
- Knowledge distillation or ONNX export if CPU inference latency becomes a constraint.
- A thin serving layer (FastAPI) — deliberately excluded here, since this project is scoped to the
  NLP problem rather than deployment infrastructure.
