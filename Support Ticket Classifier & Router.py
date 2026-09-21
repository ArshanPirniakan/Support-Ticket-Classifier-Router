#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

DATASET_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
DATASET_NAME = "CLINC150 (clinc/oos-eval, data_full.json)"

DEPARTMENT_BY_INTENT: Dict[str, str] = {
    "bill_balance": "Billing",
    "bill_due": "Billing",
    "pay_bill": "Billing",
    "min_payment": "Billing",
    "interest_rate": "Billing",
    "apr": "Billing",
    "international_fees": "Billing",
    "transfer": "Billing",
    "transactions": "Billing",
    "balance": "Billing",
    "spending_history": "Billing",
    "redeem_rewards": "Billing",
    "rewards_balance": "Billing",
    "account_blocked": "Account Support",
    "pin_change": "Account Support",
    "user_name": "Account Support",
    "change_user_name": "Account Support",
    "credit_score": "Account Support",
    "credit_limit": "Account Support",
    "credit_limit_change": "Account Support",
    "improve_credit_score": "Account Support",
    "routing": "Account Support",
    "application_status": "Account Support",
    "expiration_date": "Account Support",
    "report_fraud": "Security",
    "report_lost_card": "Security",
    "card_declined": "Security",
    "freeze_account": "Security",
    "new_card": "Orders",
    "order_checks": "Orders",
    "replacement_card_duration": "Orders",
    "damaged_card": "Orders",
    "order_status": "Orders",
    "order": "Orders",
    "cancel": "Orders",
    "sync_device": "Technical Support",
    "change_language": "Technical Support",
    "change_speed": "Technical Support",
    "change_volume": "Technical Support",
    "change_accent": "Technical Support",
    "reset_settings": "Technical Support",
}

DEFAULT_DEPARTMENT = "General Support"

HIGH_PRIORITY_INTENTS = {
    "report_fraud",
    "report_lost_card",
    "account_blocked",
    "card_declined",
    "freeze_account",
}

MEDIUM_PRIORITY_INTENTS = {
    "bill_balance",
    "bill_due",
    "pay_bill",
    "min_payment",
    "transfer",
    "transactions",
    "international_fees",
    "damaged_card",
    "order_status",
    "application_status",
    "cancel",
}

PRIORITY_RULES: Tuple[Tuple[str, str, int], ...] = (
    (r"\b(fraud|fraudulent|stolen|hacked|scam|unauthorized|unauthorised|phishing)\b", "security risk keyword", 1),
    (r"\b(charged twice|double charged|double charge|duplicate charge|overcharged|wrong amount)\b", "payment risk keyword", 1),
    (r"\b(urgent|urgently|asap|immediately|emergency|right now)\b", "urgency keyword", 2),
    (r"\b(lawyer|legal action|chargeback|dispute|complaint|cancel my account)\b", "escalation keyword", 2),
    (r"\b(no ?one (has )?(answered|responded|replied)|nobody (has )?(answered|responded|replied)|still waiting|third time)\b", "unanswered follow-up", 2),
)

PRIORITY_LABELS = {1: "P1 - High", 2: "P2 - Medium", 3: "P3 - Low"}


@dataclass
class Config:
    seed: int = 42
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    max_grad_norm: float = 1.0
    max_train_per_class: Optional[int] = None
    use_all_intents: bool = False
    max_input_chars: int = 2000
    baseline_c_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 4.0, 10.0])


class ArtifactPaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.raw_dataset = self.data_dir / "clinc150_data_full.json"
        self.prepared = self.data_dir / "prepared_splits.json"
        self.baseline_dir = root / "baseline"
        self.baseline_model = self.baseline_dir / "tfidf_logreg.joblib"
        self.transformer_dir = root / "transformer"
        self.label_mapping = root / "label_mapping.json"
        self.config = root / "config.json"
        self.results_dir = root / "results"
        self.evaluation = self.results_dir / "evaluation.json"
        self.error_analysis = self.results_dir / "error_analysis.json"
        self.confusion_baseline = self.results_dir / "confusion_matrix_baseline.csv"
        self.confusion_transformer = self.results_dir / "confusion_matrix_transformer.csv"

    def ensure(self) -> None:
        for directory in (self.root, self.data_dir, self.baseline_dir, self.transformer_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class RoutingDecision:
    department: str
    priority: str
    priority_level: int
    reasons: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_report(device: torch.device, seed: int) -> Dict[str, Any]:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "seed": seed,
    }
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
    return report


def print_environment(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    for key, value in report.items():
        print(f"{key:>18}: {value}")
    print()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    stripped = "".join(char if char.isprintable() or char in "\n\t" else " " for char in text)
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_input_text(text: Optional[str], max_chars: int) -> str:
    if text is None:
        raise ValueError("No input text was provided.")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")
    return cleaned[:max_chars]


def download_dataset(paths: ArtifactPaths, force: bool = False) -> Path:
    paths.ensure()
    if paths.raw_dataset.exists() and paths.raw_dataset.stat().st_size > 0 and not force:
        return paths.raw_dataset
    print(f"Downloading dataset from {DATASET_URL}")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not download the dataset ({error}). Check the network connection or place "
            f"data_full.json manually at {paths.raw_dataset}"
        ) from error
    try:
        json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Downloaded dataset is not valid JSON.") from error
    paths.raw_dataset.write_bytes(payload)
    print(f"Saved dataset to {paths.raw_dataset} ({len(payload)} bytes)")
    return paths.raw_dataset


def load_raw_dataset(path: Path) -> Dict[str, List[List[str]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read dataset file {path}: {error}") from error
    for split in ("train", "val", "test"):
        if split not in raw:
            raise RuntimeError(f"Dataset file is missing the '{split}' split.")
    return raw


def available_intents(raw: Dict[str, List[List[str]]]) -> List[str]:
    intents = set()
    for split in ("train", "val", "test"):
        for _, intent in raw[split]:
            intents.add(intent)
    return sorted(intents)


def resolve_selected_intents(raw: Dict[str, List[List[str]]], use_all: bool) -> List[str]:
    present = available_intents(raw)
    if use_all:
        return present
    present_set = set(present)
    selected = [intent for intent in DEPARTMENT_BY_INTENT if intent in present_set]
    if len(selected) < 5:
        print("Warning: curated intent subset did not match the dataset, falling back to all intents.")
        return present
    return sorted(selected)


def build_splits(
    raw: Dict[str, List[List[str]]],
    selected_intents: Sequence[str],
    config: Config,
) -> Dict[str, List[Dict[str, str]]]:
    selected = set(selected_intents)
    rng = random.Random(config.seed)
    splits: Dict[str, List[Dict[str, str]]] = {}
    seen_texts: Dict[str, str] = {}

    for split in ("test", "val", "train"):
        rows: List[Dict[str, str]] = []
        local_seen = set()
        for text, intent in raw[split]:
            if intent not in selected:
                continue
            cleaned = clean_text(text)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in local_seen or key in seen_texts:
                continue
            local_seen.add(key)
            rows.append({"text": cleaned, "label": intent})
        for row in rows:
            seen_texts[row["text"].lower()] = split
        rng.shuffle(rows)
        splits[split] = rows

    if config.max_train_per_class is not None:
        counts: Dict[str, int] = {}
        limited: List[Dict[str, str]] = []
        for row in splits["train"]:
            count = counts.get(row["label"], 0)
            if count >= config.max_train_per_class:
                continue
            counts[row["label"]] = count + 1
            limited.append(row)
        splits["train"] = limited

    for split in ("train", "val", "test"):
        if not splits[split]:
            raise RuntimeError(f"Split '{split}' is empty after filtering and deduplication.")
    return {"train": splits["train"], "val": splits["val"], "test": splits["test"]}


def build_label_mapping(splits: Dict[str, List[Dict[str, str]]]) -> Dict[str, int]:
    labels = sorted({row["label"] for rows in splits.values() for row in rows})
    return {label: index for index, label in enumerate(labels)}


def split_arrays(rows: List[Dict[str, str]], label_to_id: Dict[str, int]) -> Tuple[List[str], np.ndarray]:
    texts = [row["text"] for row in rows]
    labels = np.array([label_to_id[row["label"]] for row in rows], dtype=np.int64)
    return texts, labels


def save_prepared(paths: ArtifactPaths, splits: Dict[str, List[Dict[str, str]]], label_to_id: Dict[str, int]) -> None:
    paths.ensure()
    payload = {"dataset": DATASET_NAME, "splits": splits}
    paths.prepared.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    paths.label_mapping.write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prepared(paths: ArtifactPaths) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, int]]:
    if not paths.prepared.exists() or not paths.label_mapping.exists():
        raise RuntimeError(
            f"Prepared data or label mapping not found in {paths.root}. Run the 'train' command first."
        )
    try:
        payload = json.loads(paths.prepared.read_text(encoding="utf-8"))
        label_to_id = json.loads(paths.label_mapping.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read prepared artifacts: {error}") from error
    return payload["splits"], {key: int(value) for key, value in label_to_id.items()}


def id_to_label_list(label_to_id: Dict[str, int]) -> List[str]:
    ordered = sorted(label_to_id.items(), key=lambda item: item[1])
    return [label for label, _ in ordered]


def describe_splits(splits: Dict[str, List[Dict[str, str]]], label_to_id: Dict[str, int]) -> None:
    print("=" * 72)
    print("DATASET")
    print("=" * 72)
    print(f"{'source':>18}: {DATASET_NAME}")
    print(f"{'classes':>18}: {len(label_to_id)}")
    for split in ("train", "val", "test"):
        print(f"{split:>18}: {len(splits[split])} examples")
    print()


class TicketDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


class PadCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch
        ]
        padded = self.tokenizer.pad(features, return_tensors="pt")
        padded["labels"] = labels
        return padded


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> Dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=list(range(num_labels))
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]) -> Dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    frame = pd.DataFrame(matrix, index=list(label_names), columns=list(label_names))
    frame.to_csv(path)


def print_per_class_summary(report: Dict[str, Any], label_names: Sequence[str], max_rows: int = 40) -> None:
    rows = [(name, report[name]) for name in label_names if name in report]
    rows.sort(key=lambda item: item[1]["f1-score"])
    shown = rows if len(rows) <= max_rows else rows[:15]
    heading = "PER-CLASS METRICS" if len(rows) <= max_rows else "PER-CLASS METRICS (15 WEAKEST CLASSES)"
    print(heading)
    print(f"{'class':<32}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    for name, values in shown:
        print(
            f"{name:<32}{values['precision']:>11.3f}{values['recall']:>9.3f}"
            f"{values['f1-score']:>9.3f}{int(values['support']):>9}"
        )
    print()


def train_baseline(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    val_texts: Sequence[str],
    val_labels: np.ndarray,
    config: Config,
    num_labels: int,
) -> Tuple[Pipeline, Dict[str, Any]]:
    best_pipeline: Optional[Pipeline] = None
    best_score = -1.0
    best_c = config.baseline_c_grid[0]
    search_log: List[Dict[str, float]] = []
    for c_value in config.baseline_c_grid:
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2000,
                        random_state=config.seed,
                    ),
                ),
            ]
        )
        pipeline.fit(list(train_texts), train_labels)
        predictions = pipeline.predict(list(val_texts))
        score = f1_score(val_labels, predictions, average="macro", zero_division=0)
        search_log.append({"C": c_value, "val_macro_f1": float(score)})
        print(f"baseline C={c_value:<6} val macro F1={score:.4f}")
        if score > best_score:
            best_score = float(score)
            best_pipeline = pipeline
            best_c = c_value
    if best_pipeline is None:
        raise RuntimeError("Baseline training failed to produce a model.")
    print(f"Selected baseline C={best_c} (val macro F1={best_score:.4f})\n")
    info = {"selected_C": best_c, "val_macro_f1": best_score, "search": search_log, "num_labels": num_labels}
    return best_pipeline, info


@torch.no_grad()
def transformer_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for batch in loader:
        labels = batch.pop("labels", None)
        del labels
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def train_transformer(
    config: Config,
    paths: ArtifactPaths,
    splits: Dict[str, List[Dict[str, str]]],
    label_to_id: Dict[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    label_names = id_to_label_list(label_to_id)
    train_texts, train_labels = split_arrays(splits["train"], label_to_id)
    val_texts, val_labels = split_arrays(splits["val"], label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_names),
        id2label={index: name for index, name in enumerate(label_names)},
        label2id={name: index for index, name in enumerate(label_names)},
    )
    model.to(device)

    collator = PadCollator(tokenizer)
    train_loader = DataLoader(
        TicketDataset(train_texts, train_labels, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        TicketDataset(val_texts, val_labels, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    log_every = max(1, len(train_loader) // 5)
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += float(loss.detach().cpu())
            if step % log_every == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{config.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}"
                )
        probabilities = transformer_probabilities(model, val_loader, device)
        predictions = probabilities.argmax(axis=1)
        metrics = compute_metrics(val_labels, predictions, len(label_names))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch} validation accuracy={metrics['accuracy']:.4f} "
            f"macro F1={metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after epoch {epoch} (no improvement for {config.patience} epochs).")
                break

    training_seconds = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    paths.ensure()
    model.save_pretrained(paths.transformer_dir)
    tokenizer.save_pretrained(paths.transformer_dir)
    print(f"\nSaved transformer artifacts to {paths.transformer_dir}")
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "training_seconds": training_seconds,
    }


class BaselinePredictor:
    name = "TF-IDF + Logistic Regression"

    def __init__(self, pipeline: Pipeline, label_names: Sequence[str]) -> None:
        self.pipeline = pipeline
        self.label_names = list(label_names)

    @classmethod
    def load(cls, paths: ArtifactPaths, label_names: Sequence[str]) -> Optional["BaselinePredictor"]:
        if not paths.baseline_model.exists():
            return None
        try:
            pipeline = joblib.load(paths.baseline_model)
        except Exception as error:
            print(f"Warning: could not load baseline model ({error}).")
            return None
        return cls(pipeline, label_names)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_names)), dtype=np.float32)
        return np.asarray(self.pipeline.predict_proba(list(texts)), dtype=np.float32)


class TransformerPredictor:
    name = "DistilBERT (fine-tuned)"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        label_names: Sequence[str],
        device: torch.device,
        max_length: int,
        batch_size: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_names = list(label_names)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def load(
        cls,
        paths: ArtifactPaths,
        label_names: Sequence[str],
        device: torch.device,
        max_length: int,
        batch_size: int,
    ) -> Optional["TransformerPredictor"]:
        if not (paths.transformer_dir / "config.json").exists():
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(paths.transformer_dir)
            model = AutoModelForSequenceClassification.from_pretrained(paths.transformer_dir)
        except Exception as error:
            print(f"Warning: could not load transformer artifacts ({error}).")
            return None
        if model.config.num_labels != len(label_names):
            print("Warning: saved transformer label count does not match label_mapping.json.")
            return None
        model.to(device)
        model.eval()
        return cls(model, tokenizer, label_names, device, max_length, batch_size)

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_names)), dtype=np.float32)
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)


def measure_latency(predictor: Any, texts: Sequence[str], repeats: int = 20) -> Dict[str, float]:
    if not texts:
        return {"single_example_ms": 0.0, "batch_16_ms": 0.0, "batch_16_ms_per_example": 0.0}
    sample = list(texts[: min(len(texts), 16)])
    predictor.predict_proba(sample[:1])
    single_times: List[float] = []
    for index in range(repeats):
        text = sample[index % len(sample)]
        start = time.perf_counter()
        predictor.predict_proba([text])
        single_times.append((time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    predictor.predict_proba(sample)
    batch_ms = (time.perf_counter() - start) * 1000.0
    return {
        "single_example_ms": float(np.mean(single_times)),
        "single_example_ms_p95": float(np.percentile(single_times, 95)),
        "batch_16_ms": float(batch_ms),
        "batch_16_ms_per_example": float(batch_ms / len(sample)),
    }


def evaluate_predictor(
    predictor: Any,
    texts: Sequence[str],
    labels: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, Any]:
    probabilities = predictor.predict_proba(texts)
    predictions = probabilities.argmax(axis=1)
    metrics = compute_metrics(labels, predictions, len(label_names))
    return {
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "per_class": per_class_report(labels, predictions, label_names),
    }


def collect_errors(
    texts: Sequence[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for index, (true_id, predicted_id) in enumerate(zip(labels, predictions)):
        if true_id == predicted_id:
            continue
        confidence = float(probabilities[index][predicted_id])
        errors.append(
            {
                "text": texts[index],
                "true_label": label_names[int(true_id)],
                "predicted_label": label_names[int(predicted_id)],
                "confidence": confidence,
                "true_label_probability": float(probabilities[index][int(true_id)]),
            }
        )
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    return errors[:limit]


def print_errors(title: str, errors: List[Dict[str, Any]], total_errors: int) -> None:
    print("=" * 72)
    print(f"ERROR ANALYSIS - {title}")
    print("=" * 72)
    print(f"total misclassified test examples: {total_errors}")
    if not errors:
        print("no misclassified examples to show\n")
        return
    print("showing the most confident mistakes\n")
    for position, error in enumerate(errors, start=1):
        text = error["text"]
        if len(text) > 160:
            text = text[:157] + "..."
        print(f"[{position}] {text}")
        print(
            f"    true={error['true_label']}  predicted={error['predicted_label']}  "
            f"confidence={error['confidence']:.3f}  p(true)={error['true_label_probability']:.3f}"
        )
    print()


def route_ticket(intent: str, text: str) -> RoutingDecision:
    department = DEPARTMENT_BY_INTENT.get(intent, DEFAULT_DEPARTMENT)
    level = 3
    reasons: List[str] = []
    if intent in HIGH_PRIORITY_INTENTS:
        level = 1
        reasons.append(f"intent '{intent}' is in the high-risk intent list")
    elif intent in MEDIUM_PRIORITY_INTENTS:
        level = 2
        reasons.append(f"intent '{intent}' is in the payment/logistics intent list")
    lowered = text.lower()
    for pattern, reason, target_level in PRIORITY_RULES:
        if re.search(pattern, lowered):
            if target_level < level:
                level = target_level
            reasons.append(reason)
    if not reasons:
        reasons.append("no escalation rule matched, default priority applied")
    return RoutingDecision(
        department=department,
        priority=PRIORITY_LABELS[level],
        priority_level=level,
        reasons=reasons,
    )


def build_prediction(
    text: str,
    predictor: Any,
    label_names: Sequence[str],
    top_k: int = 3,
) -> Dict[str, Any]:
    probabilities = predictor.predict_proba([text])[0]
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    intent = label_names[best_index]
    decision = route_ticket(intent, text)
    return {
        "text": text,
        "model": predictor.name,
        "predicted_intent": intent,
        "confidence": float(probabilities[best_index]),
        "top_k": [
            {"intent": label_names[int(index)], "probability": float(probabilities[int(index)])}
            for index in order[: min(top_k, len(label_names))]
        ],
        "suggested_department": decision.department,
        "suggested_priority": decision.priority,
        "priority_level": decision.priority_level,
        "routing_reasons": decision.reasons,
        "routing_source": "deterministic rules (not learned)",
        "intent_source": "learned classifier",
    }


def print_prediction(result: Dict[str, Any]) -> None:
    print("-" * 72)
    print(f"text                : {result['text']}")
    print(f"model               : {result['model']}")
    print(f"predicted intent    : {result['predicted_intent']} (learned)")
    print(f"confidence          : {result['confidence']:.4f}")
    print(f"suggested department: {result['suggested_department']} (rule-based)")
    print(f"suggested priority  : {result['suggested_priority']} (rule-based)")
    print(f"routing reasons     : {'; '.join(result['routing_reasons'])}")
    alternatives = ", ".join(
        f"{item['intent']}={item['probability']:.3f}" for item in result["top_k"][1:]
    )
    if alternatives:
        print(f"alternatives        : {alternatives}")
    print("-" * 72)


def print_comparison(rows: List[Dict[str, Any]]) -> None:
    print("=" * 72)
    print("MODEL COMPARISON (identical test set)")
    print("=" * 72)
    print(f"{'Model':<32}{'Accuracy':>10}{'Macro F1':>10}{'Weighted F1':>13}")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['name']:<32}{metrics['accuracy']:>10.4f}"
            f"{metrics['macro_f1']:>10.4f}{metrics['weighted_f1']:>13.4f}"
        )
    print()
    if len(rows) == 2:
        delta = rows[1]["metrics"]["macro_f1"] - rows[0]["metrics"]["macro_f1"]
        direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal to"
        print(
            f"Measured result: {rows[1]['name']} macro F1 is {abs(delta):.4f} {direction} "
            f"than {rows[0]['name']}."
        )
        print()


def run_evaluation(paths: ArtifactPaths, config: Config, device: torch.device, error_limit: int) -> int:
    splits, label_to_id = load_prepared(paths)
    label_names = id_to_label_list(label_to_id)
    test_texts, test_labels = split_arrays(splits["test"], label_to_id)

    baseline = BaselinePredictor.load(paths, label_names)
    transformer = TransformerPredictor.load(
        paths, label_names, device, config.max_length, config.eval_batch_size
    )
    if baseline is None and transformer is None:
        print("No usable model artifacts were found. Run 'train' first.")
        return 1

    describe_splits(splits, label_to_id)
    comparison_rows: List[Dict[str, Any]] = []
    evaluation_payload: Dict[str, Any] = {
        "dataset": DATASET_NAME,
        "num_labels": len(label_names),
        "test_size": len(test_texts),
        "device": str(device),
        "models": {},
    }
    error_payload: Dict[str, Any] = {}

    if baseline is not None:
        result = evaluate_predictor(baseline, test_texts, test_labels, label_names)
        latency = measure_latency(baseline, test_texts)
        comparison_rows.append({"name": baseline.name, "metrics": result["metrics"]})
        save_confusion_matrix(paths.confusion_baseline, test_labels, result["predictions"], label_names)
        evaluation_payload["models"]["baseline"] = {
            "name": baseline.name,
            "metrics": result["metrics"],
            "latency_ms": latency,
            "per_class": result["per_class"],
            "confusion_matrix_file": paths.confusion_baseline.name,
        }
        errors = collect_errors(
            test_texts, test_labels, result["predictions"], result["probabilities"], label_names, error_limit
        )
        error_payload["baseline"] = errors
        print("=" * 72)
        print(f"TEST RESULTS - {baseline.name}")
        print("=" * 72)
        for key, value in result["metrics"].items():
            print(f"{key:>18}: {value:.4f}")
        print(f"{'single ex. latency':>18}: {latency['single_example_ms']:.2f} ms")
        print(f"{'batch(16) latency':>18}: {latency['batch_16_ms']:.2f} ms")
        print()
        print_per_class_summary(result["per_class"], label_names)
        print_errors(baseline.name, errors, int((result["predictions"] != test_labels).sum()))

    if transformer is not None:
        result = evaluate_predictor(transformer, test_texts, test_labels, label_names)
        latency = measure_latency(transformer, test_texts)
        comparison_rows.append({"name": transformer.name, "metrics": result["metrics"]})
        save_confusion_matrix(paths.confusion_transformer, test_labels, result["predictions"], label_names)
        evaluation_payload["models"]["transformer"] = {
            "name": transformer.name,
            "metrics": result["metrics"],
            "latency_ms": latency,
            "per_class": result["per_class"],
            "confusion_matrix_file": paths.confusion_transformer.name,
        }
        errors = collect_errors(
            test_texts, test_labels, result["predictions"], result["probabilities"], label_names, error_limit
        )
        error_payload["transformer"] = errors
        print("=" * 72)
        print(f"TEST RESULTS - {transformer.name}")
        print("=" * 72)
        for key, value in result["metrics"].items():
            print(f"{key:>18}: {value:.4f}")
        print(f"{'single ex. latency':>18}: {latency['single_example_ms']:.2f} ms")
        print(f"{'p95 single latency':>18}: {latency['single_example_ms_p95']:.2f} ms")
        print(f"{'batch(16) latency':>18}: {latency['batch_16_ms']:.2f} ms")
        print(f"{'per example (batch)':>18}: {latency['batch_16_ms_per_example']:.2f} ms")
        print()
        print_per_class_summary(result["per_class"], label_names)
        print_errors(transformer.name, errors, int((result["predictions"] != test_labels).sum()))

    print_comparison(comparison_rows)
    paths.ensure()
    paths.evaluation.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")
    paths.error_analysis.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics to {paths.evaluation}")
    print(f"Saved error analysis to {paths.error_analysis}")
    print(f"Saved confusion matrices to {paths.results_dir}")
    return 0


def run_training(args: argparse.Namespace, paths: ArtifactPaths, config: Config, device: torch.device) -> int:
    set_seed(config.seed)
    environment = environment_report(device, config.seed)
    print_environment(environment)

    dataset_path = download_dataset(paths, force=args.force_download)
    raw = load_raw_dataset(dataset_path)
    selected_intents = resolve_selected_intents(raw, config.use_all_intents)
    splits = build_splits(raw, selected_intents, config)
    label_to_id = build_label_mapping(splits)
    save_prepared(paths, splits, label_to_id)
    describe_splits(splits, label_to_id)

    label_names = id_to_label_list(label_to_id)
    train_texts, train_labels = split_arrays(splits["train"], label_to_id)
    val_texts, val_labels = split_arrays(splits["val"], label_to_id)

    print("=" * 72)
    print("BASELINE TRAINING (TF-IDF + Logistic Regression)")
    print("=" * 72)
    baseline_pipeline, baseline_info = train_baseline(
        train_texts, train_labels, val_texts, val_labels, config, len(label_names)
    )
    joblib.dump(baseline_pipeline, paths.baseline_model)
    print(f"Saved baseline model to {paths.baseline_model}\n")

    transformer_info: Dict[str, Any] = {"skipped": True}
    if not args.skip_transformer:
        print("=" * 72)
        print("TRANSFORMER FINE-TUNING (DistilBERT)")
        print("=" * 72)
        set_seed(config.seed)
        transformer_info = train_transformer(config, paths, splits, label_to_id, device)
        print()

    config_payload = {
        "config": asdict(config),
        "environment": environment,
        "dataset": DATASET_NAME,
        "dataset_url": DATASET_URL,
        "num_labels": len(label_names),
        "selected_intents": label_names,
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "baseline": baseline_info,
        "transformer": transformer_info,
    }
    paths.config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Saved run configuration to {paths.config}\n")

    return run_evaluation(paths, config, device, args.error_examples)


def resolve_predictor(
    paths: ArtifactPaths,
    config: Config,
    device: torch.device,
    preference: str,
) -> Tuple[Optional[Any], List[str]]:
    try:
        _, label_to_id = load_prepared(paths)
    except RuntimeError as error:
        print(str(error))
        return None, []
    label_names = id_to_label_list(label_to_id)

    transformer: Optional[TransformerPredictor] = None
    baseline: Optional[BaselinePredictor] = None
    if preference in ("auto", "transformer"):
        transformer = TransformerPredictor.load(
            paths, label_names, device, config.max_length, config.eval_batch_size
        )
    if preference in ("auto", "baseline") or (preference == "transformer" and transformer is None):
        baseline = BaselinePredictor.load(paths, label_names)

    if preference == "transformer":
        if transformer is None:
            print("Transformer artifacts are unavailable.")
            if baseline is not None:
                print("Falling back to the baseline model.")
                return baseline, label_names
            return None, label_names
        return transformer, label_names
    if preference == "baseline":
        if baseline is None:
            print("Baseline artifacts are unavailable. Run 'train' first.")
            return None, label_names
        return baseline, label_names
    if transformer is not None:
        return transformer, label_names
    if baseline is not None:
        print("Transformer artifacts not found, using the baseline model.")
        return baseline, label_names
    print("No usable model artifacts were found. Run 'train' first.")
    return None, label_names


def load_saved_max_length(paths: ArtifactPaths, config: Config) -> Config:
    if not paths.config.exists():
        return config
    try:
        payload = json.loads(paths.config.read_text(encoding="utf-8"))
        saved = payload.get("config", {})
        config.max_length = int(saved.get("max_length", config.max_length))
        config.max_input_chars = int(saved.get("max_input_chars", config.max_input_chars))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        print("Warning: could not read saved configuration, using defaults.")
    return config


def run_prediction(args: argparse.Namespace, paths: ArtifactPaths, config: Config, device: torch.device) -> int:
    config = load_saved_max_length(paths, config)
    try:
        text = normalize_input_text(args.text, config.max_input_chars)
    except ValueError as error:
        print(f"Input error: {error}")
        return 2
    predictor, label_names = resolve_predictor(paths, config, device, args.model)
    if predictor is None:
        return 1
    result = build_prediction(text, predictor, label_names)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_prediction(result)
    return 0


def run_interactive(args: argparse.Namespace, paths: ArtifactPaths, config: Config, device: torch.device) -> int:
    config = load_saved_max_length(paths, config)
    predictor, label_names = resolve_predictor(paths, config, device, args.model)
    if predictor is None:
        return 1
    print(f"Interactive mode using {predictor.name} on {device}.")
    print("Type a support ticket and press Enter. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            raw_text = input("ticket> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        if raw_text.strip().lower() in {"exit", "quit"}:
            print("Exiting.")
            return 0
        try:
            text = normalize_input_text(raw_text, config.max_input_chars)
        except ValueError as error:
            print(f"Input error: {error}")
            continue
        result = build_prediction(text, predictor, label_names)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_prediction(result)


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    config.seed = getattr(args, "seed", config.seed)
    config.max_length = getattr(args, "max_length", config.max_length)
    config.batch_size = getattr(args, "batch_size", config.batch_size)
    config.epochs = getattr(args, "epochs", config.epochs)
    config.learning_rate = getattr(args, "learning_rate", config.learning_rate)
    config.max_train_per_class = getattr(args, "max_train_per_class", config.max_train_per_class)
    config.use_all_intents = getattr(args, "all_intents", config.use_all_intents)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support_ticket_classifier.py",
        description="Support Ticket Classifier & Router: intent classification with deterministic routing.",
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", choices=["auto", "transformer", "baseline"], default="auto")
    parser.add_argument("--json", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="download data, train both models, evaluate, save artifacts")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    train_parser.add_argument("--max-length", type=int, default=64, dest="max_length")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, dest="learning_rate")
    train_parser.add_argument("--max-train-per-class", type=int, default=None, dest="max_train_per_class")
    train_parser.add_argument("--all-intents", action="store_true", dest="all_intents")
    train_parser.add_argument("--skip-transformer", action="store_true", dest="skip_transformer")
    train_parser.add_argument("--force-download", action="store_true", dest="force_download")
    train_parser.add_argument("--error-examples", type=int, default=10, dest="error_examples")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate saved models on the saved test split")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--error-examples", type=int, default=10, dest="error_examples")

    predict_parser = subparsers.add_parser("predict", help="classify and route a single ticket")
    predict_parser.add_argument("--text", type=str, required=True)
    predict_parser.add_argument("--model", choices=["auto", "transformer", "baseline"], default="auto")
    predict_parser.add_argument("--json", action="store_true")
    predict_parser.add_argument("--seed", type=int, default=42)

    interactive_parser = subparsers.add_parser("interactive", help="classify tickets in a loop")
    interactive_parser.add_argument("--model", choices=["auto", "transformer", "baseline"], default="auto")
    interactive_parser.add_argument("--json", action="store_true")
    interactive_parser.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command
    if command is None:
        if args.interactive:
            command = "interactive"
        elif args.text is not None:
            command = "predict"
        else:
            parser.print_help()
            return 2

    paths = ArtifactPaths(Path(args.artifacts).expanduser().resolve())
    config = build_config(args)
    device = select_device()
    set_seed(config.seed)

    try:
        if command == "train":
            paths.ensure()
            return run_training(args, paths, config, device)
        if command == "evaluate":
            print_environment(environment_report(device, config.seed))
            return run_evaluation(paths, config, device, args.error_examples)
        if command == "predict":
            return run_prediction(args, paths, config, device)
        if command == "interactive":
            return run_interactive(args, paths, config, device)
    except RuntimeError as error:
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())