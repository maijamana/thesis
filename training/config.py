"""
Training run presets aligned with thesis Table (Run 1–3).

Override paths and counts when your prepared CSV differs from the paper.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Hyperparameters for one T5 fine-tuning run."""

    name: str
    description: str
    learning_rate: float
    warmup_steps: int
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.01
    early_stopping_patience: int = 4
    max_length: int = 128
    train_pairs_note: str = ""


RUNS: dict[str, RunConfig] = {
    "run1": RunConfig(
        name="run1",
        description="WikiLarge (CEFR Δ > 0.8) + ASSET + GPT-augmented pairs",
        learning_rate=5e-5,
        warmup_steps=500,
        train_pairs_note="~17,430 pairs (thesis)",
    ),
    "run2": RunConfig(
        name="run2",
        description="ASSET + GPT only (no WikiLarge)",
        learning_rate=3e-5,
        warmup_steps=50,
        train_pairs_note="~3,414 pairs (thesis)",
    ),
    "run3": RunConfig(
        name="run3",
        description="WikiLarge (CEFR Δ > 1.3) + ASSET + GPT",
        learning_rate=3e-5,
        warmup_steps=100,
        train_pairs_note="~7,291 pairs (thesis)",
    ),
}

BASE_MODEL = "t5-base"
TASK_PREFIX = "summarize: "
