"""Scores shared by every model. Same definitions as the benchmark.

- Binary scores treat any class > 0 as "fire".
- IoU compares the predicted and true class curves over time:
  area under min(pred, true) / area under max(pred, true). 1 is perfect.
- Under / over: share of that union where the model predicted too low /
  too high.
The benchmark's AUOC aggregates IoU across departments; with one
department it adds nothing, so we leave it out.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

CLASSES = [0, 1, 2, 3, 4]


def scores(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    true, pred = y_true.to_numpy(float), y_pred.to_numpy(float)
    union = np.trapezoid(np.maximum(true, pred))
    fire_true, fire_pred = true > 0, pred > 0
    return {
        "f1": f1_score(fire_true, fire_pred, zero_division=0),
        "precision": precision_score(fire_true, fire_pred, zero_division=0),
        "recall": recall_score(fire_true, fire_pred, zero_division=0),
        "iou": np.trapezoid(np.minimum(true, pred)) / union,
        "under": np.trapezoid(np.maximum(true - pred, 0)) / union,
        "over": np.trapezoid(np.maximum(pred - true, 0)) / union,
        "fire_days": int(fire_true.sum()),
        "warned_days": int(fire_pred.sum()),
    }


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    m = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return pd.DataFrame(
        m,
        index=pd.Index(CLASSES, name="true"),
        columns=pd.Index(CLASSES, name="predicted"),
    )
