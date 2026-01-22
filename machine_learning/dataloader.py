import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal


def load_data(
    path: str,
    train_ratio: float = 0.85,
    phase: Literal["train"] | Literal["test"] = "train",
    class_eps: float = 0.0025,
) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=0)
    data.fillna(0, inplace=True)
    N = int(len(data) * train_ratio)
    data = data[:N] if phase == "train" else data[N:]
    data["class"] = pd.cut(
        data["log_return"],
        bins=[-np.inf, -class_eps, class_eps, np.inf],
        labels=["Bearish", "Neutral", "Bullish"],
    )
    return pd.DataFrame(data)


def standardize_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    num_data = data.select_dtypes(include="number")
    mean = num_data.mean()
    std = num_data.std()
    num_data = (num_data - mean) / std
    data = num_data.merge(
        data.select_dtypes(exclude="number"), left_index=True, right_index=True
    )
    return data, mean, std


def get_X_Y_data(
    data: pd.DataFrame,
    window_length: int,
    features: list[str],
    masked_features: list[str],
    target: str,
) -> tuple[np.ndarray, list[float | str]]:
    target_allowed_features = [
        feature for feature in features if feature not in masked_features
    ]
    n_windows = (len(data) - window_length) // window_length + 1
    n_channels = len(features) * (window_length - 1) + sum(
        [1 for feature in features if feature not in masked_features]
    )
    X = np.empty((n_windows, n_channels), dtype=np.float32)
    Y = [0 for _ in range(n_windows)]
    for i, idx in enumerate(range(0, len(data) - window_length + 1, window_length)):
        window = data[idx : idx + window_length]
        x = window[:-1][features].copy()
        if "dow" in features:
            x["dow"] = stats.zscore(x["dow"])

        x = x.to_numpy().flatten()
        target_data = (window[-1:][target_allowed_features]).to_numpy().flatten()
        x = np.concatenate([x, target_data])
        X[i] = x
        Y[i] = window[-1:][target].item()
    return X, Y
