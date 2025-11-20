import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["timestamp"] = data[["<DTYYYYMMDD>", "<TIME>"]].apply(
        lambda x: f"{x['<DTYYYYMMDD>']} {x['<TIME>']:0>6}", axis=1
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y%m%d %H%M%S")
    data["days_elapsed"] = data["timestamp"].diff().dt.days.fillna(1).astype(int)
    data["days_elapsed_z"] = (
        data["days_elapsed"] - data["days_elapsed"].mean()
    ) / data["days_elapsed"].std()
    data["vol_z"] = (data["<VOL>"] - data["<VOL>"].mean()) / data["<VOL>"].std()
    data["dow"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["year"] = data["timestamp"].dt.year
    data["log_return"] = np.log(data["<CLOSE>"] / data["<OPEN>"])
    data["overnight_gap"] = np.log((data["<OPEN>"] / data["<CLOSE>"].shift(1)))
    data["log_range"] = np.log(data["<HIGH>"] / data["<LOW>"])
    vol_windows = [5, 20]
    returns = np.array(data["log_return"])
    for window in vol_windows:
        data[f"volatility_{window}"] = np.array(
            [
                returns[max(0, i - window) : i].std(ddof=1)  # sample std
                for i in range(len(returns))
            ]
        )

    data.drop(
        columns=["<DTYYYYMMDD>", "<TIME>", "<TICKER>", "<PER>", "<OPENINT>"],
        inplace=True,
    )
    data.rename(
        columns={
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<VOL>": "vol",
        },
        inplace=True,
    )
    print(data.head())
    return data


def load_clean_data(
    path: str,
) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=0)
    data.fillna(0, inplace=True)
    return pd.DataFrame(data)


def plot_data(data: pd.DataFrame, save_path=None):
    plot_attr = ["open", "high", "low", "close", "vol"]

    fig, axs = plt.subplots(
        nrows=len(plot_attr), ncols=1, squeeze=True, figsize=(13, 15)
    )
    for i, ax in enumerate(axs):
        ax.set_title(plot_attr[i])
        ax.plot(data["timestamp"], data[plot_attr[i]])
        ax.grid()

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "raw_data.png"))


def plot_output_data(data: pd.DataFrame, save_path=None):
    fig, axs = plt.subplots(nrows=3, ncols=1, squeeze=True, figsize=(13, 15))

    axs[0].set_title("Close values of S&P500")
    axs[0].plot(data["timestamp"], data["close"].to_numpy())

    axs[1].set_title("Return ratio of S&P500")
    axs[1].plot(data["timestamp"], (data["close"] / data["open"]))

    axs[2].set_title("Return ratio of S&P500")
    axs[2].plot(data["timestamp"], np.log(data["close"] / data["open"]))

    for ax in axs:
        ax.grid()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, "close_over_return_plot.png"), dpi=200)


def plot_input_data(data: pd.DataFrame, save_path=None):
    overnight_gap = np.log((data["open"] / data["close"].shift(1)))
    log_range = np.log(data["high"] / data["low"])
    returns = np.log(data["close"] / data["open"])
    window = 10
    rolling_vol = np.array(
        [
            returns[max(0, i - window + 1) : i + 1].std(ddof=1)  # sample std
            for i in range(len(returns))
        ]
    )

    fig, axs = plt.subplots(nrows=3, ncols=1, squeeze=True, figsize=(13, 15))

    axs[0].set_title("Overnight gap")
    axs[0].plot(data["timestamp"], overnight_gap)

    axs[1].set_title("Log Range {log(H/L)}")
    axs[1].plot(data["timestamp"], log_range)

    axs[2].set_title("Volatility {window=10}")
    axs[2].plot(data["timestamp"], rolling_vol)

    if save_path is not None:
        fig.savefig(os.path.join(save_path, "features_plot.png"), dpi=200)


def plot_cat_corr(data: pd.DataFrame, save_path=None):
    fig = sns.catplot(data=data, kind="violin", x="dow", y="log_return")
    fig.savefig(os.path.join(save_path, "dow_corr.png"))
    fig = sns.catplot(data=data, kind="violin", x="days_elapsed", y="log_return")
    fig.savefig(os.path.join(save_path, "days_elapsed_corr.png"))


def plot_pairs_corr(data: pd.DataFrame, save_path=None):
    corr_cols = [
        "log_return",
        "overnight_gap",
        "log_range",
        "volatility_5",
        "volatility_20",
        "vol",
    ]
    corr = data[corr_cols].corr(method="spearman")
    fig, axs = plt.subplots()
    sns.heatmap(corr, annot=True, ax=axs)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "corr_matrix.png"))

    fig = sns.pairplot(data, vars=corr_cols)

    fig.savefig(os.path.join(save_path, "pairs_corr.png"))


def plot_returns_dist(data: pd.DataFrame, save_path=None):
    fig = sns.displot(data, x="log_return", kind="kde")
    fig.savefig(os.path.join(save_path, "returns_dist.png"))


def plot_log_return_autocorr(data: pd.DataFrame, save_path=None, max_lag: int = 50):
    """Plot autocorrelation of log returns to show dependence on past values."""
    log_returns = data["log_return"].dropna().to_numpy()
    if log_returns.size == 0:
        raise ValueError("No log_return values available for autocorrelation plot.")

    centered = log_returns - log_returns.mean()
    denom = np.sum(centered**2)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        numer = np.sum(centered[:-lag] * centered[lag:])
        acf.append(numer / denom if denom != 0 else 0.0)

    lags = np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    markerline, stemlines, baseline = ax.stem(lags, acf)
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=4)

    conf_level = 1.96 / np.sqrt(log_returns.size)
    ax.axhspan(
        -conf_level,
        conf_level,
        alpha=0.15,
        color="tab:orange",
        label="95% confidence",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, max_lag)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of log returns")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    if len(acf) > 1:
        acf_non_zero = np.array(acf[1:])
        y_min = min(acf_non_zero.min(), -conf_level)
        y_max = max(acf_non_zero.max(), conf_level)
        span = y_max - y_min
        pad = max(0.02, span * 0.15 if span > 0 else 0.05)
        # ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, "log_return_autocorr.png"), dpi=200)


def plot_classes_dist(data, eps: float, save_path: str):
    reg_y = data["log_return"]
    classes = pd.cut(
        reg_y,
        bins=[-np.inf, -eps, eps, np.inf],
        labels=["Bearish", "Neutral", "Bullish"],
    )
    sns.countplot(x=classes, palette=["tab:red", "tab:gray", "tab:green"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of log returns")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "log_return_classes.png"), dpi=200)


if __name__ == "__main__":
    DATA_PATH = "data/raw/es1dia.txt"
    DATA_CLEAN_PATH = "data/processed/es1dia_cln.csv"
    FIG_SAVE_PATH = "EDA/figures"
    # data = load_data(DATA_PATH)
    data = load_clean_data(DATA_CLEAN_PATH)
    # plot_data(data, FIG_SAVE_PATH)
    # plot_input_data(data, FIG_SAVE_PATH)
    # plot_output_data(data, FIG_SAVE_PATH)
    # plot_cat_corr(data, FIG_SAVE_PATH)
    # plot_pairs_corr(data, FIG_SAVE_PATH)
    # plot_returns_dist(data, FIG_SAVE_PATH)
    # plot_log_return_autocorr(data, FIG_SAVE_PATH)
    plot_classes_dist(data, 0.0025, FIG_SAVE_PATH)

    data.to_csv("data/processed/es1dia_cln.csv")
