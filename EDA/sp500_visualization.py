import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    DATA_PATH = "data/raw/es1dia.txt"
    FIG_SAVE_PATH = "EDA/figures"
    data = load_data(DATA_PATH)
    plot_data(data, FIG_SAVE_PATH)
    plot_input_data(data, FIG_SAVE_PATH)
    plot_output_data(data, FIG_SAVE_PATH)
    plot_cat_corr(data, FIG_SAVE_PATH)
    plot_pairs_corr(data, FIG_SAVE_PATH)
    plot_returns_dist(data, FIG_SAVE_PATH)

    data.to_csv("data/processed/es1dia_cln.csv")
