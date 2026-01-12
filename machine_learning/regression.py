import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    from dataloader import get_X_Y_data
    from collections import defaultdict

    sys.path.append("~/Documents/DeepLearning/FinanceChallenge/machine_learning/")

    from dataloader import load_data, standardize_data

    DATA_PATH = "data/processed/es1dia_cln.csv"
    FEATURES = [
        "overnight_gap",
        "log_return",
        "log_range",
        "volatility_5",
        "volatility_20",
        "vol_z",
        "days_elapsed_z",
        "dow",
    ]
    MASKED_FEATURES = ["log_return", "log_range", "vol_z"]
    REG_TARGET = "log_return"
    WINDOW_LENGTH = 50
    STANRDIZE = True


    train_data = load_data(DATA_PATH, phase="train")
    test_data = load_data(DATA_PATH, phase="test")
    if STANRDIZE:
        train_data, train_mean, train_std = standardize_data(train_data)
        test_data, test_mean, test_std = standardize_data(test_data)
    return (
        MASKED_FEATURES,
        REG_TARGET,
        defaultdict,
        get_X_Y_data,
        mo,
        test_data,
        train_data,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Evaluation of the model
    Here are defined the functions necessary to evaluate and obtain the metrics of the different models
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt


    def mse(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return ((prediction - ground_truth) ** 2).mean().item()


    def mae(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.abs(prediction - ground_truth).mean().item()


    def plot_results(
        x: list[int], results: dict[str, list[float]], model_name: str = ""
    ):
        fig, axs = plt.subplots(ncols=2, nrows=1, squeeze=True, figsize=(13, 5))
        for i, (metric, values) in enumerate(results.items()):
            ax = axs[i]
            ax.plot(x, values)
            ax.set_title(f"{metric.upper()} error of {model_name}")
            ax.set_ylabel(metric.upper())
            ax.set_xlabel("window length")
            ax.grid()
        return fig
    return mae, mse, np, plot_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average Baseline
    The baseline used will be a basic average of the log_return values in the input window
    """)
    return


@app.cell
def _(MASKED_FEATURES, REG_TARGET, get_X_Y_data, test_data):
    avg_features = [
        "log_return",
    ]
    avg_window_lenght = 10
    X_test, y_test = get_X_Y_data(
        data=test_data,
        window_length=avg_window_lenght,
        features=avg_features,
        masked_features=MASKED_FEATURES,
        target=REG_TARGET,
    )

    y_hat = X_test.mean(axis=1)

    mse_avg = ((y_hat - y_test) ** 2).mean(axis=0)
    print(f"MSE obtained from average method: {mse_avg:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear Regression

    We will start looking for the best possible length of the window by doing a grid search
    """)
    return


@app.cell
def _(
    MASKED_FEATURES,
    REG_TARGET,
    defaultdict,
    get_X_Y_data,
    mae,
    mse,
    test_data,
    train_data,
):
    from sklearn.linear_model import LinearRegression
    import pandas as pd


    def run_window_search_linear_reg(
        window_lenghts: list[int],
    ) -> dict[str, list[float]]:
        features = [
            "overnight_gap",
            "log_return",
            "log_range",
            "volatility_5",
            "volatility_20",
            "vol_z",
        ]

        # results = {"mse": [], "mae": []}
        results = defaultdict(list)
        for window_length in window_lenghts:
            X_train, y_train = get_X_Y_data(
                data=train_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=REG_TARGET,
            )
            X_test, y_test = get_X_Y_data(
                data=test_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=REG_TARGET,
            )

            linear_regressor = LinearRegression()
            linear_regressor.fit(X_train, y_train)
            y_hat = linear_regressor.predict(X_test)
            results["mse"].append(mse(y_hat, y_test))
            results["mae"].append(mae(y_hat, y_test))
        return results
    return (run_window_search_linear_reg,)


@app.cell
def _(plot_results, run_window_search_linear_reg):
    # Run and plot the results
    window_lenghts = list(range(2, 30))
    results = run_window_search_linear_reg(window_lenghts)
    fig = plot_results(window_lenghts, results, "Linear Regression")
    fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Support Vector Regressor
    The following cells train and evaluate a SVR model
    """)
    return


@app.cell
def _(
    MASKED_FEATURES,
    REG_TARGET,
    defaultdict,
    get_X_Y_data,
    mae,
    mse,
    np,
    test_data,
    train_data,
):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV


    def run_window_search_svr(window_lenghts: list[int]) -> dict[str, list[float]]:
        features = [
            "overnight_gap",
            "log_return",
            "log_range",
            "volatility_5",
            "volatility_20",
            "vol_z",
        ]
        results = defaultdict(list)

        for window_length in window_lenghts:
            X_train, y_train = get_X_Y_data(
                data=train_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=REG_TARGET,
            )
            X_test, y_test = get_X_Y_data(
                data=test_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=REG_TARGET,
            )

            model = SVR()
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
            results["mse"].append(mse(y_hat, y_test))
            results["mae"].append(mae(y_hat, y_test))
        return results


    def run_params_search_svr(
        window_length: int, select_by: str = "mse"
    ) -> dict[str, list[float]]:
        features = [
            "overnight_gap",
            "log_return",
            "log_range",
            "volatility_5",
            "volatility_20",
            "vol_z",
        ]
        c_grid = [0.1, 1, 10, 100, 1000]
        eps_grid = [0.001, 0.01, 0.1, 0.5, 1.0]
        results = defaultdict(list)

        X_train, y_train = get_X_Y_data(
            data=train_data,
            window_length=window_length,
            features=features,
            masked_features=MASKED_FEATURES,
            target=REG_TARGET,
        )
        X_test, y_test = get_X_Y_data(
            data=test_data,
            window_length=window_length,
            features=features,
            masked_features=MASKED_FEATURES,
            target=REG_TARGET,
        )

        ## Perform the search
        best_score = np.inf
        for C in c_grid:
            for eps in eps_grid:
                model = SVR(kernel="rbf", gamma="scale", C=C, epsilon=eps)
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)

                curr_mse = mse(y_test, y_hat)
                curr_mae = mae(y_test, y_hat)
                curr_score = curr_mse if select_by == "mse" else curr_mae

                if np.less(curr_score, best_score):
                    best_score = curr_score
                    best_params = (C, eps)
                    # keep latest best predictions for final metrics
                    best_mse, best_mae = curr_mse, curr_mae
                    best_model = model  # already trained with these params

        ## Do the prediction with the best hyperparamenters and evaluate
        print(f"Better params: C={best_params[0]}, eps={best_params[1]}")
        results["mse"].append(best_mse)
        results["mae"].append(best_mae)
        return results
    return run_params_search_svr, run_window_search_svr


@app.cell
def _(plot_results, run_window_search_svr):
    window_lengths_svr = list(range(2, 100))
    results_svr = run_window_search_svr(window_lengths_svr)
    fig_svr = plot_results(window_lengths_svr, results_svr, "SVR")
    fig_svr
    return


@app.cell
def _(run_params_search_svr):
    run_params_search_svr(59)
    return


if __name__ == "__main__":
    app.run()
