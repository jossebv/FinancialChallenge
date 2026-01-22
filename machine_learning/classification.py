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
    TARGET = "class"
    WINDOW_LENGTH = 50
    STANRDIZE = True


    train_data = load_data(DATA_PATH, phase="train", class_eps=0.0025)
    test_data = load_data(DATA_PATH, phase="test", class_eps=0.0025)
    if STANRDIZE:
        train_data, train_mean, train_std = standardize_data(train_data)
        test_data, test_mean, test_std = standardize_data(test_data)
    return (
        MASKED_FEATURES,
        TARGET,
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
    from sklearn.metrics import accuracy_score, f1_score
    import matplotlib.pyplot as plt


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
    return accuracy_score, f1_score, plot_results


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
    TARGET,
    accuracy_score,
    defaultdict,
    f1_score,
    get_X_Y_data,
    test_data,
    train_data,
):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd


    def run_window_search_log_reg(
        window_lenghts: list[int],
    ) -> dict[str, list[float]]:
        features = [
            # "overnight_gap",
            "log_return",
            # "log_range",
            # "volatility_5",
            # "volatility_20",
            # "vol_z",
        ]

        # results = {"mse": [], "mae": []}
        results = defaultdict(list)
        for window_length in window_lenghts:
            X_train, y_train = get_X_Y_data(
                data=train_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=TARGET,
            )
            X_test, y_test = get_X_Y_data(
                data=test_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=TARGET,
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
            results["acc"].append(accuracy_score(y_hat, y_test))
            results["f1"].append(f1_score(y_hat, y_test, average="macro"))
        return results
    return (run_window_search_log_reg,)


@app.cell
def _(plot_results, run_window_search_log_reg):
    # Run and plot the results
    window_lenghts = list(range(2, 100))
    results = run_window_search_log_reg(window_lenghts)
    fig = plot_results(window_lenghts, results, "Logistic Regression")
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
    TARGET,
    accuracy_score,
    defaultdict,
    f1_score,
    get_X_Y_data,
    test_data,
    train_data,
):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    def run_window_search_svc(window_lenghts: list[int]) -> dict[str, list[float]]:
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
                target=TARGET,
            )
            X_test, y_test = get_X_Y_data(
                data=test_data,
                window_length=window_length,
                features=features,
                masked_features=MASKED_FEATURES,
                target=TARGET,
            )

            model = SVC()
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
            results["acc"].append(accuracy_score(y_hat, y_test))
            results["f1"].append(f1_score(y_hat, y_test, average="macro"))
        return results


    def run_params_search_svc(
        window_length: int, select_by: str = "acc"
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
        gamma_grid = ['scale', 0.1, 0.01, 0.001]
        results = defaultdict(list)

        X_train, y_train = get_X_Y_data(
            data=train_data,
            window_length=window_length,
            features=features,
            masked_features=MASKED_FEATURES,
            target=TARGET,
        )
        X_test, y_test = get_X_Y_data(
            data=test_data,
            window_length=window_length,
            features=features,
            masked_features=MASKED_FEATURES,
            target=TARGET,
        )

        ## Perform the search
        best_score = -1
        for C in c_grid:
            for gamma in gamma_grid:
                model = SVC(kernel="rbf", gamma=gamma, C=C)
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)

                curr_acc = accuracy_score(y_test, y_hat)
                curr_f1 = f1_score(y_test, y_hat, average="macro")
                curr_score = curr_acc if select_by == "acc" else curr_f1

                if np.greater(curr_score, best_score):
                    best_score = curr_score
                    best_params = (C, gamma)
                    # keep latest best predictions for final metrics
                    best_acc, best_f1 = curr_acc, curr_f1
                    best_model = model  # already trained with these params

        ## Do the prediction with the best hyperparamenters and evaluate
        print(f"Better params: C={best_params[0]}, eps={best_params[1]}")
        results["acc"].append(best_acc)
        results["f1"].append(best_f1)
        return results
    return run_params_search_svc, run_window_search_svc


@app.cell
def _(plot_results, run_window_search_svc):
    window_lengths_svr = list(range(2, 100))
    results_svr = run_window_search_svc(window_lengths_svr)
    fig_svr = plot_results(window_lengths_svr, results_svr, "SVC")
    fig_svr
    return


@app.cell
def _(run_params_search_svc):
    run_params_search_svc(15)
    return


if __name__ == "__main__":
    app.run()
