import marimo

__generated_with = "0.19.4"
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
    return MASKED_FEATURES, REG_TARGET, get_X_Y_data, mo, test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-Means
    We will evaluate how K-means perform over the data.
    """)
    return


@app.cell
def _(MASKED_FEATURES, REG_TARGET, get_X_Y_data, test_data, train_data):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans


    def train_kmeans_clustering(
        window_length: int = 10, n_clusters=4, random_state=42
    ) -> dict[str, list[float]]:
        features = [
            "overnight_gap",
            "log_return",
            "log_range",
            "volatility_5",
            "volatility_20",
            "vol_z",
        ]

        X_train, _ = get_X_Y_data(
            data=train_data,
            window_length=window_length,
            features=features,
            masked_features=MASKED_FEATURES,
            target=REG_TARGET,
        )
        X_test, _ = get_X_Y_data(
            data=test_data,
            window_length=window_length,
            features=features,
            masked_features=MASKED_FEATURES,
            target=REG_TARGET,
        )

        # 2. Initialize and Train K-Means
        print(f"Training K-Means with k={n_clusters}...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(X_train)
    
        # 3. Predict Clusters
        train_labels = kmeans.labels_  # Labels for training data
        test_labels = kmeans.predict(X_test) # Labels for test data
    
        print("Clustering complete.")
        return kmeans, train_labels, test_labels, X_test

    return np, pd, train_kmeans_clustering


@app.cell
def _(np, pd, train_kmeans_clustering):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def visualize_clusters_pca(X, labels, title="Cluster Visualization (2D PCA)"):
        """
        Reduces the data to 2D using PCA and plots the clusters.
    
        Args:
            X (np.array): The feature data (N, Window, Feat) or (N, Feat).
            labels (np.array): Cluster labels derived from K-Means.
            title (str): Title for the plot.
        """
    
        # 1. Flatten the data if it's 3D time-series data
        if X.ndim > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # 2. Reduce dimensions to 2D using PCA
        print("Applying PCA to reduce dimensions...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_flat)
    
        # 3. Create Scatter Plot
        plt.figure(figsize=(10, 6))
    
        # Scatter plot colored by cluster labels
        scatter = plt.scatter(
            X_pca[:, 0], 
            X_pca[:, 1], 
            c=labels, 
            cmap='viridis', 
            alpha=0.6, 
            edgecolor='k', 
            s=40
        )
    
        # Formatting
        plt.colorbar(scatter, label='Cluster Label')
        plt.title(title, fontsize=14)
        plt.xlabel('Principal Component 1 (Variance)')
        plt.ylabel('Principal Component 2 (Variance)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --- 3. Analysis Function (Generates the Table) ---
    def analyze_cluster_stats(X, labels, returns_index=3):
        """
        Calculates statistics for each cluster to interpret their meaning.
    
        Args:
            X (np.array): Input data. If 3D, assumes shape (N, Window, Features).
            labels (np.array): Cluster labels.
            returns_index (int): Index of the 'Close' or 'Return' feature in the last dimension.
                                 Usually index 3 if features are [Open, High, Low, Close, Volume].
                             
        Returns:
            pd.DataFrame: Summary table of the clusters.
        """
        df_stats = []
    
        # We iterate through each cluster ID (0, 1, 2, 3...)
        for i in np.unique(labels):
            # Select data points belonging to this cluster
            cluster_mask = (labels == i)
            cluster_data = X[cluster_mask]
        
            # Calculate Frequency
            freq = (len(cluster_data) / len(X)) * 100
        
            # Extract Returns for statistics
            # If 3D, we typically look at the return of the *last day* in the window
            # or the mean of the window. Let's use the mean of the window for stability.
            if X.ndim > 2:
                # Slicing: All samples in cluster, All time steps, Feature 'returns_index'
                # Calculate mean return per sample, then mean of the cluster
                cluster_returns = cluster_data[:, :, returns_index] 
                mean_return = np.mean(cluster_returns) 
                volatility = np.std(cluster_returns)
            else:
                # If flat, this is harder to calculate accurately without unflattening.
                # Assuming X is the raw flattened data, we take the mean of the whole vector
                mean_return = np.mean(cluster_data)
                volatility = np.std(cluster_data)

            df_stats.append({
                "Cluster": i,
                "Freq (%)": round(freq, 2),
                "Mean Return": round(mean_return, 6),
                "Volatility": round(volatility, 6),
                "Count": len(cluster_data)
            })
        
        return pd.DataFrame(df_stats)

    kmeans_model, train_labels, test_labels, X_test = train_kmeans_clustering(
        n_clusters=4, 
        random_state=42
    )
    visualize_clusters_pca(X_test, test_labels)
    df_results = analyze_cluster_stats(X_test, test_labels, returns_index=1)
    df_results
    return


if __name__ == "__main__":
    app.run()
