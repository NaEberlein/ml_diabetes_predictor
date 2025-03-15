import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes  
import seaborn as sns


from sklearn.decomposition import PCA
from utils.data_preprocessing import standardize_apply_pca, perform_kmeans


color_defaults = {
    "All": "tab:grey",           
    "Diabetic": "tab:blue",      
    "Not Diabetic": "tab:green" 
}


# ------------------------------
#  Plotting Functions for EDA
# ------------------------------

def plot_class_distribution(data: pd.DataFrame, target: str) -> None:
    """
    Plots the distribution of the target variable.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the target column.
    target (str): The column name for the target variable (e.g., "Outcome").
    """
    class_distribution = data[target].value_counts()
    class_percentage = (class_distribution / len(data)) * 100
    
    plt.figure(figsize=(6,6))
    sns.countplot(
        x=target, 
        data=data, 
        hue=target, 
        palette={0: color_defaults["Not Diabetic"], 1: color_defaults["Diabetic"]},
        legend=False  
    )
    plt.title(f"Class Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Not Diabetic (0)", "Diabetic (1)"])  

    plt.show()
    plt.close()
    
    # Print the class distribution with percentages
    print(f"Class Distribution of {target}:")
    print(class_distribution)
    print(f"\nClass Percentages of {target}:")
    print(class_percentage)


def plot_hist(ax: Axes, data: pd.Series, label: str, color: str, bins: np.ndarray) -> None:
    """
    Helper function to plot a histogram on a given axis.
    
    Parameters:
    ax (Axes): Matplotlib axis to plot the histogram.
    data (Series): The data to plot.
    label (str): The label for the histogram.
    color (str): Color to be used for the plot.
    bins (array): Bins for the histogram.
    """
    sns.histplot(
        data, bins=bins, kde=False, color=color, ax=ax, label=label,
        stat="count", linewidth=2, element="step", alpha=1, fill=False
    )


def plot_distribution_and_boxplot(data: pd.DataFrame, feature: str, target: str) -> None:
    """
    Plots the distribution (histogram) and boxplot for a given feature, divided by diabetic status.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the feature.
    feature (str): The name of the feature to plot.
    target (str): The column name for the target variable (e.g., "Outcome").
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bins = np.linspace(data[feature].min(), data[feature].max(), 18)

    # Define diabetic and not diabetic groups
    groups = {"Diabetic": data[target] == 1, "Not Diabetic": data[target] == 0}
    for label, group in groups.items():
        plot_hist(axes[0], data[feature][group], label, color_defaults[label], bins)

    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim(data[feature].min(), data[feature].max())

    # Prepare data for the boxplot
    boxplot_data = pd.DataFrame({
        "Group": (
            ["All"] * len(data) + 
            ["Diabetic"] * len(data[groups["Diabetic"]]) + 
            ["Not Diabetic"] * len(data[groups["Not Diabetic"]])
        ),
        feature: (
            list(data[feature]) + 
            list(data[feature][groups["Diabetic"]]) + 
            list(data[feature][groups["Not Diabetic"]])
        )
    })

    sns.boxplot(x="Group", y=feature, data=boxplot_data,  hue="Group", palette=color_defaults, ax=axes[1], legend=False)

    axes[1].set_ylabel(feature)
    axes[1].set_xlabel("")
    
    plt.suptitle(f"Distribution and Boxplot of {feature} by Diabetic Status", fontsize=16)
    plt.tight_layout()
    
    plt.show()
    plt.close()


def plot_correlation_matrix(data: pd.DataFrame, correlation_calc: str = "pearson", annot: bool = True,
                            fmt: str = ".2f", cmap: str = "viridis") -> None:
    """
    Plots a correlation matrix heatmap for a given dataset.
    
    Parameters:
    data: The dataset to compute the correlation matrix.
    correlation_calc: The method to compute the correlation ('pearson', 'kendall', 'spearman').
    annot: Whether to annotate the heatmap with correlation values.
    fmt: The format for the annotation.
    cmap: The color map for the heatmap.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(
        data.corr(method=correlation_calc), 
        annot=annot, fmt=fmt, cmap=cmap
    )

    plt.title(f"Correlation Matrix ({correlation_calc})")
    
    plt.show()
    plt.close()


def plot_scatter_matrix_with_kde(data: pd.DataFrame, features: list, target: str, figsize=(10, 8), bw_adjust=1) -> None:
    """
    Create a scatter plot matrix for the given features with KDE contours in the upper triangle.
    Both positive (target=1) and negative (target=0) classes are shown in all areas.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the features.
    features (list): List of feature column names to plot.
    target (str): The name of the target column.
    figsize (tuple): The size of the figure.
    bw_adjust (float): The bandwidth adjustment for the KDE.
    
    """
    pos = data[target] == 1  # Positive class (Diabetic)
    neg = data[target] == 0  # Negative class (Not Diabetic)

    # Create the subplots in a grid layout
    num_features = len(features)
    fig, axes = plt.subplots(num_features, num_features, figsize=figsize)

    # Iterate over all pairs of features
    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])  # Remove axis ticks
            ax.set_xlabel(""); ax.set_ylabel("")  # Remove axis labels
            if i == num_features - 1:  # Set x-axis labels for the bottom-most row
                ax.set_xlabel(features[j], fontsize=10)
            if j == 0:  # Set y-axis labels for the left-most column
                ax.set_ylabel(features[i], fontsize=10)
            # Scatter plot for lower triangle (i > j) and upper triangle (i < j) KDEs
            if i > j:
                # Scatter plot for both classes
                ax.scatter(data[neg][features[j]], data[neg][features[i]], alpha=0.8, color=color_defaults["Not Diabetic"], s=4)
                ax.scatter(data[pos][features[j]], data[pos][features[i]], alpha=0.8, color=color_defaults["Diabetic"], s=4)

            elif i < j:
                # KDE contour plots for both classes
                sns.kdeplot(x=data[pos][features[j]], y=data[pos][features[i]], ax=ax, fill=False, 
                            color=color_defaults["Diabetic"], bw_adjust=bw_adjust, linewidths=0.9, thresh=0.2, levels=4)
                sns.kdeplot(x=data[neg][features[j]], y=data[neg][features[i]], ax=ax, fill=False, 
                            color=color_defaults["Not Diabetic"], bw_adjust=bw_adjust, linewidths=0.9, thresh=0.2, levels=4)

    # Add main title and adjust layout
    plt.suptitle("Scatter Matrix with KDE Contour Plots", fontsize=14)
    plt.tight_layout()  # Adjust layout for title
    plt.show()
    plt.close()


def plot_pca_components(X_pca: np.ndarray, pca: PCA, X: pd.DataFrame, scale_factor: float = 5.0, 
                             xlim: tuple[float, float] = (-4, 4), ylim: tuple[float, float] = (-4, 4)) -> None:
    """
    Plot PCA components along with arrows indicating the feature loadings and show the scale factor.

    Parameters:
    X_pca (np.ndarray): The data after PCA transformation.
    pca (PCA): The PCA model used for transformation (fitted).
    X (pd.DataFrame): Original data used for PCA (for column names).
    scale_factor (float): Scaling factor for the arrows.
    xlim (tuple[float, float]): Limits for the x-axis.
    ylim (tuple[float, float]): Limits for the y-axis.
    """
    plt.figure(figsize=(8, 6))

    # Scatter plot of the first two principal components, color by the first principal component value
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, color='blue')

    # Add arrows for each feature
    for i in range(X.shape[1]):
        plt.arrow(0, 0, pca.components_[0, i] * scale_factor, pca.components_[1, i] * scale_factor,
                  color='black', alpha=0.7, head_width=0.1, head_length=0.1)

        # Adjust the label positioning to avoid overlap
        offset_x = pca.components_[0, i] * scale_factor + 0.3
        offset_y = pca.components_[1, i] * scale_factor + 0.1
        plt.text(offset_x, offset_y, X.columns[i], color='black', ha='center', va='center', fontsize=9)

    # Add information about scaling
    plt.text(xlim[0] + 0.2, ylim[1] - 0.2, f'Arrow Scale Factor: {scale_factor}', ha='left', va='top', fontsize=12)

    # Set plot labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title('PCA Components and Feature Loadings')

    plt.show()
    plt.close()


def plot_comparison_imputation_methods(data_imputation: pd.DataFrame, features_to_plot: list[str]) -> None:
    """
    Create boxplots to compare the imputation of each feature.

    Parameters:
    data_imputation (pd.DataFrame): The dataset containing imputed values others are set to nan and imputation method.
    features_to_plot (list): List of feature names to plot.
    """
    plt.figure(figsize=(15, 8))

    for i, feature in enumerate(features_to_plot):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x="Imputation_Method", y=feature, data=data_imputation)


        num_imputed_per_method = data_imputation.groupby("Imputation_Method")[feature].apply(lambda x: x.notna().sum())
        plt.text(
            0.05, 0.1,
            f'Imputed: {num_imputed_per_method.mean()}',
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

        # Remove x-label
        plt.gca().set_xlabel('')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_rmse_vs_neighbors(rmse_results: dict[str, np.ndarray]) -> None:
    """
    Plots the Mean RMSE vs. n_neighbors for different imputation methods.
    
    Parameters:
    rmse_results: A dictionary where the key is the name of the imputation method (or pipeline),
      and the value is a 2D numpy array of RMSE values for each n_neighbors (1 to 20).
    """
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Loop over each imputation method and plot its RMSE values
    for pipeline_name, rmse_values in rmse_results.items():
        # Plot the mean RMSE for each method across the neighbors range
        plt.plot(range(1, len(rmse_values) + 1), rmse_values.mean(axis=1), label=pipeline_name, marker='o', linestyle='-', markersize=4)    

    plt.title("Mean RMSE vs. n neighbors", fontsize=14)
    plt.xlabel("n neighbors", fontsize=12)
    plt.ylabel("Mean RMSE", fontsize=12)
    plt.legend()
    
    plt.xticks(range(1, len(rmse_values) + 1))
    
    plt.show()
    plt.close()


def plot_kmeans_clusters(axes, i, k_values, X_pca, kmeans_labels, pipeline_name):
    """
    Helper function to plot KMeans clustering results for each value of k.
    """
    for j, k in enumerate(k_values):
        ax = axes[i, j]  # Directly access the axes by 2D indexing
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels[j], cmap='coolwarm', alpha=0.6, edgecolors='k', marker='o')
        ax.set_title(f'KMeans Clustering ({pipeline_name}, k={k})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        
def plot_silhouette_scores(axes, i, k_values, sil_scores, pipeline_name):
    """
    Helper function to plot silhouette scores.
    """
    # The last subplot in the row is reserved for silhouette scores
    ax = axes[i, len(k_values)]  # Last column in each row
    ax.plot(k_values, sil_scores, marker='o')
    ax.set_title(f'Silhouette Score ({pipeline_name})')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')

    
def plot_kmeans_clustering_with_silhouette(data_imputed_dict: dict[str, pd.DataFrame], features: list[str], 
                                           k_max: int = 7) -> None:
    """
    Visualize KMeans clustering for different pipelines and display silhouette scores.

    Parameters:
    data_imputed_dict: Dictionary containing imputed data pipelines as keys and data as values.
    features: List of features to use for clustering.
    k_max: Maximum number of clusters (default is 7).
    """
    num_pipelines = len(data_imputed_dict)
    k_values = range(2, k_max + 1)

    # Set up subplots: one row per pipeline, k_max columns for k-values, and 1 column for silhouette scores
    fig, axes = plt.subplots(num_pipelines, k_max, figsize=(5 * k_max, 5 * num_pipelines))
    # Loop over the pipelines
    for i, (pipeline_name, data_imputed) in enumerate(data_imputed_dict.items()):
        X = data_imputed[features]

        # Apply PCA transformation
        X_pca, _ = standardize_apply_pca(X)

        # Perform KMeans clustering and calculate silhouette scores
        kmeans_labels, sil_scores = perform_kmeans(X, k_values)

        # Plot KMeans clustering results for each k
        plot_kmeans_clusters(axes, i, k_values, X_pca, kmeans_labels, pipeline_name)

        # Plot silhouette scores in the last column
        plot_silhouette_scores(axes, i, k_values, sil_scores, pipeline_name)

    plt.tight_layout()
    plt.show()
    plt.close()


