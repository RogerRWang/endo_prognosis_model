import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

from utils.visualize import plot_kfold


def do_kfold_cross_validation(
    df: pd.DataFrame,
    target_variable: str,
    model: BaseEstimator,
    n_splits: int = 5,
) -> None:
    """
    Prepares data using k-fold cross-validation.

    :param df: The dataframe containing features and target variable.
    :param target_variable: The name of the target variable.
    :param model: The model we are performing k-fold cross validation on
    :param n_splits: Number of splits for KFold cross-validation.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_kfold(kf, X, y, ax, n_splits)
    plt.tight_layout()
    fig.subplots_adjust(right=0.6)
    plt.savefig(f'./output/visualizations/kfolds_{str(model)}_dist.png')

    # Get cross validation scores
    for scorer in ['f1', 'precision', 'roc_auc', 'accuracy']:
        scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)

        # Plot the scores
        fig, ax = plt.subplots(figsize=(8, 5))
        fold_labels = [f'Fold {i + 1}' for i in range(len(scores))]

        ax.bar(fold_labels, scores, color='skyblue', edgecolor='black')
        ax.set_ylim(0, 1)  # Assuming accuracy scores range between 0 and 1
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'Cross-Validation Scores per Fold ({scorer})')

        # Annotate each bar with the accuracy value
        for i, score in enumerate(scores):
            ax.text(i, score + 0.02, f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./output/visualizations/kfolds_{str(model)}_{scorer}_scores.png')
