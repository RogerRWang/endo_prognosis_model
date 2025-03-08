# Exploratory Data Analysis Utilities
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def create_barchart_for_feature_and_target(df: pd.DataFrame, feature: str, target: str, flip_target_values: bool = False) -> None:
    """
    Creates a barchart for the count of the target values for each value in the feature

    :param df: The data
    :param feature: Some feature in the data
    :param target: Some binary target variable in the data
    :param flip_target_values: If the target values have 0 for True and 1 for False
    :return:
    """
    data = df[[feature, target]]

    # Count occurrences of Success vs. Failure per Tooth Type
    grouped_counts = data.groupby([feature, target]).size().unstack(fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

    bar_width = 0.4  # Width of each bar
    x = np.arange(len(grouped_counts))  # X-axis positions

    # Plot bars for 0s and 1s
    ax.bar(x - bar_width / 2, grouped_counts[0], width=bar_width, label="Success (0)" if flip_target_values else "Failure (0)", color="green" if flip_target_values else "red", edgecolor="black")
    ax.bar(x + bar_width / 2, grouped_counts[1], width=bar_width, label="Failure (1)" if flip_target_values else "Success (1)", color="red" if flip_target_values else "green", edgecolor="black")

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(grouped_counts.index, rotation=35, ha="right")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(f"{target} Count by {feature}")
    ax.legend()

    # Save plot
    plt.savefig(f'./output/visualizations/{target} Count by {feature}.png')
