# Exploratory Data Analysis Utilities
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats


def create_barchart_for_feature_and_target(df: pd.DataFrame, feature: str, target: str, flip_target_values: bool = False, rotate_x_lables: bool = True, rename_index_to_strings: bool = False) -> None:
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

    if rename_index_to_strings:
        grouped_counts = grouped_counts.rename(index={0: 'No', 1: 'Yes'})

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
    if rotate_x_lables:
        ax.set_xticklabels(grouped_counts.index, rotation=35, ha="right")
    else:
        ax.set_xticklabels(grouped_counts.index)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(f"{target} Count by {feature}")
    ax.legend()

    # Save plot
    plt.savefig(f'./output/visualizations/{target} Count by {feature}.png')


def calculate_odds_ratio(data: pd.DataFrame, independent_variable: str, dependent_variable: str):
    # Define independent (X) and dependent (y) variables
    data = data.dropna()
    X = sm.add_constant(data[independent_variable]).astype(float)  # Add intercept
    y = data[dependent_variable]

    # Fit logistic regression model
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    # Display model summary
    print(result.summary())

    # Calculate Odds Ratios and Confidence Intervals
    odds_ratios = pd.DataFrame({
        'OR': result.params.apply(lambda x: np.exp(x)),
        'Lower CI': result.conf_int()[0].apply(lambda x: np.exp(x)),
        'Upper CI': result.conf_int()[1].apply(lambda x: np.exp(x))
    })

    print("\nOdds Ratios and 95% Confidence Intervals:")
    print(odds_ratios)


def calculate_point_biserial_corr(data: pd.DataFrame, independent_var: str, dependent_var: str):
    r, p_value = stats.pointbiserialr(data[independent_var], data[dependent_var])

    print(f"Point-Biserial Correlation: {r}")
    print(f"P-Value: {p_value}")