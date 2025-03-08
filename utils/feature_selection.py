import pandas as pd
from pandas._typing import CorrelationMethod
from scipy.stats import chi2_contingency

from utils import visualize


def get_correlation_matrix(df: pd.DataFrame, method: CorrelationMethod):
    df = df[['Age during Tx', 'Follow-up Time (Months)', '# of Canals']]
    corr_matrix = df.corr(method=method)
    print(f"Calculation of {method.capitalize()} correlation matrix complete!")

    visualize.create_table_viz_from_df(corr_matrix, f"{method.capitalize()} Coefficient Matrix")
    print(f"Saved table visualization for {method.capitalize()} correlation matrix!")


def get_chi_square_matrix(df: pd.DataFrame):
    # df = df.drop(['Age during Tx', 'Follow-up Time (Months)', '# of Canals'], axis=1)
    chi_square_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculate Chi Square every combination of features
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:  # Skip when both columns are the same
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                chi_square_matrix.loc[col1, col2] = p
    print("Calculation of Chi Square matrix complete!")

    visualize.create_table_viz_from_df(chi_square_matrix.astype(float), tablename="Chi-square p-value Matrix", cmap="coolwarm_r")
    print("Saved table visualization for Chi Square p-value matrix!")


def remove_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    return df.drop(feature, axis=1)


def identify_dependent_features():
    # See https://www.statology.org/pearson-correlation-assumptions/ for info on the level of measurement of features
    # Summary
    # Nominal - Categorical features with no natural ordering. E.g. Binary features are typically Nominal (Yes/No).
    # Ordinal - Categorical features with natural ordering. E.g. Pain Scale (Low/Mid/High)
    # Interval - Features with equal intervals between values. E.g. Temperature, where each interval is x degrees.
    # Ratio - Features with equal intervals between values, but has a "true" 0. E.g. Height, where each interval is x cm, but you can't be negative cm.

    # Summary
    # Chi-square is for measuring dependency between nominal features
    # Spearman/Kendall is for measuring dependency between ordinal features
    # Pearson is for measuring dependency between either interval or ratio features

    # For chi-square p-values, p-values <0.05 are likely dependent

    # For Pearson/Spearman/Kendall correlation, values closer to -1 or 1 are more highly correlated/dependent.
    pass
