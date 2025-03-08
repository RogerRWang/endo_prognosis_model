from typing import Tuple

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prep_data(df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preps data for model training

    :param df:              The dataframe we are prepping data for
    :param target_variable: The name of the target variable

    :return: Separate training and test datasets for features (X) and target variable (y)
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
