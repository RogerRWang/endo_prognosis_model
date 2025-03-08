from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import plot_tree

from utils import cross_validation


class GradientBoostModel:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        criterion: str = 'friedman_mse',
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 42
    ):
        self._model = GradientBoostingClassifier(
            criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self._X_train = X_train
        self._y_train = y_train
        self._X_train_cleaned = None
        self._y_train_cleaned = None
        self._criterion = criterion
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state

    def cross_validation(self, processed_data: pd.DataFrame) -> None:
        # For Gradient Boost model, we need to remove rows with NaN values
        processed_data = processed_data.dropna()
        cross_validation.do_kfold_cross_validation(
            df=processed_data,
            target_variable="Success vs. Failure",
            model=self._model,
            n_splits=5
        )

    def train(self) -> None:
        # For Gradient Boost model, we need to remove rows with NaN values
        self._X_train_cleaned, self._y_train_cleaned = self.clean_data(
            self._X_train,
            self._y_train
        )

        self._model.fit(self._X_train_cleaned, self._y_train_cleaned)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self._model.predict(X)

    def visualize(self, n: int):
        # Create subplots
        figsize = (60, 20)
        fig, axes = plt.subplots(1, n, figsize=figsize)

        if n == 1:
            axes = [axes]  # Ensure iterable if only one tree

        # Plot the first n trees in order
        for i in range(n):
            tree_model = self._model.estimators_[i, 0]  # Get the i-th tree
            plot_tree(tree_model, feature_names=self._X_train_cleaned.columns, class_names=['Success', 'Failure'], filled=True, ax=axes[i], fontsize=6)
            axes[i].set_title(f'Tree {i + 1}')

        plt.tight_layout()
        plt.savefig('./output/visualizations/gradient_boost_trees.svg', format='svg')

        # Visualize Feature Importances
        importances = self._model.feature_importances_

        feat_importances = pd.DataFrame({'Feature': self._X_train_cleaned.columns, 'Importance': importances})
        feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 20))
        plt.yticks(rotation=45)  # Rotate labels 45 degrees
        plt.barh(feat_importances['Feature'], feat_importances['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance in Gradient Boost')
        plt.subplots_adjust(left=0.35)
        plt.savefig('./output/visualizations/gradient_boost_feature_importances.png')

    @staticmethod
    def clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        # For Gradient Boost model, we need to remove rows with NaN values
        # Concatenate X and y temporarily
        data_combined = pd.concat([X, y], axis=1)
        # Drop rows with NaN values
        data_cleaned = data_combined.dropna()
        # Separate X and y again
        X_cleaned = data_cleaned.iloc[:, :-1]  # All columns except last (features)
        y_cleaned = data_cleaned.iloc[:, -1]   # Last column (target)

        return X_cleaned, y_cleaned
