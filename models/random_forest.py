import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from utils import cross_validation


class RandomForestModel:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        criterion: str = 'gini',
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 42
    ):
        self._model = RandomForestClassifier(
            criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self._X_train = X_train
        self._y_train = y_train
        self._criterion = criterion
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state

    def cross_validation(self, processed_data: pd.DataFrame) -> None:
        cross_validation.do_kfold_cross_validation(
            df=processed_data,
            target_variable="Success vs. Failure",
            model=self._model,
            n_splits=5
        )

    def train(self) -> None:
        self._model.fit(self._X_train, self._y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self._model.predict(X)

    def visualize(self, n: int):
        # Visualize n randomly selected decision trees of the random forest
        selected_trees = random.sample(self._model.estimators_, n)

        # Create subplots
        figsize = (60, 20)
        fig, axes = plt.subplots(1, n, figsize=figsize)

        if n == 1:
            axes = [axes]  # Ensure iterable if only one tree

        # Plot each selected tree
        for i, (tree_model, ax) in enumerate(zip(selected_trees, axes)):
            plot_tree(tree_model, feature_names=self._X_train.columns, class_names=['Success', 'Failure'], filled=True, ax=ax)
            ax.set_title(f'Tree {i + 1}')

        plt.tight_layout()
        plt.savefig('./output/visualizations/random_forest.png')

        # Visualize Feature Importances
        importances = self._model.feature_importances_

        feat_importances = pd.DataFrame({'Feature': self._X_train.columns, 'Importance': importances})
        feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 20))
        plt.yticks(rotation=45)  # Rotate labels 45 degrees
        plt.barh(feat_importances['Feature'], feat_importances['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance in Random Forest')
        plt.subplots_adjust(left=0.35)
        plt.savefig('./output/visualizations/random_forest_feature_importances.png')
