import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from utils import cross_validation


class DecisionTreeModel:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        criterion: str = 'gini',
        max_depth: int = 3,
        random_state: int = 42
    ):
        self._model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
        self._X_train = X_train
        self._y_train = y_train
        self._criterion = criterion
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

    def visualize(self):
        plt.figure(figsize=(19, 6))
        tree.plot_tree(self._model, feature_names=self._X_train.columns, class_names=['Success', 'Failure'], filled=True, fontsize=6)
        plt.savefig('./output/visualizations/decision_tree.svg', format='svg')

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
        plt.title('Feature Importance in Decision Tree')
        plt.subplots_adjust(left=0.35)
        plt.savefig('./output/visualizations/decision_tree_feature_importances.png')
