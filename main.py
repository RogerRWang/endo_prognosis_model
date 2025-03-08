from constants import ToothType, SealerType, SuccessFailure, Sex, PracticeLevel
from models.decision_tree import DecisionTreeModel
from models.gradient_boost import GradientBoostModel
from models.random_forest import RandomForestModel
from utils import feature_selection, data_reading, training, eda, analysis


def main():
    # Process data
    processed_data = data_reading.process_raw_data("./data/raw/Final Combined Data 2.xlsx")
    print("Raw Data processed!")
    total_cases = len(processed_data)
    print(f"Total cases: {total_cases}")

    num_successes = (processed_data['Success vs. Failure'] == SuccessFailure.SUCCESS.value).sum()
    num_failures = (processed_data['Success vs. Failure'] == SuccessFailure.FAILURE.value).sum()
    print(f"Num successes: {num_successes}")
    print(f"Num failures: {num_failures}")

    num_male = (processed_data['Sex'] == Sex.MALE.value).sum()
    num_female = (processed_data['Sex'] == Sex.FEMALE.value).sum()
    print(f"Num male: {num_male}")
    print(f"Num female: {num_female}")

    num_resident = (processed_data['Practice Level'] == PracticeLevel.AGEN.value).sum()
    num_faculty = (processed_data['Practice Level'] == PracticeLevel.FGP.value).sum()
    print(f"Num resident: {num_resident}")
    print(f"Num faculty: {num_faculty}")

    processed_data = feature_selection.remove_feature(processed_data, 'Practice Level')
    processed_data = feature_selection.remove_feature(processed_data, 'Sex')
    processed_data = feature_selection.remove_feature(processed_data, 'Diabetes')
    processed_data = feature_selection.remove_feature(processed_data, 'Separated Files')

    # Do any EDA
    chart_data = processed_data.copy()
    chart_data["Tooth Type"] = chart_data["Tooth Type"].map(lambda x: ToothType(x).name)
    eda.create_barchart_for_feature_and_target(chart_data, "Tooth Type", "Success vs. Failure", True)

    chart_data = processed_data.copy()
    chart_data["Sealer Type"] = chart_data["Sealer Type"].map(lambda x: SealerType(x).name)
    eda.create_barchart_for_feature_and_target(chart_data, "Sealer Type", "Success vs. Failure", True)

    # Calculate correlations/chi square matrix
    feature_selection.get_correlation_matrix(processed_data, method="pearson")
    feature_selection.get_correlation_matrix(processed_data, method="spearman")
    feature_selection.get_correlation_matrix(processed_data, method="kendall")
    feature_selection.get_chi_square_matrix(processed_data)

    # Perform one hot encoding on all categorical nominal data
    processed_data = data_reading.do_one_hot_encoding(processed_data)

    num_MandibularAnteriorTooth = (processed_data['Tooth Type_MANDIBULAR_ANTERIOR_TOOTH'] == 1).sum()
    num_MaxillaryPremolarTooth = (processed_data['Tooth Type_MAXILLARY_PREMOLAR_TOOTH'] == 1).sum()
    num_MandibularPremolarTooth = (processed_data['Tooth Type_MANDIBULAR_PREMOLAR_TOOTH'] == 1).sum()
    num_MaxillaryMolarTooth = (processed_data['Tooth Type_MAXILLARY_MOLAR_TOOTH'] == 1).sum()
    num_MandibularMolarTooth = (processed_data['Tooth Type_MANDIBULAR_MOLAR_TOOTH'] == 1).sum()
    # Need to do this (total - the others) b/c we drop MaxillaryAnteriorTooth column to avoid multicolinearity
    num_MaxillaryAnteriorTooth = (total_cases - (num_MandibularAnteriorTooth + num_MaxillaryPremolarTooth + num_MandibularPremolarTooth + num_MaxillaryMolarTooth + num_MandibularMolarTooth))

    print(f"Num MaxillaryAnteriorTooth: {num_MaxillaryAnteriorTooth}")
    print(f"Num MandibularAnteriorTooth: {num_MandibularAnteriorTooth}")
    print(f"Num MaxillaryPremolarTooth: {num_MaxillaryPremolarTooth}")
    print(f"Num MandibularPremolarTooth: {num_MandibularPremolarTooth}")
    print(f"Num MaxillaryMolarTooth: {num_MaxillaryMolarTooth}")
    print(f"Num MandibularMolarTooth: {num_MandibularMolarTooth}")

    # Create training and testing datasets
    X_train, X_test, y_train, y_test = training.prep_data(processed_data, "Success vs. Failure")

    # Train and predict with the models
    # Decision Tree
    print("Starting decision tree.")
    decision_tree = DecisionTreeModel(
        X_train=X_train,
        y_train=y_train,
        criterion='gini',
        max_depth=5,
        random_state=42
    )
    decision_tree.cross_validation(processed_data)
    decision_tree.train()
    y_pred = decision_tree.predict(X=X_test)
    accuracy = training.accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Decision Tree Accuracy: {accuracy}")

    decision_tree.visualize()

    analysis.get_ppv_and_npv(y_pred, y_test, 'decision_tree')

    # Random Forest
    print("Starting random forest.")
    random_forest = RandomForestModel(
        X_train=X_train,
        y_train=y_train,
        criterion='gini',
        n_estimators=30,
        max_depth=5,
        random_state=42
    )
    random_forest.cross_validation(processed_data)
    random_forest.train()
    y_pred = random_forest.predict(X=X_test)
    accuracy = training.accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Random Forest Accuracy: {accuracy}")

    random_forest.visualize(n=1)

    analysis.get_ppv_and_npv(y_pred, y_test, 'random_forest')

    # Gradient Boost
    print("Starting gradient boost.")
    gradient_boost = GradientBoostModel(
        X_train=X_train,
        y_train=y_train,
        criterion='friedman_mse',
        n_estimators=30,
        max_depth=3,
        random_state=42
    )
    gradient_boost.cross_validation(processed_data)
    gradient_boost.train()

    # For gradient boost, we need to remove data with NaNs
    X_test, y_test = GradientBoostModel.clean_data(X_test, y_test)
    y_pred = gradient_boost.predict(X=X_test)
    accuracy = training.accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Gradient Boost Accuracy: {accuracy}")

    gradient_boost.visualize(n=1)

    analysis.get_ppv_and_npv(y_pred, y_test, 'gradient_boost')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
