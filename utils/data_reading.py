import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from constants import Sex, PracticeLevel, TOOTH_TO_TOOTH_TYPE_MAPPING, ToothType, SealerType, RootFillingDensity, \
    RootFillingLength, PreTxPulpalDiagnosis, PreTxPeriapicalDiagnosis, ApicalExtensionOfPost


def read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(
        io=path,
        sheet_name="Complete Data",
        index_col=0,
        usecols="A:T"
    )


def process_raw_data(path: str) -> pd.DataFrame:
    """
    This function will process the raw data into a more standardized format usable for model training.
    It saves it to ./data/processed/processed_data.csv and then returns the new dataframe.

    :param path: Path to raw data
    :return: The dataframe of the processed data
    """
    data = read_excel(path)

    # Make success 1 and failure 0 instead of other way around
    data["Success vs. Failure"] = data["Success vs. Failure"].map(
        {
            0: 1,
            1: 0
        }
    )

    # Clean column names
    data = data.rename(columns={"Satisfactory Coronal Restoration ": "Satisfactory Coronal Restoration"})
    data = data.rename(columns={"Root Fill Density ": "Root Fill Density"})
    data = data.rename(columns={"Root Filled Length ": "Root Filled Length"})
    data = data.rename(columns={"Separated Files ": "Separated Files"})

    # Change string/object columns into int columns
    data["Sex"] = data["Sex"].map(
        {
            "Male": Sex.MALE.value,
            "Female": Sex.FEMALE.value
        }
    )
    data["Practice Level"] = data["Practice Level"].map(
        {
            "AGEN": PracticeLevel.AGEN.value,
            "FGP": PracticeLevel.FGP.value
        }
    )

    # Clean columns with unexpected characters
    data["# of Canals"] = data['# of Canals'].astype(str).str.replace('?', '').astype("Int64")
    data["Pulpal Diagnosis"] = pd.to_numeric(data["Pulpal Diagnosis"], errors="coerce").astype("Int64")
    data["Periapical Diagnosis"] = pd.to_numeric(data["Periapical Diagnosis"], errors="coerce").astype("Int64")
    data["Satisfactory Coronal Restoration"] = pd.to_numeric(data["Satisfactory Coronal Restoration"], errors="coerce").astype("Int64")

    # Map Tooth # to Tooth Type
    def map_tooth_num_to_type(tooth_num: int) -> int:
        for tooth_type, assoc_tooth_nums in TOOTH_TO_TOOTH_TYPE_MAPPING.items():
            if tooth_num in assoc_tooth_nums:
                return tooth_type.value
        raise Exception(f"Provided tooth number {tooth_num} does not map to a tooth type.")
    data.insert(1, "Tooth Type", data["Tooth #"].map(map_tooth_num_to_type))
    data = data.drop(columns=["Tooth #"])

    # Save processed data to CSV
    data.to_csv("./data/processed/processed_data.csv")
    print("Processed data saved to ./data/processed/processed_data.csv")

    return data


def do_one_hot_encoding(data: pd.DataFrame) -> pd.DataFrame:
    # Select categorical columns to encode
    categorical_cols = [
        'Tooth Type',
        'Sealer Type',
        'Root Fill Density',
        'Root Filled Length',
        'Pulpal Diagnosis',
        'Periapical Diagnosis',
        'Apical Extension of Post',
    ]

    # Initialize encoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity

    # Fit and transform
    encoded_array = encoder.fit_transform(data[categorical_cols])

    encoded_columns = encoder.get_feature_names_out(categorical_cols)

    # Convert to DataFrame
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_columns,
        dtype="Int64",
        index=data.index
    )

    # Rename columns for human-friendliness
    def map_column_name(col_name):
        feature, value = col_name.split('_')  # Example: "Tooth Type_2" → ["Tooth Type", "2"]

        if feature == 'Tooth Type':
            return f"Tooth Type_{ToothType(int(value)).name}"
        elif feature == 'Sealer Type':
            return f"Sealer Type_{SealerType(int(value)).name}"
        elif feature == 'Root Fill Density':
            return f"Root Fill Density_{RootFillingDensity(int(value)).name}"
        elif feature == 'Root Filled Length':
            return f"Root Filled Length_{RootFillingLength(int(value)).name}"
        elif feature == 'Pulpal Diagnosis':
            if value == 'nan':
                return "Pulpal Diagnosis_NaN"
            else:
                return f"Pulpal Diagnosis_{PreTxPulpalDiagnosis(int(float(value))).name}"
        elif feature == 'Periapical Diagnosis':
            if value == 'nan':
                return "Periapical Diagnosis_NaN"
            else:
                return f"Periapical Diagnosis_{PreTxPeriapicalDiagnosis(int(float(value))).name}"
        elif feature == 'Apical Extension of Post':
            return f"Apical Extension of Post_{ApicalExtensionOfPost(int(value)).name}"

    # Apply mapping to rename columns
    mapped_column_names = [map_column_name(col) for col in encoded_columns]

    # Rename columns in DataFrame
    encoded_df.columns = mapped_column_names

    # Drop original columns and merge encoded columns
    data = data.drop(columns=categorical_cols).join(encoded_df)

    # Move Success vs. Failure to the end for clarity of the target variable
    data["Success vs. Failure"] = data.pop("Success vs. Failure")

    return data
