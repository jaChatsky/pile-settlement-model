"""
Data preprocessing utilities for pile settlement prediction.
Handles Excel files with multiple sheets: Training set, Testing set, Validation set.
Encodes categorical columns numerically.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import PROC_DIR, RAW_DIR

def load_data_from_excel(filename: str):
    """
    Load training, testing, and validation sets from Excel sheets
    located in the processed data directory.
    """
    file_path = RAW_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    xls = pd.ExcelFile(file_path)

    def clean_sheet(name):
        df = pd.read_excel(xls, name)
        # Clean column names of spaces and special characters
        df.columns = [
            c.strip().replace("\xa0", " ").replace("–", "-") for c in df.columns
        ]
        return df

    return clean_sheet("Training set"), clean_sheet("Testing set"), clean_sheet("Validation set")

def clean_and_scale(train_df, test_df, val_df, target_col: str):
    """
    Clean and scale data from Excel sheets.
    Keeps categorical columns and encodes them numerically.
    Args:
        train_df, test_df, val_df: DataFrames from each sheet.
        target_col: Name of the target variable ("S-mm").
    Returns:
        X_train, y_train, X_test, y_test, X_val, y_val (numpy arrays)
    """

    categorical_cols = ["Type of test", "Type of pile", "Type of instalation", "End of Pile"]
    drop_cols = ["Reference", "Assumption"]

    def prepare(df):
        # Replace comma decimal separators (e.g., 4,25 → 4.25)
        df = df.replace(",", ".", regex = True)

        # Strip spaces and unify column names
        df.columns = [c.strip() for c in df.columns]

        # Drop unneeded text columns
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns = [col])

        # Convert all columns except categorical ones to numeric (if possible)
        for col in df.columns:
            if col not in categorical_cols:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass  # leave non-convertible columns unchanged

        # If target is stored as string with commas, force numeric conversion
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors = "coerce")
        else:
            raise KeyError(f"Target column '{target_col}' not found. Check Excel headers.")

        # Drop rows missing the target variable
        df = df.dropna(subset = [target_col])

        # Separate features and target
        X = df.drop(columns = [target_col])
        y = df[target_col]
        return X, y

    X_train, y_train = prepare(train_df)
    X_test, y_test = prepare(test_df)
    X_val, y_val = prepare(val_df)

    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
    encoder.fit(X_train[categorical_cols])

    def encode_and_scale(X):
        X_cat = pd.DataFrame(
            encoder.transform(X[categorical_cols]),
            columns = encoder.get_feature_names_out(categorical_cols),
            index = X.index)
        X_num = X.drop(columns = categorical_cols)
        X_combined = pd.concat([X_num, X_cat], axis = 1)
        return X_combined

    X_train_enc = encode_and_scale(X_train)
    X_test_enc = encode_and_scale(X_test)
    X_val_enc = encode_and_scale(X_val)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    X_val_scaled = scaler.transform(X_val_enc)

    return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val

def preprocess_and_save(filename: str, target_col: str):
    """
    Load, clean, encode, and scale data. Save processed arrays to PROC_DIR.
    """
    train_df, test_df, val_df = load_data_from_excel(filename)
    X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val = clean_and_scale(train_df, test_df, val_df, target_col)

    # Save processed numpy arrays
    np.save(PROC_DIR / "X_train.npy", X_train_scaled)
    np.save(PROC_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROC_DIR / "X_test.npy", X_test_scaled)
    np.save(PROC_DIR / "y_test.npy", y_test.to_numpy())
    np.save(PROC_DIR / "X_val.npy", X_val_scaled)
    np.save(PROC_DIR / "y_val.npy", y_val.to_numpy())

    # Save cleaned DataFrames too
    train_df.to_csv(PROC_DIR / "train_cleaned.csv", index = False)
    test_df.to_csv(PROC_DIR / "test_cleaned.csv", index = False)
    val_df.to_csv(PROC_DIR / "val_cleaned.csv", index = False)

    print(f"Processed data saved to: {PROC_DIR}")