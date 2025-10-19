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
        # Print original column names for debugging
        print(f"Original columns in {name}: {df.columns.tolist()}")
        
        # Clean column names of spaces and special characters
        df.columns = (
            df.columns
            .str.replace("\xa0", " ", regex=False)
            .str.replace("–", "-", regex=False)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        
        # Print cleaned column names for debugging
        print(f"Cleaned columns in {name}: {df.columns.tolist()}")
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

        # Drop unneeded text columns
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns = [col])

        # Convert all columns except categorical ones to numeric
        for col in df.columns:
            if col not in categorical_cols:
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # Verify target exists and is numeric
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found. Check Excel headers: {df.columns.tolist()}")

        df[target_col] = pd.to_numeric(df[target_col], errors = "coerce")
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
    
    # Fit scaler on training data
    scaler.fit(X_train_enc)
    
    # Transform all datasets while preserving column names
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_enc),
        columns = X_train_enc.columns,
        index = X_train_enc.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_enc),
        columns = X_test_enc.columns,
        index = X_test_enc.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_enc),
        columns = X_val_enc.columns,
        index = X_val_enc.index
    )

    feature_names = X_train_enc.columns.tolist()

    return X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val, feature_names

def preprocess_and_save(filename: str, target_col: str):
    """
    Load, clean, encode, and scale data. Save processed arrays and DataFrames to PROC_DIR.
    """
    train_df, test_df, val_df = load_data_from_excel(filename)
    (
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        X_val_scaled,
        y_val,
        feature_names,
    ) = clean_and_scale(train_df, test_df, val_df, target_col)

    # Debugging check: ensure names match array shape
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("Number of feature names:", len(feature_names))
    if X_train_scaled.shape[1] != len(feature_names):
        raise ValueError(f"Mismatch between data columns ({X_train_scaled.shape[1]}) and feature names ({len(feature_names)})")

    # Save numpy arrays (these will lose column names, that's expected)
    np.save(PROC_DIR / "X_train.npy", X_train_scaled)
    np.save(PROC_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROC_DIR / "X_test.npy", X_test_scaled)
    np.save(PROC_DIR / "y_test.npy", y_test.to_numpy())
    np.save(PROC_DIR / "X_val.npy", X_val_scaled)
    np.save(PROC_DIR / "y_val.npy", y_val.to_numpy())

    # FIX: Save CSVs WITH column names - X_train_scaled is already a DataFrame with columns
    X_train_scaled.to_csv(PROC_DIR / "train_cleaned.csv", index=False)
    X_test_scaled.to_csv(PROC_DIR / "test_cleaned.csv", index=False)
    X_val_scaled.to_csv(PROC_DIR / "val_cleaned.csv", index=False)
    
    # Also save target variables
    y_train.to_csv(PROC_DIR / "train_target.csv", index=False)
    y_test.to_csv(PROC_DIR / "test_target.csv", index=False)
    y_val.to_csv(PROC_DIR / "val_target.csv", index=False)
    
    # Save feature names to text file for later reuse
    feature_names_path = PROC_DIR / "feature_names.txt"
    with open(feature_names_path, "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    print(f"Processed data with {len(feature_names)} columns saved to {PROC_DIR}")
    print(f"Feature names list saved to {feature_names_path}")