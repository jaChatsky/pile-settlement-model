# scripts/run_preprocessing.py
from src.data_preprocessing import load_data_from_excel, clean_and_scale
from src.config import PROC_DIR

import pandas as pd

def main():
    filename = "pile_data.xlsx"
    target_col = "S-mm"

    train_df, test_df, val_df = load_data_from_excel(filename)
    X_train, y_train, X_test, y_test, X_val, y_val = clean_and_scale(train_df, test_df, val_df, target_col)

    # Save processed data
    pd.DataFrame(X_train).to_csv(PROC_DIR / "train_cleaned.csv", index = False)
    y_train.to_csv(PROC_DIR / "train_target.csv", index = False)

    pd.DataFrame(X_test).to_csv(PROC_DIR / "test_cleaned.csv", index = False)
    y_test.to_csv(PROC_DIR / "test_target.csv", index = False)

    pd.DataFrame(X_val).to_csv(PROC_DIR / "val_cleaned.csv", index = False)
    y_val.to_csv(PROC_DIR / "val_target.csv", index = False)

    print(f"Processed data saved to: {PROC_DIR}")

if __name__ == "__main__":
    main()