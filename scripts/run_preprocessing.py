# scripts/run_preprocessing.py
from src.data_preprocessing import preprocess_and_save
from src.config import PROC_DIR

import pandas as pd

def main():
    filename = "pile_data.xlsx"
    target_col = "S-mm"

    preprocess_and_save(filename, target_col)
    print(f"Processing complete! Data saved to: {PROC_DIR}")

if __name__ == "__main__":
    main()