"""
Script: 01_prepare_train_test_data.py
Purpose: Create reproducible 70:30 training/testing partitions for Lake Erie
         chlorophyll-a regression datasets.

Repository keywords/tags:
# Lake-Erie #chlorophyll-a #remote-sensing #water-quality #machine-learning
# data-preprocessing #train-test-split #reproducible-research

Input:
    Excel workbooks (*.xlsx) in INPUT_DIR. Each workbook contains the full
    observation table on its first worksheet.

Output:
    One workbook per input file, with "train" and "test" worksheets.

The split is random and reproducible (random_state=42). No oversampling or
additional observations are generated.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
INPUT_DIR = Path(r"E:\WQ_Erie_lake\Temporal\Data_testing")
OUTPUT_DIR = INPUT_DIR / "Model data"
TEST_SIZE = 0.30
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# PROCESS ALL EXCEL FILES
# ---------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        p for p in INPUT_DIR.glob("*.xlsx")
        if not p.name.startswith("~$")
    )

    if not input_files:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_DIR}")

    for input_path in input_files:
        print(f"\nProcessing: {input_path.name}")

        data = pd.read_excel(input_path, sheet_name=0)
        if data.empty:
            print("Skipped: workbook contains no observations.")
            continue

        # Reproducible random 70:30 split.
        train_df, test_df = train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        output_path = OUTPUT_DIR / f"{input_path.stem}_train_test.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            train_df.to_excel(writer, sheet_name="train", index=False)
            test_df.to_excel(writer, sheet_name="test", index=False)

        print(
            f"Saved: {output_path}\n"
            f"Training observations: {len(train_df)} | "
            f"Testing observations: {len(test_df)}"
        )

    print("\nData partitioning completed.")


if __name__ == "__main__":
    main()
