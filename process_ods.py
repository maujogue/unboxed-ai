import pandas as pd
import os

# Check for file existence
file_path = "assets/new-reports.ods"
if not os.path.exists(file_path):
    file_path = "assets/new_reports.ods"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

print(f"Reading file: {file_path}")

try:
    # Read the ODS file
    df = pd.read_excel(file_path, engine="odf")

    # Print columns to help debug if needed
    print("Columns found:", df.columns.tolist())

    # Check if 'clinical information data' exists
    col_name = "clinical information data"
    if col_name not in df.columns:
        # Try to find a similar column name (case insensitive or partial match)
        possible_cols = [c for c in df.columns if "clinical" in str(c).lower()]
        if possible_cols:
            col_name = possible_cols[0]
            print(f"Using column '{col_name}' instead of '{col_name}'")
        else:
            print(f"Column '{col_name}' not found.")
            exit(1)

    # Add 'fait' column
    # Ensure the column is string type before calculating length
    df[col_name] = df[col_name].astype(str)
    df["fait"] = df[col_name].apply(lambda x: len(x) > 100)

    # Save the new dataframe
    output_path = "assets/processed_reports.ods"
    df.to_excel(output_path, engine="odf", index=False)
    print(f"Saved processed file to: {output_path}")

    # Print the entire dataframe
    print("\nProcessed DataFrame:")
    print(df)

except Exception as e:
    print(f"An error occurred: {e}")
