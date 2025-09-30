import pandas as pd

df = pd.read_csv("out", header=None, skipinitialspace=True)

# Keep only the first 10 columns
df = df.iloc[:, :10]

# Convert all columns to numeric, invalid parsing becomes NaN
df = df.apply(pd.to_numeric, errors="coerce")

# Detect rows with any NaN
bad_rows = df[df.isna().any(axis=1)]

if not bad_rows.empty:
    print("Found malformed rows:")
    print(bad_rows)
else:
    print("All rows parsed correctly")
