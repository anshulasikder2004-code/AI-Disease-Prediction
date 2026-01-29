import pandas as pd

# Load dataset
df = pd.read_csv("../data/symptom_dataset.csv")

# Show first 5 rows
print(df.head())

# Show columns
print(df.columns)

# Show shape (rows, columns)
print(df.shape)
