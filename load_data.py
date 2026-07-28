import pandas as pd
from sklearn.model_selection import train_test_split

# Load Data
individual_data = pd.read_csv("./Chayan_individual_maternal_dataset.csv")

df = individual_data.copy()

# Missingness assessment (as stratified splitting didn't work due to missing y values)
missing_counts = individual_data.isnull().sum()
print("Missing values per column:\n", missing_counts[missing_counts > 0])

# Test for MAR
# Does missing Risk Level relate to how extreme other vitals are?
# E.g. are patients with extreme values (and therefore possibly
# immediately high risk) more likely to have missing risk levels?
risk_missing = individual_data[individual_data['Risk Level'].isnull()]
risk_present = individual_data[individual_data['Risk Level'].notnull()]

df.dropna(subset=['Risk Level'], inplace=True)

# Splitting into train, validation and testing datasets (60, 20, 20 split)
train_df, test_df = train_test_split(
df,
    test_size=0.2,
    stratify=df["Risk Level"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.25,
    stratify=train_df["Risk Level"],
    random_state=42
)

# Save split data separately to different csv files
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)