import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

datasets = ["./train.csv", "./validation.csv", "./test.csv"]

# Load Training Data
training_data = pd.read_csv(datasets[0])
validation_data = pd.read_csv(datasets[1])
test_data = pd.read_csv(datasets[2])
individual_data = training_data.copy()

print("\n--------- PREPROCESSING ANALYSIS USING TRAINING DATA---------\n")

# Missingness assessment
missing_counts = individual_data.isnull().sum()
print("Missing values per column:\n", missing_counts[missing_counts > 0])

# Compare feature distributions with vs without the rows to be dropped.
numeric_cols = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Heart Rate']
check_cols = numeric_cols + ['Previous Complications', 'Preexisting Diabetes']

rows_with_any_missing = individual_data[check_cols].isnull().any(axis=1)
df_complete = individual_data[~rows_with_any_missing]

print(f"\nRows dropped: {rows_with_any_missing.sum()} of {len(individual_data)} "
      f"({rows_with_any_missing.sum() / len(individual_data) * 100:.2f}%)")

comparison = pd.DataFrame({
    'full_mean': individual_data[numeric_cols].mean(),
    'complete_mean': df_complete[numeric_cols].mean(),
    'full_std': individual_data[numeric_cols].std(),
    'complete_std': df_complete[numeric_cols].std(),
})
comparison['mean_diff'] = comparison['complete_mean'] - comparison['full_mean']
comparison['std_diff'] = comparison['complete_std'] - comparison['full_std']
print("\nDistribution comparison, full vs complete-case:\n", comparison)

print("\nRisk Level balance, full data (excl. missing labels):")
print(individual_data[individual_data['Risk Level'].notnull()]['Risk Level'].value_counts(normalize=True))
print("\nRisk Level balance, complete-case only:")
print(df_complete[df_complete['Risk Level'].notnull()]['Risk Level'].value_counts(normalize=True))

# remove columns that aren't features or risk level
model_cols = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI',
               'Previous Complications', 'Preexisting Diabetes',
               'Gestational Diabetes', 'Mental Health', 'Heart Rate', 'Risk Level']

# Apply complete-case deletion
training_df = training_data.dropna(subset=model_cols)
validation_df = validation_data.dropna(subset=model_cols)
test_df = test_data.dropna(subset=model_cols)

original_sizes = {
    "Training": len(training_data),
    "Validation": len(validation_data),
    "Test": len(test_data)
}

print(f"Training: dropped {(original_sizes['Training'] - len(training_df))*100/original_sizes['Training']:.2f}% of records")
print(f"Validation: dropped {(original_sizes['Validation'] - len(validation_df))*100/original_sizes['Validation']:.2f}% of records")
print(f"Test: dropped {(original_sizes['Test'] - len(test_df))*100/original_sizes['Test']:.2f}% of records")

# load cleaned datasets  to csv files
training_df.to_csv("./cleaned_train.csv", index=False)
validation_df.to_csv("./cleaned_validation.csv", index=False)
test_df.to_csv("./cleaned_test.csv", index=False)
