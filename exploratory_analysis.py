import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set_theme(style="whitegrid")

# Load Data
training_data = pd.read_csv("./cleaned_train.csv")

df = training_data.copy()

print(f'\n--- Description of {df} ---')
pd.set_option('display.max_columns', None)
print(df.describe())

# Boxplots
binary_cols = ['Previous Complications', 'Preexisting Diabetes',
               'Gestational Diabetes', 'Mental Health']

exploring_numeric_df = df.select_dtypes(exclude=['object']).drop(columns=binary_cols)

n_cols = len(exploring_numeric_df.columns)
n_rows = (n_cols + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3))
axes = axes.flatten()

for i, column in enumerate(exploring_numeric_df.columns):
    sns.boxplot(x=exploring_numeric_df[column], ax=axes[i])
    axes[i].set_title(f'Boxplot of {column}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# histograms
n_cols = len(exploring_numeric_df.columns)
n_rows = (n_cols + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3))
axes = axes.flatten()

for i, column in enumerate(exploring_numeric_df.columns):
    sns.histplot(x=exploring_numeric_df[column], ax=axes[i])
    axes[i].set_title(f'Histogram of {column}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

def get_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (df[column] < lower) | (df[column] > upper)
    return df.loc[mask, [column]].assign(outlier_column=column)

outlier_frames = []
for col in exploring_numeric_df.columns:
    out = get_outliers(exploring_numeric_df, col)
    outlier_frames.append(out)

all_outliers = pd.concat(outlier_frames)

# Summary: how many outliers per column
outlier_counts = all_outliers['outlier_column'].value_counts()
print(outlier_counts)

# BMI 0 outlier
print(df[df['BMI'] == 0])
df.loc[df['BMI'] == 0, 'BMI'] = np.nan
median_bmi = df['BMI'].median()
df['BMI'] = df['BMI'].fillna(median_bmi)

# log transform to deal with skewed data - BS (blood sugar), Age
exploring_numeric_df['BS_log'] = np.log(exploring_numeric_df['BS'])
sns.boxplot(x=exploring_numeric_df['BS_log'])
plt.show()

sns.histplot(x=exploring_numeric_df['BS_log'])
plt.show()

exploring_numeric_df['Age_log'] = np.log(exploring_numeric_df['Age'])
sns.boxplot(x=exploring_numeric_df['Age_log'])
plt.show()

sns.histplot(x=exploring_numeric_df['Age_log'])
plt.show()

exploring_numeric_df['Temp_log'] = np.log(exploring_numeric_df['Body Temp'])
sns.boxplot(x=exploring_numeric_df['Temp_log'])
plt.show()

sns.histplot(x=exploring_numeric_df['Temp_log'])
plt.show()

outlier_frames = []
for col in exploring_numeric_df.columns:
    out = get_outliers(exploring_numeric_df, col)
    outlier_frames.append(out)

all_outliers = pd.concat(outlier_frames)

# Summary: how many outliers per column
outlier_counts = all_outliers['outlier_column'].value_counts()
print("Outlier counts including Log transforms of skewed data columns")
print(outlier_counts)


# explore relationships between features and risk level
numeric_cols = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Heart Rate"
]

fig, axes = plt.subplots((len(numeric_cols) + 2) // 3, 3,
                         figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.violinplot(
        data=df,
        x="Risk Level",
        y=col,
        ax=axes[i]
    )
    axes[i].set_title(f"{col} by Risk Level")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# stacked bar charts for binary features
plot_df = df.copy()

plot_df["Previous Complications"] = plot_df["Previous Complications"].map({
    0: "No Previous Complications",
    1: "Previous Complications"
})

plot_df["Preexisting Diabetes"] = plot_df["Preexisting Diabetes"].map({
    0: "No Pre-existing Diabetes",
    1: "Pre-existing Diabetes"
})

plot_df["Gestational Diabetes"] = plot_df["Gestational Diabetes"].map({
    0: "No Gestational Diabetes",
    1: "Gestational Diabetes"
})

plot_df["Mental Health"] = plot_df["Mental Health"].map({
    0: "No Mental Health Condition",
    1: "Mental Health Condition"
})

binary_cols = [
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(binary_cols):

    ct = pd.crosstab(
        plot_df[col],
        plot_df["Risk Level"],
        normalize="index"
    )

    ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[i],
        legend=False
    )

    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Proportion")
    axes[i].tick_params(axis='x', rotation=0)

# One shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           title="Risk Level",
           loc="upper center",
           ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# chi squared for binary features
for col in binary_cols:
    print(f"\n---- {col} ----")

    crosstab = pd.crosstab(df[col], df["Risk Level"])

    chi2, p, dof, expected = stats.chi2_contingency(crosstab)

    print("Observed")
    print(crosstab)

    print("\nExpected")
    print(pd.DataFrame(expected,
                       index=crosstab.index,
                       columns=crosstab.columns))

# chi squared for binary features 
for col in binary_cols:
    print(f"---- Chi Squared Test for {col} ----")
    crosstab = pd.crosstab(df[col], df['Risk Level'])
    print(f"{crosstab}\n")
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    print(f"Chi2 value= {chi2}\n p-value= {p} \n Degrees of freedom= {dof} \n")

# mann whitney for continuous
for col in numeric_cols:
    print(f"---- Mann Whitney U Test for {col}")
    low = df.loc[df["Risk Level"] == "Low", col]
    high = df.loc[df["Risk Level"] == "High", col]

    u_stat, p = stats.mannwhitneyu(
        low,
        high,
        alternative="two-sided"
    )

    print(f"\n---- {col} ----")
    print(f"U statistic = {u_stat:.1f}")
    print(f"p-value = {p:.4e}")


# multicolinearity exploration
corr_matrix = df[numeric_cols].corr(method='spearman')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Spearman Correlation Between Continuous Features')
plt.tight_layout()
plt.show()

def cramers_v(col1, col2):
    contingency = pd.crosstab(col1, col2)
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

print("---- Cramér's V for binary feature pairs ----")
for i in range(len(binary_cols)):
    for j in range(i + 1, len(binary_cols)):
        v = cramers_v(df[binary_cols[i]], df[binary_cols[j]])
        print(f"{binary_cols[i]} vs {binary_cols[j]}: Cramér's V = {v:.3f}")

print("---- Point-biserial correlation: continuous vs binary ----")
for cont_col in numeric_cols:
    for bin_col in binary_cols:
        r, p = stats.pointbiserialr(df[bin_col], df[cont_col])
        print(f"{cont_col} vs {bin_col}: r = {r:.3f}, p = {p}")
