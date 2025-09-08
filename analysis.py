# ================================
# Import Modules
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import (
    chi2_contingency,
    ttest_ind_from_stats,
    ttest_ind,
    pearsonr,
    spearmanr,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Create a folder to save plots
os.makedirs("plots", exist_ok=True)

# ================================
# Step 1: Load & Inspect
# ================================
base_dir = os.getcwd()
dataset1_path = os.path.join(base_dir, "dataset1.csv")
dataset2_path = os.path.join(base_dir, "dataset2.csv")

dataset1 = pd.read_csv(dataset1_path)
dataset2 = pd.read_csv(dataset2_path)

print("=== Dataset1 Preview ===")
print(dataset1.head(), "\n")

print("=== Dataset2 Preview ===")
print(dataset2.head(), "\n")

print("=== Dataset1 Info ===")
print(dataset1.info(), "\n")

print("=== Dataset2 Info ===")
print(dataset2.info(), "\n")

# ================================
# Step 2: Handle Missing Values & Anomalies
# ================================
print("Missing values in Dataset1:\n", dataset1.isnull().sum(), "\n")
print("Missing values in Dataset2:\n", dataset2.isnull().sum(), "\n")

dataset1["habit"] = dataset1["habit"].fillna("unknown")
dataset1 = dataset1[dataset1["bat_landing_to_food"] >= 0]
dataset2 = dataset2[
    (dataset2["rat_arrival_number"] >= 0) & (dataset2["bat_landing_number"] >= 0)
]

# ================================
# Step 3: Convert Data Types
# ================================
dataset1["start_time"] = pd.to_datetime(
    dataset1["start_time"], errors="coerce", dayfirst=True
)
dataset1["rat_period_start"] = pd.to_datetime(
    dataset1["rat_period_start"], errors="coerce", dayfirst=True
)
dataset1["rat_period_end"] = pd.to_datetime(
    dataset1["rat_period_end"], errors="coerce", dayfirst=True
)
dataset1["sunset_time"] = pd.to_datetime(
    dataset1["sunset_time"], errors="coerce", dayfirst=True
)
dataset2["time"] = pd.to_datetime(dataset2["time"], errors="coerce", dayfirst=True)

dataset1["habit"] = dataset1["habit"].astype("category")
dataset1["season"] = dataset1["season"].astype("category")
dataset1["month"] = dataset1["month"].astype("category")

# ================================
# Step 4: Dataset 1 Exploratory Analysis
# ================================


# Helper function to save plot
def save_plot(fig, filename):
    fig.savefig(os.path.join("plots", filename), bbox_inches="tight", dpi=300)
    plt.close(fig)  # Close the figure to prevent display in notebooks or scripts


# 1. Distribution of Bat Risk-Taking Behaviour
fig = plt.figure(figsize=(8, 6))
ax = sns.countplot(x="risk", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Distribution of Bat Risk-Taking Behaviour (0 = Avoidance, 1 = Risk-Taking)")
plt.xlabel("Risk Behaviour")
plt.ylabel("Count")
save_plot(fig, "bat_risk_distribution.png")

# 2. Risk vs Reward
fig = plt.figure(figsize=(8, 6))
ax = sns.countplot(x="risk", hue="reward", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Risk vs Reward")
plt.xlabel("Risk Behaviour (0=Avoid, 1=Risk)")
plt.ylabel("Count")
plt.legend(title="Reward (0=No, 1=Yes)")
save_plot(fig, "risk_vs_reward.png")

# 3. Risk Behaviour Across Seasons
fig = plt.figure(figsize=(10, 6))
ax = sns.countplot(x="season", hue="reward", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
legend_labels = ["No reward", "Reward"]
ax.legend(title="Reward", labels=legend_labels)
plt.title("Reward Behaviour Across Seasons")
plt.xlabel("Season")
plt.ylabel("Count")
save_plot(fig, "reward_by_season.png")

# 4. Distribution of Bat Landings by Hours After Sunset
fig = plt.figure(figsize=(10, 6))
sns.histplot(dataset1["hours_after_sunset"], bins=20, kde=True, color="blue")
plt.title("Distribution of Bat Landings by Hours After Sunset")
plt.xlabel("Hours After Sunset")
plt.ylabel("Frequency")
save_plot(fig, "bat_landings_hours_after_sunset.png")

# 5. Distribution of Bat Arrival After Rat Arrival
fig = plt.figure(figsize=(10, 6))
sns.histplot(dataset1["seconds_after_rat_arrival"], bins=30, kde=True, color="green")
plt.title("Distribution of Bat Arrival After Rat Arrival")
plt.xlabel("Seconds After Rat Arrival")
plt.ylabel("Frequency")
save_plot(fig, "bat_arrival_after_rat.png")

# 6. Distribution of Bat Landing to Food
fig = plt.figure(figsize=(10, 6))
sns.histplot(dataset1["bat_landing_to_food"], bins=30, kde=True, color="blue")
plt.title("Distribution of Bat Landing to Food")
plt.xlabel("Bat Landing to Food (seconds)")
plt.ylabel("Frequency")
save_plot(fig, "bat_landing_to_food.png")

# 7. Bat Landing by Month - Horizontal Bar Plot
fig = plt.figure(figsize=(10, 8))
ax = sns.countplot(y="month", data=dataset1, color="skyblue")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bat Landing by Month")
plt.xlabel("Count")
plt.ylabel("Month")
save_plot(fig, "bat_landing_by_month.png")

# =======================================
# Step 5: Chi-square test of independence
# Between Risk vs Reward
# =======================================

contingency_table = pd.crosstab(
    dataset1["risk"], dataset1["reward"], rownames=["risk"], colnames=["reward"]
)

print("Contingency Table:")
print(contingency_table, "\n")

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)
print("Expected Frequencies:")
print(expected, "\n")

# Success rate for each group
success_rates = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("Proportion of Reward within each Risk group:")
print(success_rates)

# =======================================
# Step 6: Two sample independent t-test
# Null Hypothesis: The mean of bat_landing_to_food is the same for avoiders and risk-takers.
# Alternative Hypothesis: The mean of bat_landing_to_food is different between avoiders and risk-takers.
# =======================================

# Split the data into two groups
risk0_times = dataset1[dataset1["risk"] == 0]["bat_landing_to_food"]
risk1_times = dataset1[dataset1["risk"] == 1]["bat_landing_to_food"]

# Calculate summary statistics for two samples
x_bar_0, s0, n0 = risk0_times.mean(), risk0_times.std(), len(risk0_times)
x_bar_1, s1, n1 = risk1_times.mean(), risk1_times.std(), len(risk1_times)


# Two-tailed t-test from stats
t_stat, p_val_two_tailed = ttest_ind_from_stats(
    x_bar_1, s1, n1, x_bar_0, s0, n0, equal_var=False
)

# Convert to one-tailed (assuming H1: avoiders wait longer → mean0 > mean1)
if t_stat < 0:
    p_val_one_tailed = 1 - (p_val_two_tailed / 2)
else:
    p_val_one_tailed = p_val_two_tailed / 2

print("t-statistic:", t_stat)
print("one-tailed p-value:", p_val_one_tailed)
print("Mean time avoiders (risk=0):", x_bar_0)
print("Mean time risk-takers (risk=1):", x_bar_1)

# =======================================
# Step 7: Predictive Analysis
# Predicting if each bat landing results to reward or not.
# =======================================

num_cols = ["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset"]
cat_cols = ["risk", "month", "season"]
fig = plt.figure(figsize=(10, 8))
sns.heatmap(
    dataset1[num_cols + ["reward", "risk", "month", "season"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Heatmap - Dataset 1")
save_plot(fig, "dataset1_correlation.png")

# Define features and target
X = dataset1.drop(columns=["habit", "reward"])
y = dataset1["reward"]

# ColumnTransformer: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ]
)

# Pipeline: data preprocessing followed by classification model
pipe = Pipeline(
    [("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
)

# Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit pipeline
pipe.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipe.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# Step 8: Dataset 2 Exploratory Analysis
# ================================

print("\n" + "=" * 50)
print("DATASET 2 COMPREHENSIVE ANALYSIS")
print("=" * 50)

# Basic statistics for Dataset 2
print("\nDataset 2 Descriptive Statistics:")
print(dataset2.describe())

# 1. Group by month: sum bat landings and rat arrivals
dataset2_grouped = (
    dataset2.groupby("month")[["bat_landing_number", "rat_arrival_number"]]
    .sum()
    .reset_index()
)
fig = plt.figure(figsize=(12, 8))
bar_width = 0.35
x = range(len(dataset2_grouped))
plt.bar(
    [i - bar_width / 2 for i in x],
    dataset2_grouped["bat_landing_number"],
    width=bar_width,
    label="Bat Landings",
    color="skyblue",
    alpha=0.8,
)
plt.bar(
    [i + bar_width / 2 for i in x],
    dataset2_grouped["rat_arrival_number"],
    width=bar_width,
    label="Rat Arrivals",
    color="orange",
    alpha=0.8,
)
plt.xticks(x, dataset2_grouped["month"])
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Monthly Comparison of Bat Landings and Rat Arrivals")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_plot(fig, "rat_bat_group_by_month.png")

# ================================
# Dataset 2: Additional High-Quality Plots
# ================================

# 2. Distribution of Bat Landing Numbers
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(
    dataset2["bat_landing_number"], bins=30, kde=True, color="skyblue", alpha=0.7
)
plt.title("Distribution of Bat Landing Numbers")
plt.xlabel("Number of Bat Landings")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(y=dataset2["bat_landing_number"], color="lightblue")
plt.title("Boxplot of Bat Landing Numbers")
plt.ylabel("Number of Bat Landings")
plt.tight_layout()
save_plot(fig, "bat_landing_distribution_dataset2.png")

# 3. Distribution of Rat Arrival Numbers
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(
    dataset2["rat_arrival_number"], bins=30, kde=True, color="orange", alpha=0.7
)
plt.title("Distribution of Rat Arrival Numbers")
plt.xlabel("Number of Rat Arrivals")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(y=dataset2["rat_arrival_number"], color="lightsalmon")
plt.title("Boxplot of Rat Arrival Numbers")
plt.ylabel("Number of Rat Arrivals")
plt.tight_layout()
save_plot(fig, "rat_arrival_distribution_dataset2.png")

# 4. Hours After Sunset Distribution
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(
    dataset2["hours_after_sunset"], bins=25, kde=True, color="purple", alpha=0.7
)
plt.title("Distribution of Hours After Sunset")
plt.xlabel("Hours After Sunset")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(y=dataset2["hours_after_sunset"], color="plum")
plt.title("Boxplot of Hours After Sunset")
plt.ylabel("Hours After Sunset")
plt.tight_layout()
save_plot(fig, "hours_after_sunset_dataset2.png")

# 5. Food Availability Distribution
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(dataset2["food_availability"], bins=25, kde=True, color="green", alpha=0.7)
plt.title("Distribution of Food Availability")
plt.xlabel("Food Availability")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(y=dataset2["food_availability"], color="lightgreen")
plt.title("Boxplot of Food Availability")
plt.ylabel("Food Availability")
plt.tight_layout()
save_plot(fig, "food_availability_dataset2.png")

# 6. Time Series Analysis - Bat Landings Over Time
fig = plt.figure(figsize=(15, 8))
dataset2_sorted = dataset2.sort_values("time")
plt.plot(
    dataset2_sorted["time"],
    dataset2_sorted["bat_landing_number"],
    color="blue",
    alpha=0.6,
    linewidth=1,
)
plt.scatter(
    dataset2_sorted["time"],
    dataset2_sorted["bat_landing_number"],
    color="blue",
    alpha=0.3,
    s=10,
)
plt.title("Bat Landings Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Bat Landings")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plot(fig, "bat_landings_timeseries.png")

# 7. Scatter Plot Matrix for all numeric variables
numeric_cols = [
    "hours_after_sunset",
    "bat_landing_number",
    "food_availability",
    "rat_minutes",
    "rat_arrival_number",
]
fig = plt.figure(figsize=(15, 12))
pd.plotting.scatter_matrix(
    dataset2[numeric_cols], figsize=(15, 12), alpha=0.6, diagonal="hist"
)
plt.suptitle("Scatter Plot Matrix - Dataset 2", fontsize=16, y=0.95)
save_plot(fig, "scatter_matrix_dataset2.png")

# 8. Bat Landings vs Rat Arrivals Scatter Plot
fig = plt.figure(figsize=(10, 8))
plt.scatter(
    dataset2["rat_arrival_number"],
    dataset2["bat_landing_number"],
    alpha=0.6,
    color="purple",
    s=30,
)
plt.xlabel("Rat Arrival Number")
plt.ylabel("Bat Landing Number")
plt.title("Relationship between Rat Arrivals and Bat Landings")
plt.grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(dataset2["rat_arrival_number"], dataset2["bat_landing_number"], 1)
p = np.poly1d(z)
plt.plot(
    dataset2["rat_arrival_number"], p(dataset2["rat_arrival_number"]), "r--", alpha=0.8
)
plt.tight_layout()
save_plot(fig, "rat_vs_bat_scatter.png")

# 9. Food Availability vs Bat Landings
fig = plt.figure(figsize=(10, 8))
plt.scatter(
    dataset2["food_availability"],
    dataset2["bat_landing_number"],
    alpha=0.6,
    color="green",
    s=30,
)
plt.xlabel("Food Availability")
plt.ylabel("Bat Landing Number")
plt.title("Relationship between Food Availability and Bat Landings")
plt.grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(dataset2["food_availability"], dataset2["bat_landing_number"], 1)
p = np.poly1d(z)
plt.plot(
    dataset2["food_availability"], p(dataset2["food_availability"]), "r--", alpha=0.8
)
plt.tight_layout()
save_plot(fig, "food_vs_bat_scatter.png")

# 10. Hours After Sunset vs Activity (Both Bats and Rats)
fig = plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.scatter(
    dataset2["hours_after_sunset"],
    dataset2["bat_landing_number"],
    alpha=0.6,
    color="blue",
    s=20,
    label="Bat Landings",
)
plt.xlabel("Hours After Sunset")
plt.ylabel("Bat Landing Number")
plt.title("Bat Activity by Hours After Sunset")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.scatter(
    dataset2["hours_after_sunset"],
    dataset2["rat_arrival_number"],
    alpha=0.6,
    color="orange",
    s=20,
    label="Rat Arrivals",
)
plt.xlabel("Hours After Sunset")
plt.ylabel("Rat Arrival Number")
plt.title("Rat Activity by Hours After Sunset")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
save_plot(fig, "activity_by_sunset_hours.png")

# ================================
# Dataset 2: Correlation Analysis
# ================================
print("\n" + "=" * 30)
print("CORRELATION ANALYSIS - DATASET 2")
print("=" * 30)

# Calculate correlation matrix
correlation_matrix = dataset2[numeric_cols].corr()
print("\nPearson Correlation Matrix:")
print(correlation_matrix.round(4))

# High-quality correlation heatmap
fig = plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt=".3f",
)
plt.title("Correlation Matrix - Dataset 2\n(Upper triangle masked)", fontsize=14)
plt.tight_layout()
save_plot(fig, "correlation_heatmap_dataset2.png")

# Detailed correlation analysis with p-values
print("\nDetailed Correlation Analysis with P-values:")
print("-" * 60)
for i, col1 in enumerate(numeric_cols):
    for j, col2 in enumerate(numeric_cols):
        if i < j:  # Only upper triangle to avoid duplicates
            corr_coef, p_value = pearsonr(dataset2[col1], dataset2[col2])
            spear_coef, spear_p = spearmanr(dataset2[col1], dataset2[col2])
            print(f"{col1} vs {col2}:")
            print(f"  Pearson r = {corr_coef:.4f}, p = {p_value:.4f}")
            print(f"  Spearman ρ = {spear_coef:.4f}, p = {spear_p:.4f}")
            if p_value < 0.05:
                print(f"  *** Significant correlation (p < 0.05) ***")
            print()

# ================================
# Dataset 2: T-tests
# ================================
print("\n" + "=" * 20)
print("T-TEST ANALYSIS - DATASET 2")
print("=" * 20)

# Create binary variables for t-tests based on median splits
dataset2["high_food"] = (
    dataset2["food_availability"] > dataset2["food_availability"].median()
).astype(int)
dataset2["late_hour"] = (
    dataset2["hours_after_sunset"] > dataset2["hours_after_sunset"].median()
).astype(int)
dataset2["high_rat_activity"] = (dataset2["rat_arrival_number"] > 0).astype(int)

print("\nT-test 1: Bat landings in high vs low food availability periods")
high_food_bats = dataset2[dataset2["high_food"] == 1]["bat_landing_number"]
low_food_bats = dataset2[dataset2["high_food"] == 0]["bat_landing_number"]

t_stat1, p_val1 = ttest_ind(high_food_bats, low_food_bats, equal_var=False)
print(
    f"High food availability - Mean bat landings: {high_food_bats.mean():.2f} (SD: {high_food_bats.std():.2f}, n: {len(high_food_bats)})"
)
print(
    f"Low food availability - Mean bat landings: {low_food_bats.mean():.2f} (SD: {low_food_bats.std():.2f}, n: {len(low_food_bats)})"
)
print(f"t-statistic: {t_stat1:.4f}")
print(f"p-value: {p_val1:.4f}")
print(f"Significant difference: {'Yes' if p_val1 < 0.05 else 'No'}")

print("\nT-test 2: Bat landings in early vs late hours after sunset")
early_hour_bats = dataset2[dataset2["late_hour"] == 0]["bat_landing_number"]
late_hour_bats = dataset2[dataset2["late_hour"] == 1]["bat_landing_number"]

t_stat2, p_val2 = ttest_ind(late_hour_bats, early_hour_bats, equal_var=False)
print(
    f"Early hours - Mean bat landings: {early_hour_bats.mean():.2f} (SD: {early_hour_bats.std():.2f}, n: {len(early_hour_bats)})"
)
print(
    f"Late hours - Mean bat landings: {late_hour_bats.mean():.2f} (SD: {late_hour_bats.std():.2f}, n: {len(late_hour_bats)})"
)
print(f"t-statistic: {t_stat2:.4f}")
print(f"p-value: {p_val2:.4f}")
print(f"Significant difference: {'Yes' if p_val2 < 0.05 else 'No'}")

print("\nT-test 3: Food availability when rats are present vs absent")
rats_present_food = dataset2[dataset2["high_rat_activity"] == 1]["food_availability"]
rats_absent_food = dataset2[dataset2["high_rat_activity"] == 0]["food_availability"]

t_stat3, p_val3 = ttest_ind(rats_present_food, rats_absent_food, equal_var=False)
print(
    f"Rats present - Mean food availability: {rats_present_food.mean():.2f} (SD: {rats_present_food.std():.2f}, n: {len(rats_present_food)})"
)
print(
    f"Rats absent - Mean food availability: {rats_absent_food.mean():.2f} (SD: {rats_absent_food.std():.2f}, n: {len(rats_absent_food)})"
)
print(f"t-statistic: {t_stat3:.4f}")
print(f"p-value: {p_val3:.4f}")
print(f"Significant difference: {'Yes' if p_val3 < 0.05 else 'No'}")

# Visualization of t-test results
fig = plt.figure(figsize=(15, 10))

# T-test 1 visualization
plt.subplot(2, 3, 1)
sns.boxplot(
    x=None,
    y="bat_landing_number",
    hue="high_food",
    data=dataset2,
    palette=["lightcoral", "lightblue"],
    legend=False,
)
plt.title(f"Bat Landings by Food Availability\np-value: {p_val1:.4f}")
plt.xlabel("Food Availability (0=Low, 1=High)")
plt.ylabel("Bat Landing Number")

plt.subplot(2, 3, 2)
sns.violinplot(
    x=None,
    y="bat_landing_number",
    hue="high_food",
    data=dataset2,
    palette=["lightcoral", "lightblue"],
    legend=False,
)
plt.title("Distribution Shape Comparison")
plt.xlabel("Food Availability (0=Low, 1=High)")
plt.ylabel("Bat Landing Number")

# T-test 2 visualization
plt.subplot(2, 3, 3)
sns.boxplot(
    x=None,
    y="bat_landing_number",
    hue="late_hour",
    data=dataset2,
    palette=["lightgreen", "gold"],
    legend=False,
)
plt.title(f"Bat Landings by Time Period\np-value: {p_val2:.4f}")
plt.xlabel("Time Period (0=Early, 1=Late)")
plt.ylabel("Bat Landing Number")

plt.subplot(2, 3, 4)
sns.violinplot(
    x=None,
    y="bat_landing_number",
    hue="late_hour",
    data=dataset2,
    palette=["lightgreen", "gold"],
    legend=False,
)
plt.title("Distribution Shape Comparison")
plt.xlabel("Time Period (0=Early, 1=Late)")
plt.ylabel("Bat Landing Number")

# T-test 3 visualization
plt.subplot(2, 3, 5)
sns.boxplot(
    x=None,
    y="food_availability",
    hue="high_rat_activity",
    data=dataset2,
    palette=["plum", "orange"],
    legend=False,
)
plt.title(f"Food Availability by Rat Presence\np-value: {p_val3:.4f}")
plt.xlabel("Rat Activity (0=Absent, 1=Present)")
plt.ylabel("Food Availability")

plt.subplot(2, 3, 6)
sns.violinplot(
    x=None,
    y="food_availability",
    hue="high_rat_activity",
    data=dataset2,
    palette=["plum", "orange"],
    legend=False,
)
plt.title("Distribution Shape Comparison")
plt.xlabel("Rat Activity (0=Absent, 1=Present)")
plt.ylabel("Food Availability")

plt.tight_layout()
save_plot(fig, "ttest_visualizations_dataset2.png")

# ================================
# Dataset 2: Summary Statistics by Groups
# ================================
print("\n" + "=" * 30)
print("SUMMARY STATISTICS BY GROUPS")
print("=" * 30)

print("\nBat landings by food availability level:")
food_summary = dataset2.groupby("high_food")["bat_landing_number"].agg(
    ["count", "mean", "std", "min", "max"]
)
print(food_summary.round(2))

print("\nBat landings by time period:")
time_summary = dataset2.groupby("late_hour")["bat_landing_number"].agg(
    ["count", "mean", "std", "min", "max"]
)
print(time_summary.round(2))

print("\nFood availability by rat activity:")
rat_summary = dataset2.groupby("high_rat_activity")["food_availability"].agg(
    ["count", "mean", "std", "min", "max"]
)
print(rat_summary.round(2))

print(f"\nTotal plots saved: Multiple high-quality analysis plots for Dataset 2")
print("All plots saved in 'plots' directory with 300 DPI resolution")
