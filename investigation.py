# ================================
# Import Modules
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from scipy.stats import (
    ttest_ind_from_stats,
    ttest_ind,
)

# Create a folder to save plots
os.makedirs("plots", exist_ok=True)

# ================================
# Load Datasets
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
# Handle Missing Values & Anomalies
# ================================
print("Missing values in Dataset1:\n", dataset1.isnull().sum(), "\n")
print("Missing values in Dataset2:\n", dataset2.isnull().sum(), "\n")

dataset1["habit"] = dataset1["habit"].fillna("unknown")
dataset1 = dataset1[dataset1["bat_landing_to_food"] >= 0]
dataset2 = dataset2[
    (dataset2["rat_arrival_number"] >= 0) & (dataset2["bat_landing_number"] >= 0)
]

# ================================
# Convert Data Types
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
# Dataset 1 Exploratory Analysis
# ================================

# Function to save plot
def save_plot(fig, filename):
    fig.savefig(os.path.join("plots", filename), bbox_inches="tight", dpi=300)
    plt.close(fig) 

# Style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

print("\n" + "=" * 50)
print("DATASET 1 COMPREHENSIVE ANALYSIS")
print("=" * 50)

# Basic statistics for Dataset 1
print("\nDataset 1 Descriptive Statistics:")
print(dataset1.describe())

# 1. Distribution of Bat Risk-Taking Behaviour
fig = plt.figure(figsize=(8, 6))
ax = sns.countplot(x="risk", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Distribution of Bat Risk-Taking Behaviour (0 = Avoid, 1 = Take)")
plt.xlabel("Risk Behaviour")
plt.ylabel("Count")
save_plot(fig, "d1_bat_risk_distribution.png")

# 2. Risk vs Reward
fig = plt.figure(figsize=(8, 6))
ax = sns.countplot(x="risk", hue="reward", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Risk vs Reward")
plt.xlabel("Risk Behaviour (0=Avoid, 1=Risk)")
plt.ylabel("Count")
plt.legend(title="Reward (0=No, 1=Yes)")
save_plot(fig, "d1_risk_vs_reward.png")

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
save_plot(fig, "d1_reward_by_season.png")

# 4. Distribution of Bat Landings by Hours After Sunset
fig = plt.figure(figsize=(10, 6))
sns.histplot(dataset1["hours_after_sunset"], bins=20, kde=True, color="blue")
plt.title("Distribution of Bat Landings by Hours After Sunset")
plt.xlabel("Hours After Sunset")
plt.ylabel("Frequency")
save_plot(fig, "d1_bat_landings_hours_after_sunset.png")

# 5. Distribution of Bat Arrival After Rat Arrival
fig = plt.figure(figsize=(10, 6))
sns.histplot(dataset1["seconds_after_rat_arrival"], bins=30, kde=True, color="green")
plt.title("Distribution of Bat Arrival After Rat Arrival")
plt.xlabel("Seconds After Rat Arrival")
plt.ylabel("Frequency")
save_plot(fig, "d1_bat_arrival_after_rat.png")

# 6. Distribution of Bat Landing to Food
fig = plt.figure(figsize=(10, 6))
sns.histplot(dataset1["bat_landing_to_food"], bins=30, kde=True, color="blue")
plt.title("Distribution of Bat Landing to Food")
plt.xlabel("Bat Landing to Food (seconds)")
plt.ylabel("Frequency")
save_plot(fig, "d1_bat_landing_to_food.png")

# 7. Bat Landing by Month - Horizontal Bar Plot
fig = plt.figure(figsize=(10, 8))
ax = sns.countplot(y="month", data=dataset1, color="skyblue")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bat Landing by Month")
plt.xlabel("Count")
plt.ylabel("Month")
save_plot(fig, "d1_bat_landing_by_month.png")

# ================================
# Dataset 2 Exploratory Analysis
# ================================

print("\n" + "=" * 50)
print("DATASET 2 COMPREHENSIVE ANALYSIS")
print("=" * 50)

# Basic statistics for Dataset 2
print("\nDataset 2 Descriptive Statistics:")
print(dataset2.describe())

# 1. Distribution of Bat Landing Numbers
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
save_plot(fig, "d2_bat_landing_distribution.png")

# 2. Distribution of Rat Arrival Numbers
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
save_plot(fig, "d2_rat_arrival_distribution.png")

# 3. Hours After Sunset Distribution
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
save_plot(fig, "d2_hours_after_sunset.png")

# 4. Food Availability Distribution
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
save_plot(fig, "d2_food_availability.png")

# 5. Group by month: sum bat landings and rat arrivals
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
save_plot(fig, "d2_rat_bat_group_by_month.png")

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
save_plot(fig, "d2_bat_landings_timeseries.png")

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
plt.tight_layout()
save_plot(fig, "d2_rat_vs_bat_scatter.png")

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
plt.tight_layout()
save_plot(fig, "d2_food_vs_bat_scatter.png")

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
save_plot(fig, "d2_activity_by_sunset_hours.png")

# ================================
# Dataset 1 Correlation Analysis
# ================================

d1_numeric_cols = [
    "bat_landing_to_food",
    "seconds_after_rat_arrival",
    "risk",
    "reward",
    "month",
    "hours_after_sunset",
    "season"
]

# Calculate correlation matrix
d1_corr_mat = dataset1[d1_numeric_cols].corr()

# High-quality correlation heatmap
fig = plt.figure(figsize=(12, 10))
sns.heatmap(
    d1_corr_mat,
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt=".3f",
)
plt.title("Correlation Matrix - Dataset 1\n(Upper triangle masked)", fontsize=14)
plt.tight_layout()
save_plot(fig, "d1_correlation_heatmap.png")

# ================================
# Dataset 2: Correlation Analysis
# ================================

d2_numeric_cols = [
    "month",
    "bat_landing_number",
    "food_availability",
    "rat_minutes",
    "hours_after_sunset",
    "rat_arrival_number"
]

# Calculate correlation matrix
d2_corr_mat = dataset2[d2_numeric_cols].corr()

# High-quality correlation heatmap
fig = plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(d2_corr_mat, dtype=bool))
sns.heatmap(
    d2_corr_mat,
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
save_plot(fig, "d2_correlation_heatmap.png")


# ================================
# Dataset 1: Inferential Analysis
# ================================

# ==================================================
# 1: Risk-taking vs risk-avoidance reward rate (two-proportion z-test)
# ==================================================
# Null Hypothesis (H0): Risk-taking bats are equally or more successful than risk-avoiding bats (p1 >= p0).
# Alternative Hypothesis (H1): Risk-taking bats are less successful than risk-avoiding bats (p1 < p0).
# Test: z-test
# =======================================

risk_take = dataset1[dataset1["risk"] == 1]
risk_avoid = dataset1[dataset1["risk"] == 0]

success_take = risk_take["reward"].sum()
success_avoid = risk_avoid["reward"].sum()

n_take = len(risk_take)
n_avoid = len(risk_avoid)

p1 = success_take / n_take
p0 = success_avoid / n_avoid
p = (success_take + success_avoid) / (n_take + n_avoid)

std_err = np.sqrt(p * (1 - p) * (1/n_take + 1/n_avoid))
z = (p1 - p0) / std_err

# One-sided p-value for H1: p1 < p0
p_value = stats.norm.cdf(z)

# Independent SE for CI
std_err_ind = np.sqrt(p1*(1-p1)/n_take + p0*(1-p0)/n_avoid)
ci_low, ci_high = (p1 - p0) - 1.96*std_err_ind, (p1 - p0) + 1.96*std_err_ind

print("\n=== H1: Reward Rate (Risk-taking vs Avoidance) ===")
print(f"Risk-takers success proportion (p1): {p1:.3f}")
print(f"Risk-avoiders success proportion (p0): {p0:.3f}")
print("z =", round(z, 4), "p-value (one-sided, p1<p0) =", round(p_value, 4))
print("95% CI for (p1 - p0):", (round(ci_low, 4), round(ci_high, 4)))
print("Interpretation: If CI < 0 and p < 0.05 → risk-taking has lower success, consistent with rats imposing costs.")

# =======================================
# 2: Bat Landing Delay for Risk-Takers and Risk-Avoiders
# ==================================================
# Null Hypothesis: The mean of bat_landing_to_food is the same for avoiders and risk-takers.
# Alternative Hypothesis: The mean of bat_landing_to_food is different between avoiders and risk-takers.
# Test: Two sample independent t-test
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

print("\n=== H2: Bat Landing Rate (Risk-taking vs Avoidance) ===")
print("t-statistic:", t_stat)
print("one-tailed p-value:", p_val_one_tailed)
print("Mean time avoiders (risk=0):", x_bar_0)
print("Mean time risk-takers (risk=1):", x_bar_1)

# ================================
# Dataset 2: Inferential Analysis
# ================================
print("\n" + "=" * 20)
print("T-TEST ANALYSIS - DATASET 2")
print("=" * 20)

# =======================================
# 3: Food Availability vs Bat Landing
# ==================================================

# Encoding food availability as High Food (1) or Low Food (0)
dataset2["high_food"] = (
    dataset2["food_availability"] > dataset2["food_availability"].median()
).astype(int)

print("\nT-test 1: Bat landings in high vs low food availability periods")
high_food_bats = dataset2[dataset2["high_food"] == 1]["bat_landing_number"]
low_food_bats = dataset2[dataset2["high_food"] == 0]["bat_landing_number"]

# Perform T-test on two samples
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

# =======================================
# 4: Hours after Sunset vs Bat Landing
# ==================================================

# Encoding hours after sunset as Late Hour (1) or Early Hour (0)
dataset2["late_hour"] = (
    dataset2["hours_after_sunset"] > dataset2["hours_after_sunset"].median()
).astype(int)

print("\nT-test 2: Bat landings in early vs late hours after sunset")
early_hour_bats = dataset2[dataset2["late_hour"] == 0]["bat_landing_number"]
late_hour_bats = dataset2[dataset2["late_hour"] == 1]["bat_landing_number"]

# Perform T-test on two samples
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

# =======================================
# 3: Rat Activity (Presence) vs Bat Landing
# ==================================================

# Encoding high_rat_activity (1) if there is at least one rat.
dataset2["high_rat_activity"] = (dataset2["rat_arrival_number"] > 0).astype(int)

print("\nT-test 3: Food availability when rats are present vs absent")
rats_present_food = dataset2[dataset2["high_rat_activity"] == 1]["food_availability"]
rats_absent_food = dataset2[dataset2["high_rat_activity"] == 0]["food_availability"]

# Perform T-test on two samples
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