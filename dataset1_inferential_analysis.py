# ================================
# Import Modules
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency, ttest_ind_from_stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

dataset1['habit'] = dataset1['habit'].fillna("unknown")
dataset1 = dataset1[dataset1['bat_landing_to_food'] >= 0]
dataset2 = dataset2[(dataset2['rat_arrival_number'] >= 0) & 
                    (dataset2['bat_landing_number'] >= 0)]

# ================================
# Step 3: Convert Data Types
# ================================
dataset1['start_time'] = pd.to_datetime(dataset1['start_time'], errors='coerce', dayfirst=True)
dataset1['rat_period_start'] = pd.to_datetime(dataset1['rat_period_start'], errors='coerce', dayfirst=True)
dataset1['rat_period_end'] = pd.to_datetime(dataset1['rat_period_end'], errors='coerce', dayfirst=True)
dataset1['sunset_time'] = pd.to_datetime(dataset1['sunset_time'], errors='coerce', dayfirst=True)
dataset2['time'] = pd.to_datetime(dataset2['time'], errors='coerce', dayfirst=True)

dataset1['habit'] = dataset1['habit'].astype('category')
dataset1['season'] = dataset1['season'].astype('category')
dataset1['month'] = dataset1['month'].astype('category')

# ================================
# Step 4: Dataset 1 Exploratory Analysis
# ================================

# Helper function to save plot
def save_plot(fig, filename):
    fig.savefig(os.path.join("plots", filename), bbox_inches='tight')
    plt.close(fig)  # Close the figure to prevent display in notebooks or scripts

# 1. Distribution of Bat Risk-Taking Behaviour
fig = plt.figure(figsize=(6,4))
ax = sns.countplot(x="risk", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Distribution of Bat Risk-Taking Behaviour (0 = Avoidance, 1 = Risk-Taking)")
plt.xlabel("Risk Behaviour")
plt.ylabel("Count")
#save_plot(fig, "bat_risk_distribution.png")

# 2. Risk vs Reward
fig = plt.figure(figsize=(6,4))
ax = sns.countplot(x="risk", hue="reward", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Risk vs Reward")
plt.xlabel("Risk Behaviour (0=Avoid, 1=Risk)")
plt.ylabel("Count")
plt.legend(title="Reward (0=No, 1=Yes)")
#save_plot(fig, "risk_vs_reward.png")

# 3. Risk Behaviour Across Seasons
fig = plt.figure(figsize=(8,5))
ax = sns.countplot(x="season", hue="reward", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
legend_labels = ['No reward', 'Reward']
ax.legend(title='Reward', labels=legend_labels)
plt.title("Reward Behaviour Across Seasons")
plt.xlabel("Season")
plt.ylabel("Count")
#save_plot(fig, "reward_by_season.png")

# 4. Distribution of Bat Landings by Hours After Sunset
fig = plt.figure(figsize=(7,4))
sns.histplot(dataset1["hours_after_sunset"], bins=20, kde=True, color="blue")
plt.title("Distribution of Bat Landings by Hours After Sunset")
plt.xlabel("Hours After Sunset")
plt.ylabel("Frequency")
#save_plot(fig, "bat_landings_hours_after_sunset.png")

# 5. Distribution of Bat Arrival After Rat Arrival
fig = plt.figure(figsize=(7,4))
sns.histplot(dataset1["seconds_after_rat_arrival"], bins=30, kde=True, color="green")
plt.title("Distribution of Bat Arrival After Rat Arrival")
plt.xlabel("Seconds After Rat Arrival")
plt.ylabel("Frequency")
#save_plot(fig, "bat_arrival_after_rat.png")

# 6. Distribution of Bat Landing to Food
fig = plt.figure(figsize=(7,4))
sns.histplot(dataset1["bat_landing_to_food"], bins=30, kde=True, color="blue")
plt.title("Distribution of Bat Landing to Food")
plt.xlabel("Bat Landing to Food (seconds)")
plt.ylabel("Frequency")
#save_plot(fig, "bat_landing_to_food.png")

# 7. Bat Landing by Month - Horizontal Bar Plot
fig = plt.figure(figsize=(8,5))
ax = sns.countplot(y="month", data=dataset1, color="skyblue")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bat Landing by Month")
plt.xlabel("Count")
plt.ylabel("Month")
#save_plot(fig, "bat_landing_by_month.png")

# =======================================
# Step 5: Chi-square test of independence
# Between Risk vs Reward
# =======================================

contingency_table = pd.crosstab(
    dataset1['risk'], 
    dataset1['reward'], 
    rownames=['risk'], 
    colnames=['reward']
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
risk0_times = dataset1[dataset1['risk'] == 0]['bat_landing_to_food']
risk1_times = dataset1[dataset1['risk'] == 1]['bat_landing_to_food']

# Calculate summary statistics for two samples
x_bar_0, s0, n0 = risk0_times.mean(), risk0_times.std(), len(risk0_times)
x_bar_1, s1, n1 = risk1_times.mean(), risk1_times.std(), len(risk1_times)


# Two-tailed t-test from stats
t_stat, p_val_two_tailed = ttest_ind_from_stats(x_bar_1, s1, n1,
                                                x_bar_0, s0, n0,
                                                equal_var=False)

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

num_cols = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
cat_cols = ['risk', 'month', 'season']
fig = plt.figure(figsize=(8,6))
sns.heatmap(dataset1[num_cols + ['reward', 'risk', 'month', 'season']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
save_plot(fig, "dataset1_correlation.png")

# Define features and target
X = dataset1.drop(columns=['habit', 'reward'])
y = dataset1['reward']

# ColumnTransformer: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

# Pipeline: data preprocessing followed by classification model
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# 1. Group by month: sum bat landings and rat arrivals
dataset2_grouped = dataset2.groupby('month')[['bat_landing_number', 'rat_arrival_number']].sum().reset_index()
fig = plt.figure(figsize=(10,6))
bar_width = 0.35
x = range(len(dataset2_grouped))
plt.bar([i - bar_width/2 for i in x], dataset2_grouped['bat_landing_number'], 
        width=bar_width, label='Bat Landings', color='skyblue')
plt.bar([i + bar_width/2 for i in x], dataset2_grouped['rat_arrival_number'], 
        width=bar_width, label='Rat Arrivals', color='orange')
plt.xticks(x, dataset2_grouped['month'])
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Monthly Comparison of Bat Landings and Rat Arrivals')
plt.legend()
plt.tight_layout()
save_plot(fig, "rat_bat_group_by_month.png")