import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to save plots
os.makedirs("plots", exist_ok=True)

# ================================
# Step 1: Load & Inspect
# ================================
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

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

dataset1['habit'] = dataset1['habit'].astype('category')
dataset1['season'] = dataset1['season'].astype('category')
dataset1['month'] = dataset1['month'].astype('category')

# Visualizations (Save Plots)

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
save_plot(fig, "bat_risk_distribution.png")

# 2. Risk vs Reward
fig = plt.figure(figsize=(6,4))
ax = sns.countplot(x="risk", hue="reward", data=dataset1)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Risk vs Reward")
plt.xlabel("Risk Behaviour (0=Avoid, 1=Risk)")
plt.ylabel("Count")
plt.legend(title="Reward (0=No, 1=Yes)")
save_plot(fig, "risk_vs_reward.png")

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
save_plot(fig, "reward_by_season.png")

# 4. Distribution of Bat Landings by Hours After Sunset
fig = plt.figure(figsize=(7,4))
sns.histplot(dataset1["hours_after_sunset"], bins=20, kde=True, color="blue")
plt.title("Distribution of Bat Landings by Hours After Sunset")
plt.xlabel("Hours After Sunset")
plt.ylabel("Frequency")
save_plot(fig, "bat_landings_hours_after_sunset.png")

# 5. Distribution of Bat Arrival After Rat Arrival
fig = plt.figure(figsize=(7,4))
sns.histplot(dataset1["seconds_after_rat_arrival"], bins=30, kde=True, color="green")
plt.title("Distribution of Bat Arrival After Rat Arrival")
plt.xlabel("Seconds After Rat Arrival")
plt.ylabel("Frequency")
save_plot(fig, "bat_arrival_after_rat.png")

# 6. Distribution of Bat Landing to Food
fig = plt.figure(figsize=(7,4))
sns.histplot(dataset1["bat_landing_to_food"], bins=30, kde=True, color="blue")
plt.title("Distribution of Bat Landing to Food")
plt.xlabel("Bat Landing to Food (seconds)")
plt.ylabel("Frequency")
save_plot(fig, "bat_landing_to_food.png")

# 7. Bat Landing by Month - Horizontal Bar Plot
fig = plt.figure(figsize=(8,5))
ax = sns.countplot(y="month", data=dataset1, color="skyblue")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bat Landing by Month")
plt.xlabel("Count")
plt.ylabel("Month")
save_plot(fig, "bat_landing_by_month.png")
