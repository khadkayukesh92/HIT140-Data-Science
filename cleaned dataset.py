import pandas as pd

# ================================
# Step 1: Load & Inspect
# ================================
dataset1 = pd.read_csv("dataset1(1).csv")
dataset2 = pd.read_csv("dataset2(1).csv")

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

# Fill missing values in 'habit'
dataset1['habit'] = dataset1['habit'].fillna("unknown")

# Remove impossible values (negative times or counts)
dataset1 = dataset1[dataset1['bat_landing_to_food'] >= 0]
dataset2 = dataset2[(dataset2['rat_arrival_number'] >= 0) & 
                    (dataset2['bat_landing_number'] >= 0)]

# ================================
# Step 3: Convert Data Types
# ================================
# Convert datetime columns
dataset1['start_time'] = pd.to_datetime(dataset1['start_time'], errors='coerce')
dataset1['rat_period_start'] = pd.to_datetime(dataset1['rat_period_start'], errors='coerce')
dataset1['rat_period_end'] = pd.to_datetime(dataset1['rat_period_end'], errors='coerce')
dataset1['sunset_time'] = pd.to_datetime(dataset1['sunset_time'], errors='coerce')
dataset2['time'] = pd.to_datetime(dataset2['time'], errors='coerce')

# Convert categorical variables
dataset1['habit'] = dataset1['habit'].astype('category')
dataset1['season'] = dataset1['season'].astype('category')
dataset1['month'] = dataset1['month'].astype('category')
dataset2['month'] = dataset2['month'].astype('category')

# ================================
# Final Summary After Cleaning
# ================================
print("\n=== Final Dataset1 Summary ===")
print("Shape:", dataset1.shape)
print("Columns and dtypes:\n", dataset1.dtypes)
print("Preview:\n", dataset1.head(), "\n")

print("\n=== Final Dataset2 Summary ===")
print("Shape:", dataset2.shape)
print("Columns and dtypes:\n", dataset2.dtypes)
print("Preview:\n", dataset2.head())
