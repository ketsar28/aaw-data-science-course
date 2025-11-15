"""
Create Sand Production Dataset (Imbalanced Binary Classification)
© Muhammad Ketsar Ali Abi Wahid

This script generates a synthetic sand production dataset for oil & gas wells.
The dataset is intentionally imbalanced (~15% sand production) to simulate
real-world conditions.
"""

import numpy as np
import pandas as pd

# Set random seed
np.random.seed(42)

# Number of samples
n_samples = 5000

print("="*80)
print(" Creating Sand Production Dataset ".center(80, "="))
print("="*80)
print(f"\nGenerating {n_samples} samples...")

# Target distribution (imbalanced!)
# 15% sand production, 85% no sand
n_sand = int(n_samples * 0.15)  # Minority class
n_no_sand = n_samples - n_sand   # Majority class

print(f"\nClass Distribution:")
print(f"  No Sand (Class 0): {n_no_sand} ({n_no_sand/n_samples*100:.1f}%)")
print(f"  Sand (Class 1): {n_sand} ({n_sand/n_samples*100:.1f}%)")
print(f"  Imbalance Ratio: {n_no_sand/n_sand:.2f}:1")

# Generate features for NO SAND cases (Class 0)
# These wells have better conditions
no_sand_data = {
    'Porosity': np.random.normal(0.20, 0.05, n_no_sand).clip(0.05, 0.40),
    'Permeability': np.random.lognormal(1.5, 0.8, n_no_sand).clip(1, 1000),
    'Formation_Strength': np.random.normal(8000, 1500, n_no_sand).clip(3000, 15000),
    'Oil_Production_Rate': np.random.normal(500, 150, n_no_sand).clip(50, 2000),
    'Water_Cut': np.random.normal(0.30, 0.15, n_no_sand).clip(0, 0.80),
    'Gas_Oil_Ratio': np.random.normal(800, 300, n_no_sand).clip(100, 3000),
    'Reservoir_Pressure': np.random.normal(3500, 500, n_no_sand).clip(1500, 6000),
    'Bottom_Hole_Pressure': np.random.normal(2800, 400, n_no_sand).clip(1000, 5000),
    'Drawdown_Pressure': np.random.normal(700, 200, n_no_sand).clip(100, 2000),
    'Sand_Grain_Size': np.random.normal(0.15, 0.04, n_no_sand).clip(0.05, 0.40),
    'Formation_Depth': np.random.normal(2500, 500, n_no_sand).clip(500, 5000),
    'Well_Age_Days': np.random.uniform(1, 3650, n_no_sand),
    'Completion_Type': np.random.choice([0, 1, 2], n_no_sand, p=[0.6, 0.3, 0.1]),  # 0=Cased, 1=OpenHole, 2=Gravel
    'Production_Rate_Change': np.random.normal(0, 50, n_no_sand).clip(-500, 500),
    'Pressure_Decline_Rate': np.random.normal(50, 20, n_no_sand).clip(0, 200),
}

# Generate features for SAND cases (Class 1)
# These wells have conditions favoring sand production
sand_data = {
    'Porosity': np.random.normal(0.28, 0.06, n_sand).clip(0.15, 0.40),  # Higher porosity
    'Permeability': np.random.lognormal(2.5, 1.0, n_sand).clip(50, 2000),  # Higher permeability
    'Formation_Strength': np.random.normal(5000, 1000, n_sand).clip(2000, 10000),  # Lower strength
    'Oil_Production_Rate': np.random.normal(800, 250, n_sand).clip(200, 2500),  # Higher production
    'Water_Cut': np.random.normal(0.50, 0.20, n_sand).clip(0.10, 0.95),  # Higher water
    'Gas_Oil_Ratio': np.random.normal(1200, 400, n_sand).clip(300, 4000),  # Higher GOR
    'Reservoir_Pressure': np.random.normal(4000, 600, n_sand).clip(2000, 7000),  # Higher pressure
    'Bottom_Hole_Pressure': np.random.normal(2500, 500, n_sand).clip(800, 5500),  # Lower BHP
    'Drawdown_Pressure': np.random.normal(1500, 400, n_sand).clip(500, 3000),  # Higher drawdown
    'Sand_Grain_Size': np.random.normal(0.25, 0.06, n_sand).clip(0.10, 0.50),  # Larger grains
    'Formation_Depth': np.random.normal(2000, 600, n_sand).clip(300, 4500),  # Shallower
    'Well_Age_Days': np.random.uniform(365, 3650, n_sand),  # Older wells
    'Completion_Type': np.random.choice([0, 1, 2], n_sand, p=[0.3, 0.5, 0.2]),  # More open hole
    'Production_Rate_Change': np.random.normal(-100, 80, n_sand).clip(-600, 200),  # Declining
    'Pressure_Decline_Rate': np.random.normal(120, 40, n_sand).clip(50, 300),  # Faster decline
}

# Combine data
features = []
for key in no_sand_data.keys():
    features.append(np.concatenate([no_sand_data[key], sand_data[key]]))

# Create labels
labels = np.concatenate([np.zeros(n_no_sand), np.ones(n_sand)])

# Create DataFrame
df = pd.DataFrame({
    'Porosity': features[0],
    'Permeability': features[1],
    'Formation_Strength': features[2],
    'Oil_Production_Rate': features[3],
    'Water_Cut': features[4],
    'Gas_Oil_Ratio': features[5],
    'Reservoir_Pressure': features[6],
    'Bottom_Hole_Pressure': features[7],
    'Drawdown_Pressure': features[8],
    'Sand_Grain_Size': features[9],
    'Formation_Depth': features[10],
    'Well_Age_Days': features[11],
    'Completion_Type': features[12],
    'Production_Rate_Change': features[13],
    'Pressure_Decline_Rate': features[14],
    'Sand_Production': labels.astype(int)
})

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Round numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if col not in ['Completion_Type', 'Sand_Production', 'Well_Age_Days']:
        df[col] = df[col].round(2)

# Save to CSV
df.to_csv('sand_production_data.csv', index=False)

print("\n" + "="*80)
print("✅ Dataset created successfully!")
print("="*80)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📝 Features ({len(df.columns)-1}):")
for i, col in enumerate(df.columns[:-1], 1):
    print(f"  {i:2d}. {col}")

print(f"\n🎯 Target Variable: Sand_Production")
print(f"\n📊 Class Distribution:")
print(df['Sand_Production'].value_counts().to_string())
print(f"\nClass Balance:")
class_counts = df['Sand_Production'].value_counts()
print(f"  Majority (0): {class_counts[0]/len(df)*100:.2f}%")
print(f"  Minority (1): {class_counts[1]/len(df)*100:.2f}%")

print(f"\n📊 First 5 rows:")
print(df.head().to_string())

print(f"\n💾 Saved as: sand_production_data.csv")
print("="*80)
