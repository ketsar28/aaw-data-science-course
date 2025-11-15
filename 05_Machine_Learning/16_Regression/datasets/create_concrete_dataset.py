"""
Create Concrete Compressive Strength Dataset
© Muhammad Ketsar Ali Abi Wahid

This script generates a synthetic concrete compressive strength dataset
based on real-world concrete properties and relationships.
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1030

# Generate features based on realistic ranges and distributions
cement = np.random.uniform(102, 540, n_samples)
blast_furnace_slag = np.random.uniform(0, 359.4, n_samples)
fly_ash = np.random.uniform(0, 200.1, n_samples)
water = np.random.uniform(121.8, 247, n_samples)
superplasticizer = np.random.uniform(0, 32.2, n_samples)
coarse_aggregate = np.random.uniform(801, 1145, n_samples)
fine_aggregate = np.random.uniform(594, 992.6, n_samples)
age = np.random.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], n_samples)

# Calculate concrete compressive strength based on realistic relationships
# Formula is based on known cement science relationships
strength = (
    0.15 * cement +
    0.10 * blast_furnace_slag +
    0.08 * fly_ash -
    0.08 * water +
    2.5 * superplasticizer +
    0.005 * coarse_aggregate +
    0.003 * fine_aggregate +
    0.12 * np.log1p(age)  # Logarithmic relationship with age
    + np.random.normal(0, 5, n_samples)  # Add some noise
)

# Ensure strength is within realistic bounds (2.33 to 82.6 MPa)
strength = np.clip(strength, 2.33, 82.6)

# Create DataFrame
df = pd.DataFrame({
    'Cement': cement,
    'Blast_Furnace_Slag': blast_furnace_slag,
    'Fly_Ash': fly_ash,
    'Water': water,
    'Superplasticizer': superplasticizer,
    'Coarse_Aggregate': coarse_aggregate,
    'Fine_Aggregate': fine_aggregate,
    'Age': age,
    'Concrete_Compressive_Strength': strength
})

# Round to 2 decimal places
df = df.round(2)

# Save to CSV
df.to_csv('concrete_data.csv', index=False)

print("✅ Concrete dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nBasic statistics:")
print(df.describe())
