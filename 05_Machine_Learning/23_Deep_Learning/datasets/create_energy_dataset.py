"""
Create Energy Efficiency Dataset for Deep Learning Regression
Dataset: Building Energy Performance (Heating Load Prediction)

© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 768

print("=" * 80)
print(" Creating Energy Efficiency Dataset ".center(80, "="))
print("=" * 80)
print(f"\n📊 Generating {n_samples} building samples...")

# Generate features based on building characteristics

# 1. Relative Compactness (0.6 to 1.0)
# More compact = less surface area = lower heat loss
relative_compactness = np.random.uniform(0.62, 0.98, n_samples)

# 2. Surface Area (500 to 850 m²)
# Larger surface = more heat loss
surface_area = np.random.uniform(514, 808, n_samples)

# 3. Wall Area (200 to 450 m²)
wall_area = np.random.uniform(245, 416, n_samples)

# 4. Roof Area (100 to 250 m²)
roof_area = np.random.uniform(110, 220, n_samples)

# 5. Overall Height (3.5 to 7.0 meters)
# Taller building = more volume to heat
overall_height = np.random.choice([3.5, 7.0], n_samples)

# 6. Orientation (2, 3, 4, 5 - representing North, East, South, West)
# South-facing = more sunlight = lower heating need
orientation = np.random.choice([2, 3, 4, 5], n_samples)

# 7. Glazing Area (0% to 40% of surface area)
# More windows = more heat loss (but also more sunlight)
glazing_area = np.random.choice([0, 0.1, 0.25, 0.4], n_samples)

# 8. Glazing Area Distribution (0-5)
# 0=uniform, 1=North, 2=East, 3=South, 4=West, 5=mixed
glazing_distribution = np.random.choice([0, 1, 2, 3, 4, 5], n_samples)

# Calculate Heating Load based on building physics
# Formula based on heat transfer principles (simplified model)

heating_load = np.zeros(n_samples)

for i in range(n_samples):
    # Base load from compactness (range: 8-15)
    base = 10 + (1 - relative_compactness[i]) * 15

    # Surface area effect (range: +3 to +8)
    surf_effect = (surface_area[i] - 514) / (808 - 514) * 5 + 3

    # Wall area effect (range: +2 to +5)
    wall_effect = (wall_area[i] - 245) / (416 - 245) * 3 + 2

    # Roof area effect (range: +1 to +3)
    roof_effect = (roof_area[i] - 110) / (220 - 110) * 2 + 1

    # Height effect (3.5m: +0, 7m: +5)
    height_effect = (overall_height[i] - 3.5) / 3.5 * 5

    # Glazing effect (0%: +0, 40%: +6)
    glazing_effect = glazing_area[i] * 15

    # Orientation (South: -3, North: +3, other: 0)
    if orientation[i] == 4:  # South
        orient_effect = -3
    elif orientation[i] == 2:  # North
        orient_effect = +3
    else:
        orient_effect = 0

    # Glazing distribution (South windows: -2, North: +2)
    if glazing_distribution[i] == 3 and glazing_area[i] > 0:  # South
        glaz_dist_effect = -2
    elif glazing_distribution[i] == 1 and glazing_area[i] > 0:  # North
        glaz_dist_effect = +2
    else:
        glaz_dist_effect = 0

    # Total heating load
    heating_load[i] = (
        base + surf_effect + wall_effect + roof_effect +
        height_effect + glazing_effect + orient_effect + glaz_dist_effect
    )

# Add realistic noise
heating_load += np.random.normal(0, 2.5, n_samples)

# Ensure realistic range (10 to 43 kWh/m²)
heating_load = np.clip(heating_load, 10, 43)

# Create DataFrame
df = pd.DataFrame({
    'Relative_Compactness': relative_compactness,
    'Surface_Area': surface_area,
    'Wall_Area': wall_area,
    'Roof_Area': roof_area,
    'Overall_Height': overall_height,
    'Orientation': orientation,
    'Glazing_Area': glazing_area,
    'Glazing_Area_Distribution': glazing_distribution,
    'Heating_Load': heating_load
})

# Round for cleaner data
df = df.round(2)

# Save to CSV
df.to_csv('energy_efficiency.csv', index=False)

print(f"\n✅ Dataset created successfully!")
print(f"\n📊 Dataset Information:")
print(f"   - Rows: {df.shape[0]}")
print(f"   - Columns: {df.shape[1]}")
print(f"   - Features: {df.shape[1] - 1}")
print(f"   - Target: Heating_Load")

print(f"\n📈 Target Statistics:")
print(f"   - Min: {df['Heating_Load'].min():.2f} kWh/m²")
print(f"   - Max: {df['Heating_Load'].max():.2f} kWh/m²")
print(f"   - Mean: {df['Heating_Load'].mean():.2f} kWh/m²")
print(f"   - Std: {df['Heating_Load'].std():.2f} kWh/m²")

print(f"\n💾 Saved to: energy_efficiency.csv")
print(f"\n📝 First 5 rows:")
print(df.head())

print("\n" + "=" * 80)
print(" Dataset Ready for Deep Learning Regression! ".center(80, "="))
print("=" * 80)
