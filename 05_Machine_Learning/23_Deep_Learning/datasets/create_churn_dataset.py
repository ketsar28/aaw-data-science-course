"""
Create Bank Customer Churn Dataset for Deep Learning Classification
Dataset: Customer Churn Prediction (Imbalanced Binary Classification)

© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 10000

print("=" * 80)
print(" Creating Bank Customer Churn Dataset ".center(80, "="))
print("=" * 80)
print(f"\n📊 Generating {n_samples} customer samples...")

# Target distribution (imbalanced)
# 20% churn, 80% stay (realistic banking scenario)
churn_ratio = 0.20
n_churn = int(n_samples * churn_ratio)
n_stay = n_samples - n_churn

print(f"\n⚖️ Class Distribution:")
print(f"   - Churn (1): {n_churn} customers ({churn_ratio*100:.0f}%)")
print(f"   - Stay (0): {n_stay} customers ({(1-churn_ratio)*100:.0f}%)")

# Generate features for STAY customers (better characteristics)
print(f"\n🔧 Generating features for STAY customers...")

stay_data = {
    # 1. Credit Score (higher = more stable)
    'CreditScore': np.random.normal(650, 80, n_stay).clip(350, 850),

    # 2. Age (loyal customers tend to be middle-aged)
    'Age': np.random.normal(42, 12, n_stay).clip(18, 80),

    # 3. Tenure (years with bank - longer = more loyal)
    'Tenure': np.random.exponential(5, n_stay).clip(0, 10),

    # 4. Balance (higher balance = less likely to leave)
    'Balance': np.random.lognormal(10.5, 1.2, n_stay).clip(0, 250000),

    # 5. Number of Products (1-4)
    'NumOfProducts': np.random.choice([1, 2, 3, 4], n_stay, p=[0.3, 0.5, 0.15, 0.05]),

    # 6. Has Credit Card (1=Yes, 0=No) - more engaged customers
    'HasCrCard': np.random.choice([0, 1], n_stay, p=[0.2, 0.8]),

    # 7. Is Active Member (more engaged = less churn)
    'IsActiveMember': np.random.choice([0, 1], n_stay, p=[0.3, 0.7]),

    # 8. Estimated Salary
    'EstimatedSalary': np.random.uniform(20000, 150000, n_stay),

    # 9. Gender (0=Female, 1=Male)
    'Gender': np.random.choice([0, 1], n_stay),

    # 10. Geography (0=France, 1=Germany, 2=Spain)
    'Geography': np.random.choice([0, 1, 2], n_stay, p=[0.5, 0.3, 0.2]),

    # Target
    'Exited': np.zeros(n_stay)
}

# Generate features for CHURN customers (worse characteristics)
print(f"🔧 Generating features for CHURN customers...")

# Age generation (very young or very old more likely to churn)
age_young = np.random.normal(28, 5, n_churn//2).clip(18, 40)
age_old = np.random.normal(58, 8, n_churn - n_churn//2).clip(50, 80)
age_churn = np.concatenate([age_young, age_old])

churn_data = {
    # 1. Credit Score (lower = less stable)
    'CreditScore': np.random.normal(580, 90, n_churn).clip(350, 850),

    # 2. Age (very young or very old more likely to churn)
    'Age': age_churn,

    # 3. Tenure (shorter tenure = higher churn)
    'Tenure': np.random.exponential(2, n_churn).clip(0, 10),

    # 4. Balance (very low or zero balance)
    'Balance': np.random.lognormal(8, 2, n_churn).clip(0, 250000),

    # 5. Number of Products (fewer products = less engaged)
    'NumOfProducts': np.random.choice([1, 2, 3, 4], n_churn, p=[0.6, 0.3, 0.08, 0.02]),

    # 6. Has Credit Card (less likely to have)
    'HasCrCard': np.random.choice([0, 1], n_churn, p=[0.4, 0.6]),

    # 7. Is Active Member (inactive = more likely to churn)
    'IsActiveMember': np.random.choice([0, 1], n_churn, p=[0.7, 0.3]),

    # 8. Estimated Salary (similar distribution)
    'EstimatedSalary': np.random.uniform(20000, 150000, n_churn),

    # 9. Gender (slightly more female churn in banking data)
    'Gender': np.random.choice([0, 1], n_churn, p=[0.55, 0.45]),

    # 10. Geography (Germany has higher churn in this dataset)
    'Geography': np.random.choice([0, 1, 2], n_churn, p=[0.3, 0.5, 0.2]),

    # Target
    'Exited': np.ones(n_churn)
}

# Combine both classes
print(f"\n🔀 Combining and shuffling dataset...")

df_stay = pd.DataFrame(stay_data)
df_churn = pd.DataFrame(churn_data)
df = pd.concat([df_stay, df_churn], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add Customer ID
df.insert(0, 'CustomerId', range(15000000, 15000000 + len(df)))

# Round numerical columns
df['CreditScore'] = df['CreditScore'].round(0).astype(int)
df['Age'] = df['Age'].round(0).astype(int)
df['Tenure'] = df['Tenure'].round(0).astype(int)
df['Balance'] = df['Balance'].round(2)
df['EstimatedSalary'] = df['EstimatedSalary'].round(2)
df['NumOfProducts'] = df['NumOfProducts'].astype(int)
df['HasCrCard'] = df['HasCrCard'].astype(int)
df['IsActiveMember'] = df['IsActiveMember'].astype(int)
df['Gender'] = df['Gender'].astype(int)
df['Geography'] = df['Geography'].astype(int)
df['Exited'] = df['Exited'].astype(int)

# Save to CSV
df.to_csv('bank_churn.csv', index=False)

print(f"\n✅ Dataset created successfully!")
print(f"\n📊 Dataset Information:")
print(f"   - Rows: {df.shape[0]}")
print(f"   - Columns: {df.shape[1]}")
print(f"   - Features: {df.shape[1] - 2} (excluding CustomerId and target)")
print(f"   - Target: Exited (0=Stay, 1=Churn)")

print(f"\n📈 Target Distribution:")
target_counts = df['Exited'].value_counts()
print(f"   - Stay (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"   - Churn (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
print(f"   - Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")

print(f"\n📊 Feature Statistics:")
print(f"   - Credit Score: {df['CreditScore'].min()} to {df['CreditScore'].max()} (mean: {df['CreditScore'].mean():.0f})")
print(f"   - Age: {df['Age'].min()} to {df['Age'].max()} (mean: {df['Age'].mean():.0f})")
print(f"   - Tenure: {df['Tenure'].min()} to {df['Tenure'].max()} years (mean: {df['Tenure'].mean():.1f})")
print(f"   - Balance: ${df['Balance'].min():.0f} to ${df['Balance'].max():.0f} (mean: ${df['Balance'].mean():.0f})")

print(f"\n💾 Saved to: bank_churn.csv")
print(f"\n📝 First 5 rows:")
print(df.head())

print(f"\n📊 Class Balance Analysis:")
print(f"\nChurned Customers (Exited=1):")
print(f"   - Avg Credit Score: {df[df['Exited']==1]['CreditScore'].mean():.0f}")
print(f"   - Avg Age: {df[df['Exited']==1]['Age'].mean():.0f}")
print(f"   - Avg Tenure: {df[df['Exited']==1]['Tenure'].mean():.1f} years")
print(f"   - Avg Balance: ${df[df['Exited']==1]['Balance'].mean():.0f}")
print(f"   - Active Member %: {df[df['Exited']==1]['IsActiveMember'].mean()*100:.1f}%")

print(f"\nRetained Customers (Exited=0):")
print(f"   - Avg Credit Score: {df[df['Exited']==0]['CreditScore'].mean():.0f}")
print(f"   - Avg Age: {df[df['Exited']==0]['Age'].mean():.0f}")
print(f"   - Avg Tenure: {df[df['Exited']==0]['Tenure'].mean():.1f} years")
print(f"   - Avg Balance: ${df[df['Exited']==0]['Balance'].mean():.0f}")
print(f"   - Active Member %: {df[df['Exited']==0]['IsActiveMember'].mean()*100:.1f}%")

print("\n" + "=" * 80)
print(" Dataset Ready for Deep Learning Classification! ".center(80, "="))
print("=" * 80)

print("\n💡 Key Insights:")
print("   - Churned customers have LOWER credit scores")
print("   - Churned customers are either VERY YOUNG or VERY OLD")
print("   - Churned customers have SHORTER tenure")
print("   - Churned customers are LESS ACTIVE")
print("   - This creates a realistic classification challenge!")
