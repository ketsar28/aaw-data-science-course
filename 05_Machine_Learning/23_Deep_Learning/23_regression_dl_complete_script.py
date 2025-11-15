"""
================================================================================
MODULE 23 - DEEP LEARNING REGRESSION COMPLETE
Complete Neural Network Pipeline for Regression Tasks
================================================================================

© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering

Dataset: Energy Efficiency (Building Heating Load Prediction)
- 768 building samples with 8 features
- Target: Heating Load (kWh/m²)
- Continuous regression problem

Complete 10-FASE Pipeline + Deep Learning Specific:
1. Data Loading & Initial Exploration
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing (CRITICAL for DL!)
4. Train-Validation-Test Split
5. Build Neural Network Models (5 architectures)
6. Train with Callbacks
7. Model Evaluation
8. Visualization
9. Model Comparison
10. Save Best Model

================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print(" MODULE 23 - DEEP LEARNING REGRESSION COMPLETE ".center(80, "="))
print("=" * 80)
print(f"\n© Muhammad Ketsar Ali Abi Wahid")
print(f"Dataset: Energy Efficiency (Heating Load Prediction)")
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU Available: {gpus}")
    print(f"   Using GPU for training!")
else:
    print(f"💻 No GPU found. Using CPU (slower but works!)")

print("=" * 80)

# Create directories
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

print("\n✅ Directories created: models/, outputs/")
print("=" * 80)

# ============================================================================
# FASE 1: DATA LOADING & INITIAL EXPLORATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 1: DATA LOADING & INITIAL EXPLORATION ".center(80, "="))
print("=" * 80)

# Load dataset
df = pd.read_csv('datasets/energy_efficiency.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Rows (samples): {df.shape[0]}")
print(f"   - Columns (features + target): {df.shape[1]}")

print(f"\n📝 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n💾 Memory Usage:")
print(f"   Total: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n🔍 Data Types:")
print(df.dtypes)

print(f"\n👀 First 5 rows:")
print(df.head())

print(f"\n📊 Statistical Summary:")
print(df.describe())

print("\n" + "=" * 80)

# ============================================================================
# FASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 2: EXPLORATORY DATA ANALYSIS (EDA) ".center(80, "="))
print("=" * 80)

# 2.1 Missing Values
print("\n📊 2.1 Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(missing[missing > 0])

# 2.2 Duplicate Check
print("\n📊 2.2 Duplicate Rows Check:")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print("   ✅ No duplicate rows found!")
else:
    print(f"   ⚠️ Found {duplicates} duplicate rows")

# 2.3 Target Distribution
print("\n📊 2.3 Target Distribution:")
print(f"   Min: {df['Heating_Load'].min():.2f}")
print(f"   Max: {df['Heating_Load'].max():.2f}")
print(f"   Mean: {df['Heating_Load'].mean():.2f}")
print(f"   Median: {df['Heating_Load'].median():.2f}")
print(f"   Std: {df['Heating_Load'].std():.2f}")

# 2.4 Correlation Analysis
print("\n📊 2.4 Correlation Analysis:")
print("\nGenerating correlation heatmap...")

plt.figure(figsize=(12, 9))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap\n© Muhammad Ketsar Ali Abi Wahid',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/01_correlation_heatmap.png")

# Top correlations with target
target_corr = correlation_matrix['Heating_Load'].abs().sort_values(ascending=False)
print(f"\n🎯 Top 5 Features Correlated with Heating Load:")
for i, (feature, corr) in enumerate(target_corr[1:6].items(), 1):
    print(f"   {i}. {feature}: {corr:.4f}")

# 2.5 Feature Distributions
print("\n📊 2.5 Feature Distributions:")
print("   Generating distribution plots...")

feature_cols = df.columns[:-1]
n_features = len(feature_cols)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    sns.histplot(df[col], kde=True, ax=axes[idx], color='skyblue', alpha=0.7)
    axes[idx].set_title(f'{col} Distribution', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)

# Remove extra subplot
fig.delaxes(axes[-1])

plt.suptitle('Feature Distributions\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/02_feature_distributions.png")

# 2.6 Target Distribution
print("\n📊 2.6 Target Distribution:")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(df['Heating_Load'], kde=True, ax=axes[0], color='coral', bins=30)
axes[0].set_title('Heating Load Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Heating Load (kWh/m²)', fontsize=10)
axes[0].set_ylabel('Frequency', fontsize=10)

# Box plot
sns.boxplot(y=df['Heating_Load'], ax=axes[1], color='lightgreen')
axes[1].set_title('Heating Load Box Plot', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Heating Load (kWh/m²)', fontsize=10)

plt.suptitle('Target Variable Distribution\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('outputs/03_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/03_target_distribution.png")

print("\n" + "=" * 80)

# ============================================================================
# FASE 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 3: DATA PREPROCESSING ".center(80, "="))
print("=" * 80)

print("\n📊 3.1 Separating Features and Target:")

X = df.drop('Heating_Load', axis=1)
y = df['Heating_Load']

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

print("\n⚠️ CRITICAL for Deep Learning:")
print("   Deep Learning models are VERY sensitive to feature scales!")
print("   We MUST normalize/standardize features before training.")
print("   Otherwise, gradients will explode/vanish!")

print("\n📊 3.2 Feature Scaling (StandardScaler):")
print("   StandardScaler: (x - mean) / std → mean=0, std=1")

scaler = StandardScaler()

# Note: We'll fit scaler only on training data later
# For now, just create the scaler object

print("   ✅ StandardScaler initialized")
print("   ℹ️ Will fit on training data only (after split) to prevent data leakage!")

print("\n" + "=" * 80)

# ============================================================================
# FASE 4: TRAIN-VALIDATION-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 4: TRAIN-VALIDATION-TEST SPLIT ".center(80, "="))
print("=" * 80)

print("\n📊 4.1 Three-way Split:")
print("   Why 3 splits for Deep Learning?")
print("   - Training set: Learn patterns (70%)")
print("   - Validation set: Tune hyperparameters & early stopping (15%)")
print("   - Test set: Final unbiased evaluation (15%)")

# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE  # 0.176 of 85% ≈ 15% of total
)

print(f"\n   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\n📊 4.2 Fit StandardScaler on Training Data:")

# Fit only on training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ Scaler fitted on training data only")
print(f"   ✅ Applied to train, val, and test sets")

print(f"\n📊 After scaling:")
print(f"   Training features mean: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"   Training features std: {X_train_scaled.std():.6f} (should be ~1)")

print("\n" + "=" * 80)

# ============================================================================
# FASE 5: BUILD NEURAL NETWORK MODELS
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 5: BUILD NEURAL NETWORK MODELS ".center(80, "="))
print("=" * 80)

print("\n🧠 We'll build 5 different architectures:")
print("   1. Simple (2 hidden layers) - Baseline")
print("   2. Deep (5 hidden layers) - More complex patterns")
print("   3. Wide (more neurons) - More capacity")
print("   4. Dropout (regularization) - Prevent overfitting")
print("   5. Batch Normalization - Stabilize training")

n_features = X_train_scaled.shape[1]

# Store models
models = {}
histories = {}

# ============================================================================
# Model 1: Simple (2 hidden layers)
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 1: SIMPLE (2 Hidden Layers)")
print("-" * 80)

print("\n🏗️ Architecture:")
print("   Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)")
print("   Total params: ~3,000")

model_simple = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,), name='hidden_1'),
    layers.Dense(32, activation='relu', name='hidden_2'),
    layers.Dense(1, name='output')  # Linear activation for regression
], name='Simple_Model')

print("\n📊 Model Summary:")
model_simple.summary()

# Compile
model_simple.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error
)

print("\n✅ Compiled with:")
print("   - Optimizer: Adam (adaptive learning rate)")
print("   - Loss: MSE (Mean Squared Error)")
print("   - Metrics: MAE (Mean Absolute Error)")

models['Simple'] = model_simple

# ============================================================================
# Model 2: Deep (5 hidden layers)
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 2: DEEP (5 Hidden Layers)")
print("-" * 80)

print("\n🏗️ Architecture:")
print("   Input → Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(8) → Dense(1)")

model_deep = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,), name='hidden_1'),
    layers.Dense(64, activation='relu', name='hidden_2'),
    layers.Dense(32, activation='relu', name='hidden_3'),
    layers.Dense(16, activation='relu', name='hidden_4'),
    layers.Dense(8, activation='relu', name='hidden_5'),
    layers.Dense(1, name='output')
], name='Deep_Model')

model_deep.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n📊 Model Summary:")
model_deep.summary()

models['Deep'] = model_deep

# ============================================================================
# Model 3: Wide (more neurons per layer)
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 3: WIDE (More Neurons)")
print("-" * 80)

print("\n🏗️ Architecture:")
print("   Input → Dense(256) → Dense(128) → Dense(1)")
print("   Wider layers = more capacity per layer")

model_wide = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(n_features,), name='hidden_1'),
    layers.Dense(128, activation='relu', name='hidden_2'),
    layers.Dense(1, name='output')
], name='Wide_Model')

model_wide.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n📊 Model Summary:")
model_wide.summary()

models['Wide'] = model_wide

# ============================================================================
# Model 4: With Dropout (Regularization)
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 4: DROPOUT (Regularization)")
print("-" * 80)

print("\n🏗️ Architecture:")
print("   Input → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(1)")
print("   Dropout: Randomly drops 30% of neurons during training")
print("   Prevents overfitting!")

model_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,), name='hidden_1'),
    layers.Dropout(0.3, name='dropout_1'),
    layers.Dense(64, activation='relu', name='hidden_2'),
    layers.Dropout(0.3, name='dropout_2'),
    layers.Dense(1, name='output')
], name='Dropout_Model')

model_dropout.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n📊 Model Summary:")
model_dropout.summary()

models['Dropout'] = model_dropout

# ============================================================================
# Model 5: With Batch Normalization
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 5: BATCH NORMALIZATION")
print("-" * 80)

print("\n🏗️ Architecture:")
print("   Input → Dense(128) → BatchNorm → ReLU → Dense(64) → BatchNorm → ReLU → Dense(1)")
print("   BatchNorm: Normalizes activations → stabilizes training")

model_bn = keras.Sequential([
    layers.Dense(128, input_shape=(n_features,), name='hidden_1'),
    layers.BatchNormalization(name='bn_1'),
    layers.Activation('relu', name='relu_1'),
    layers.Dense(64, name='hidden_2'),
    layers.BatchNormalization(name='bn_2'),
    layers.Activation('relu', name='relu_2'),
    layers.Dense(1, name='output')
], name='BatchNorm_Model')

model_bn.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n📊 Model Summary:")
model_bn.summary()

models['BatchNorm'] = model_bn

print("\n" + "=" * 80)

# ============================================================================
# FASE 6: TRAIN WITH CALLBACKS
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 6: TRAIN WITH CALLBACKS ".center(80, "="))
print("=" * 80)

print("\n🎯 Training Configuration:")
print("   - Epochs: 200 (with early stopping)")
print("   - Batch size: 32")
print("   - Validation data: For monitoring overfitting")

print("\n📊 Callbacks:")
print("   1. EarlyStopping: Stop if val_loss doesn't improve for 20 epochs")
print("   2. ModelCheckpoint: Save best model automatically")
print("   3. ReduceLROnPlateau: Reduce learning rate when stuck")

# Training parameters
EPOCHS = 200
BATCH_SIZE = 32

# Train all models
for name, model in models.items():
    print("\n" + "=" * 80)
    print(f"🚀 Training {name} Model")
    print("=" * 80)

    # Setup callbacks
    model_path = f'models/best_{name.lower()}_model.h5'

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    # Train
    start_time = time.time()

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=0  # Suppress epoch-by-epoch output
    )

    training_time = time.time() - start_time

    # Store history
    histories[name] = history

    # Print results
    final_epoch = len(history.history['loss'])
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1

    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    print(f"   - Total epochs run: {final_epoch}")
    print(f"   - Best epoch: {best_epoch}")
    print(f"   - Best val_loss: {best_val_loss:.4f}")
    print(f"   - Final train_loss: {history.history['loss'][-1]:.4f}")
    print(f"   - Final val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"   ✅ Best model saved: {model_path}")

print("\n" + "=" * 80)
print(" ✅ ALL MODELS TRAINED! ".center(80, "="))
print("=" * 80)

# ============================================================================
# FASE 7: MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 7: MODEL EVALUATION ".center(80, "="))
print("=" * 80)

print("\n📊 Evaluating all models on Test Set...")

results = []

for name, model in models.items():
    # Predict
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    })

    print(f"\n{name} Model:")
    print(f"   - MSE: {mse:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - R²: {r2:.4f}")
    print(f"   - MAPE: {mape:.2f}%")

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False).reset_index(drop=True)

print("\n📊 Complete Model Comparison (Sorted by R²):")
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('outputs/model_results.csv', index=False)
print("\n💾 Results saved to: outputs/model_results.csv")

print("\n" + "=" * 80)

# ============================================================================
# FASE 8: VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 8: VISUALIZATION ".center(80, "="))
print("=" * 80)

# 8.1 Training History
print("\n📊 8.1 Training History Plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for idx, (name, history) in enumerate(histories.items()):
    ax = axes[idx]

    # Plot training & validation loss
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2, color=colors[idx], alpha=0.8)
    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--', color=colors[idx])

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss (MSE)', fontsize=10)
    ax.set_title(f'{name} Model - Training History', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[-1])

plt.suptitle('Training History - All Models\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/04_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/04_training_history.png")

# 8.2 Model Comparison
print("\n📊 8.2 Model Comparison Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² Score
ax1 = axes[0, 0]
bars1 = ax1.barh(results_df['Model'], results_df['R²'], color='skyblue')
ax1.set_xlabel('R² Score', fontsize=11)
ax1.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1)
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}',
             ha='left', va='center', fontsize=10)

# RMSE
ax2 = axes[0, 1]
bars2 = ax2.barh(results_df['Model'], results_df['RMSE'], color='lightcoral')
ax2.set_xlabel('RMSE', fontsize=11)
ax2.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}',
             ha='left', va='center', fontsize=10)

# MAE
ax3 = axes[1, 0]
bars3 = ax3.barh(results_df['Model'], results_df['MAE'], color='lightgreen')
ax3.set_xlabel('MAE', fontsize=11)
ax3.set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}',
             ha='left', va='center', fontsize=10)

# MAPE
ax4 = axes[1, 1]
bars4 = ax4.barh(results_df['Model'], results_df['MAPE (%)'], color='plum')
ax4.set_xlabel('MAPE (%)', fontsize=11)
ax4.set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
for i, bar in enumerate(bars4):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2, f' {width:.2f}%',
             ha='left', va='center', fontsize=10)

plt.suptitle('Model Performance Comparison\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/05_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/05_model_comparison.png")

# 8.3 Actual vs Predicted (Best Model)
print("\n📊 8.3 Actual vs Predicted for Best Model...")

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

y_pred_best = best_model.predict(X_test_scaled, verbose=0).flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax1 = axes[0]
ax1.scatter(y_test, y_pred_best, alpha=0.6, s=50, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Heating Load (kWh/m²)', fontsize=11)
ax1.set_ylabel('Predicted Heating Load (kWh/m²)', fontsize=11)
ax1.set_title(f'Actual vs Predicted - {best_model_name} Model', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Residual plot
ax2 = axes[1]
residuals = y_test - y_pred_best
ax2.scatter(y_pred_best, residuals, alpha=0.6, s=50, color='coral', edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Heating Load (kWh/m²)', fontsize=11)
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
ax2.set_title(f'Residual Plot - {best_model_name} Model', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Best Model Performance Analysis\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('outputs/06_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/06_actual_vs_predicted.png")

print("\n" + "=" * 80)

# ============================================================================
# FASE 9: FINAL MODEL SELECTION & REPORT
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 9: FINAL MODEL SELECTION & REPORT ".center(80, "="))
print("=" * 80)

print("\n🏆 Best Model Selection:")

best_model_row = results_df.iloc[0]
print(f"\n✅ SELECTED BEST MODEL: {best_model_row['Model']} Model")
print(f"\n📊 Performance Metrics:")
print(f"   - MSE: {best_model_row['MSE']:.4f}")
print(f"   - RMSE: {best_model_row['RMSE']:.4f}")
print(f"   - MAE: {best_model_row['MAE']:.4f}")
print(f"   - R²: {best_model_row['R²']:.4f}")
print(f"   - MAPE: {best_model_row['MAPE (%)']:.2f}%")

print(f"\n💡 Interpretation:")
print(f"   - R² = {best_model_row['R²']:.4f} means model explains {best_model_row['R²']*100:.2f}% of variance")
print(f"   - On average, predictions are off by {best_model_row['MAE']:.2f} kWh/m²")
print(f"   - Average percentage error: {best_model_row['MAPE (%)']:.2f}%")

# ============================================================================
# FASE 10: SAVE BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 10: SAVE BEST MODEL ".center(80, "="))
print("=" * 80)

# Save best model
best_model_path = f'models/best_model_final.h5'
best_model.save(best_model_path)

print(f"\n✅ Best model saved: {best_model_path}")

# Save scaler
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler saved: models/scaler.pkl")

print("\n📊 How to Use Saved Model:")
print("""
# Load model and scaler
from tensorflow import keras
import pickle
import numpy as np
import pandas as pd

# Load
model = keras.models.load_model('models/best_model_final.h5')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data (must have same 8 features)
new_data = pd.DataFrame({
    'Relative_Compactness': [0.75],
    'Surface_Area': [600],
    'Wall_Area': [300],
    'Roof_Area': [150],
    'Overall_Height': [3.5],
    'Orientation': [3],
    'Glazing_Area': [0.25],
    'Glazing_Area_Distribution': [2]
})

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)[0][0]

print(f"Predicted Heating Load: {prediction:.2f} kWh/m²")
""")

print("\n" + "=" * 80)
print(" FINAL SUMMARY REPORT ".center(80, "="))
print("=" * 80)

print(f"""
🎯 PROJECT SUMMARY
================================================================================

Dataset: Energy Efficiency (Building Heating Load Prediction)
  - Total samples: {len(df)}
  - Features: {X.shape[1]}
  - Target: Heating Load (kWh/m²)
  - Range: {df['Heating_Load'].min():.2f} to {df['Heating_Load'].max():.2f} kWh/m²

📊 BEST MODEL: {best_model_row['Model']} Neural Network

Performance Metrics:
  - MSE: {best_model_row['MSE']:.4f}
  - RMSE: {best_model_row['RMSE']:.4f} kWh/m²
  - MAE: {best_model_row['MAE']:.4f} kWh/m² (average error)
  - R²: {best_model_row['R²']:.4f} ({best_model_row['R²']*100:.2f}% variance explained)
  - MAPE: {best_model_row['MAPE (%)']:.2f}% (average % error)

🧠 MODELS TRAINED:
================================================================================
1. Simple (2 layers) - Baseline neural network
2. Deep (5 layers) - More complex patterns
3. Wide (large layers) - More capacity
4. Dropout - Regularization to prevent overfitting
5. Batch Normalization - Stabilized training

🎓 KEY LEARNINGS:
================================================================================

1. 🔢 Feature Scaling is CRITICAL:
   - Neural networks need normalized features (mean=0, std=1)
   - Without scaling, training fails or is very slow
   - Always use StandardScaler or MinMaxScaler

2. 📊 Train-Validation-Test Split:
   - 3-way split is essential for DL (not just 2-way!)
   - Validation set used for early stopping and hyperparameter tuning
   - Test set for final unbiased evaluation

3. 🎯 Callbacks Save Time:
   - EarlyStopping prevents wasting time on plateaued training
   - ModelCheckpoint automatically saves best model
   - ReduceLROnPlateau helps escape local minima

4. 🏗️ Architecture Matters:
   - Simple models often work best for tabular data
   - Too deep = harder to train (vanishing gradients)
   - Regularization (Dropout, BatchNorm) helps prevent overfitting

5. 📈 Deep Learning vs Traditional ML:
   - For tabular data, XGBoost/Random Forest often better
   - DL shines with unstructured data (images, text, audio)
   - DL requires more data and computational power

📂 GENERATED FILES:
================================================================================

Models:
  ✅ models/best_model_final.h5 (Best model)
  ✅ models/scaler.pkl (Preprocessing scaler)
  ✅ models/best_simple_model.h5
  ✅ models/best_deep_model.h5
  ✅ models/best_wide_model.h5
  ✅ models/best_dropout_model.h5
  ✅ models/best_batchnorm_model.h5

Results:
  ✅ outputs/model_results.csv

Visualizations (6 plots):
  ✅ outputs/01_correlation_heatmap.png
  ✅ outputs/02_feature_distributions.png
  ✅ outputs/03_target_distribution.png
  ✅ outputs/04_training_history.png
  ✅ outputs/05_model_comparison.png
  ✅ outputs/06_actual_vs_predicted.png

🚀 NEXT STEPS:
================================================================================

1. ✅ Try classification with Deep Learning (Module 23b)
2. ✅ Experiment with different architectures
3. ✅ Learn CNNs for image data (Module 25)
4. ✅ Learn RNNs/LSTMs for sequence data (Module 26)
5. ✅ Deploy model as API (Module 30)

💡 WHEN TO USE DEEP LEARNING:
================================================================================

✅ USE Deep Learning When:
   - Large datasets (> 100k samples)
   - Unstructured data (images, text, audio)
   - Complex patterns difficult for traditional ML
   - No domain expertise for feature engineering

❌ DON'T Use Deep Learning When:
   - Small datasets (< 10k samples) → Use Random Forest, XGBoost
   - Tabular data with clear features → Traditional ML often better!
   - Need interpretability → Use linear models, trees
   - Limited computational resources → Too slow to train

================================================================================
© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering
================================================================================

✅ MODULE 23 - DEEP LEARNING REGRESSION COMPLETE!

Thank you for using this comprehensive Deep Learning pipeline!
For questions or feedback, please refer to the documentation.

Happy Learning! 🚀🧠
================================================================================
""")
