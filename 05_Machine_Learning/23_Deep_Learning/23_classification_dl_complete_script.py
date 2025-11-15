"""
================================================================================
MODULE 23 - DEEP LEARNING CLASSIFICATION COMPLETE
Complete Neural Network Pipeline for Binary Classification
================================================================================

© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering

Dataset: Bank Customer Churn (Binary Classification)
- 10,000 customers with 10 features
- Target: Churn (0=Stay, 1=Leave)
- Imbalanced: 80% stay, 20% churn

Complete 10-FASE Pipeline + DL Specific for Classification:
1. Data Loading & Initial Exploration
2. Exploratory Data Analysis
3. Data Preprocessing (Scaling + Handle Imbalance)
4. Train-Validation-Test Split (Stratified!)
5. Build Neural Network Models (4 architectures)
6. Train with Callbacks
7. Model Evaluation (Classification Metrics)
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.utils import class_weight

# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print(" MODULE 23 - DEEP LEARNING CLASSIFICATION COMPLETE ".center(80, "="))
print("=" * 80)
print(f"\n© Muhammad Ketsar Ali Abi Wahid")
print(f"Dataset: Bank Customer Churn (Binary Classification)")
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU Available: {gpus}")
else:
    print(f"💻 No GPU found. Using CPU")

print("=" * 80)

# Create directories
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

print("\n✅ Directories ready")
print("=" * 80)

# ============================================================================
# FASE 1: DATA LOADING & EXPLORATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 1: DATA LOADING & EXPLORATION ".center(80, "="))
print("=" * 80)

df = pd.read_csv('datasets/bank_churn.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Samples: {df.shape[0]}")
print(f"   - Features: {df.shape[1] - 2} (excluding CustomerId and target)")

print(f"\n👀 First 5 rows:")
print(df.head())

print(f"\n🎯 Target Distribution (CRITICAL for Classification!):")
target_counts = df['Exited'].value_counts()
target_pct = df['Exited'].value_counts(normalize=True) * 100

print(f"\n   Class 0 (Stay): {target_counts[0]} ({target_pct[0]:.1f}%)")
print(f"   Class 1 (Churn): {target_counts[1]} ({target_pct[1]:.1f}%)")
print(f"   Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")
print(f"\n   ⚖️ IMBALANCED DATA - Need special handling!")

print("\n" + "=" * 80)

# ============================================================================
# FASE 2: EDA
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 2: EXPLORATORY DATA ANALYSIS ".center(80, "="))
print("=" * 80)

# Missing values
print("\n📊 Missing Values:")
if df.isnull().sum().sum() == 0:
    print("   ✅ No missing values!")

# Target Distribution Visualization
print("\n📊 Creating Target Distribution Plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='Exited', ax=axes[0], palette='Set2')
axes[0].set_title('Churn Distribution', fontweight='bold')
axes[0].set_xlabel('Exited (0=Stay, 1=Churn)')
for container in axes[0].containers:
    axes[0].bar_label(container)

colors = ['#90EE90', '#FFB6C1']
axes[1].pie(target_counts, labels=['Stay (0)', 'Churn (1)'], autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Churn Percentage', fontweight='bold')

plt.suptitle('Target Distribution\n© Muhammad Ketsar Ali Abi Wahid', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/01_target_distribution.png")

print("\n" + "=" * 80)

# ============================================================================
# FASE 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 3: DATA PREPROCESSING ".center(80, "="))
print("=" * 80)

print("\n📊 Preparing Features:")

# Drop CustomerId (not a feature)
X = df.drop(['CustomerId', 'Exited'], axis=1)
y = df['Exited']

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

print("\n⚠️ CRITICAL:")
print("   1. Feature scaling (StandardScaler) - MUST for Deep Learning!")
print("   2. Class weights - Handle imbalanced data!")

scaler = StandardScaler()

print("\n✅ StandardScaler initialized")
print("   Will fit on training data only!")

print("\n" + "=" * 80)

# ============================================================================
# FASE 4: TRAIN-VALIDATION-TEST SPLIT (STRATIFIED!)
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 4: STRATIFIED TRAIN-VALIDATION-TEST SPLIT ".center(80, "="))
print("=" * 80)

print("\n📊 Using STRATIFIED split to maintain class distribution!")

# Three-way split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\n   Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Fit scaler on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Features scaled!")
print(f"   Mean: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"   Std: {X_train_scaled.std():.6f} (should be ~1)")

# Calculate class weights for imbalance handling
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\n⚖️ Class Weights for Imbalance:")
print(f"   Class 0 (Stay): {class_weight_dict[0]:.4f}")
print(f"   Class 1 (Churn): {class_weight_dict[1]:.4f}")
print(f"   Higher weight for minority class!")

print("\n" + "=" * 80)

# ============================================================================
# FASE 5: BUILD NEURAL NETWORK MODELS
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 5: BUILD NEURAL NETWORK MODELS ".center(80, "="))
print("=" * 80)

print("\n🧠 Building 4 Classification Models:")
print("   1. Simple (baseline)")
print("   2. Dropout (regularization)")
print("   3. Batch Normalization")
print("   4. Combined (Dropout + BatchNorm)")

n_features = X_train_scaled.shape[1]

models = {}
histories = {}

# ============================================================================
# Model 1: Simple
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 1: SIMPLE")
print("-" * 80)

model_simple = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification!
], name='Simple')

model_simple.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary crossentropy for binary classification!
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\n✅ Compiled with:")
print("   - Loss: Binary Crossentropy")
print("   - Metrics: Accuracy, Precision, Recall")
print("   - Output: Sigmoid (probability 0-1)")

models['Simple'] = model_simple

# ============================================================================
# Model 2: Dropout
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 2: DROPOUT (30% drop rate)")
print("-" * 80)

model_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
], name='Dropout')

model_dropout.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

models['Dropout'] = model_dropout

# ============================================================================
# Model 3: Batch Normalization
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 3: BATCH NORMALIZATION")
print("-" * 80)

model_bn = keras.Sequential([
    layers.Dense(128, input_shape=(n_features,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1, activation='sigmoid')
], name='BatchNorm')

model_bn.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

models['BatchNorm'] = model_bn

# ============================================================================
# Model 4: Combined (Dropout + BatchNorm)
# ============================================================================

print("\n" + "-" * 80)
print("🧠 MODEL 4: COMBINED (Dropout + BatchNorm)")
print("-" * 80)

model_combined = keras.Sequential([
    layers.Dense(128, input_shape=(n_features,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='Combined')

model_combined.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

models['Combined'] = model_combined

print("\n✅ All models built!")
print("=" * 80)

# ============================================================================
# FASE 6: TRAIN WITH CALLBACKS & CLASS WEIGHTS
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 6: TRAIN WITH CALLBACKS & CLASS WEIGHTS ".center(80, "="))
print("=" * 80)

print("\n⚖️ Using class_weight to handle imbalance during training!")
print(f"   This penalizes misclassifying minority class (Churn) more heavily")

EPOCHS = 150
BATCH_SIZE = 32

for name, model in models.items():
    print("\n" + "=" * 80)
    print(f"🚀 Training {name} Model")
    print("=" * 80)

    model_path = f'models/best_{name.lower()}_classification.h5'

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
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
        patience=8,
        min_lr=1e-7,
        verbose=1
    )

    start_time = time.time()

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,  # Handle imbalance!
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=0
    )

    training_time = time.time() - start_time
    histories[name] = history

    final_epoch = len(history.history['loss'])
    best_val_loss = min(history.history['val_loss'])

    print(f"\n✅ Completed in {training_time:.2f}s")
    print(f"   Epochs: {final_epoch}")
    print(f"   Best val_loss: {best_val_loss:.4f}")
    print(f"   ✅ Saved: {model_path}")

print("\n" + "=" * 80)
print(" ✅ ALL MODELS TRAINED! ".center(80, "="))
print("=" * 80)

# ============================================================================
# FASE 7: MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 7: MODEL EVALUATION ".center(80, "="))
print("=" * 80)

print("\n📊 Evaluating on Test Set...")

results = []

for name, model in models.items():
    # Predict probabilities
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    # Convert to binary (threshold 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

    print(f"\n{name} Model:")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

print("\n📊 Model Comparison (Sorted by F1-Score):")
print("\n" + results_df.to_string(index=False))

results_df.to_csv('outputs/classification_results.csv', index=False)
print("\n💾 Saved: outputs/classification_results.csv")

print("\n" + "=" * 80)

# ============================================================================
# FASE 8: VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 8: VISUALIZATION ".center(80, "="))
print("=" * 80)

# Training History
print("\n📊 Training History Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (name, history) in enumerate(histories.items()):
    ax = axes[idx]
    ax.plot(history.history['loss'], label='Train Loss', linewidth=2, color=colors[idx], alpha=0.8)
    ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--', color=colors[idx])
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title(f'{name} - Training History', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Training History\n© Muhammad Ketsar Ali Abi Wahid', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/02_training_history.png")

# Model Comparison
print("\n📊 Model Comparison Plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# F1-Score
ax1 = axes[0, 0]
bars1 = ax1.barh(results_df['Model'], results_df['F1-Score'], color='skyblue')
ax1.set_xlabel('F1-Score')
ax1.set_title('F1-Score Comparison', fontweight='bold')
for bar in bars1:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}', va='center')

# Recall
ax2 = axes[0, 1]
bars2 = ax2.barh(results_df['Model'], results_df['Recall'], color='lightcoral')
ax2.set_xlabel('Recall')
ax2.set_title('Recall Comparison', fontweight='bold')
for bar in bars2:
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}', va='center')

# Precision
ax3 = axes[1, 0]
bars3 = ax3.barh(results_df['Model'], results_df['Precision'], color='lightgreen')
ax3.set_xlabel('Precision')
ax3.set_title('Precision Comparison', fontweight='bold')
for bar in bars3:
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}', va='center')

# ROC-AUC
ax4 = axes[1, 1]
bars4 = ax4.barh(results_df['Model'], results_df['ROC-AUC'], color='plum')
ax4.set_xlabel('ROC-AUC')
ax4.set_title('ROC-AUC Comparison', fontweight='bold')
for bar in bars4:
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}', va='center')

plt.suptitle('Model Performance Comparison\n© Muhammad Ketsar Ali Abi Wahid', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/03_model_comparison.png")

# Confusion Matrix (Best Model)
print("\n📊 Confusion Matrix for Best Model...")

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

y_pred_best_proba = best_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_best = (y_pred_best_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name}\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=13, fontweight='bold', pad=15)

tn, fp, fn, tp = cm.ravel()
ax.text(0.5, 1.8, f'TN={tn}', ha='center', fontsize=10)
ax.text(1.5, 1.8, f'FP={fp}', ha='center', fontsize=10)
ax.text(0.5, 2.8, f'FN={fn}', ha='center', fontsize=10)
ax.text(1.5, 2.8, f'TP={tp}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/04_confusion_matrix.png")

# ROC Curve
print("\n📊 ROC Curve...")

fig, ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5000)')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves\n© Muhammad Ketsar Ali Abi Wahid', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/05_roc_curves.png")

print("\n" + "=" * 80)

# ============================================================================
# FASE 9: FINAL MODEL & REPORT
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 9: FINAL MODEL SELECTION & REPORT ".center(80, "="))
print("=" * 80)

best_row = results_df.iloc[0]

print(f"\n🏆 BEST MODEL: {best_row['Model']}")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {best_row['Accuracy']:.4f}")
print(f"   - Precision: {best_row['Precision']:.4f}")
print(f"   - Recall: {best_row['Recall']:.4f}")
print(f"   - F1-Score: {best_row['F1-Score']:.4f}")
print(f"   - ROC-AUC: {best_row['ROC-AUC']:.4f}")

print(f"\n💡 Interpretation:")
print(f"   - F1-Score = {best_row['F1-Score']:.4f} (balance of Precision & Recall)")
print(f"   - Recall = {best_row['Recall']:.4f} (catches {best_row['Recall']*100:.1f}% of churners)")
print(f"   - Precision = {best_row['Precision']:.4f} ({best_row['Precision']*100:.1f}% of predicted churners are correct)")

# ============================================================================
# FASE 10: SAVE BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 10: SAVE BEST MODEL ".center(80, "="))
print("=" * 80)

best_model_path = 'models/best_classification_final.h5'
best_model.save(best_model_path)

print(f"\n✅ Model saved: {best_model_path}")

import pickle
with open('models/scaler_classification.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler saved: models/scaler_classification.pkl")

print("\n" + "=" * 80)
print(" FINAL SUMMARY ".center(80, "="))
print("=" * 80)

print(f"""
🎯 PROJECT SUMMARY
================================================================================

Dataset: Bank Customer Churn
  - Samples: 10,000 customers
  - Features: 10
  - Target: Binary (0=Stay, 1=Churn)
  - Imbalance: 80% vs 20%

📊 BEST MODEL: {best_row['Model']} Neural Network

Performance:
  - F1-Score: {best_row['F1-Score']:.4f} (PRIMARY METRIC)
  - Recall: {best_row['Recall']:.4f} (catch churners!)
  - Precision: {best_row['Precision']:.4f} (minimize false alarms)
  - ROC-AUC: {best_row['ROC-AUC']:.4f}

🧠 MODELS TRAINED:
================================================================================
1. Simple - Baseline
2. Dropout - Regularization
3. Batch Normalization - Stable training
4. Combined - Dropout + BatchNorm

🎓 KEY LEARNINGS:
================================================================================

1. ⚖️ Handle Imbalance:
   - Used class_weight during training
   - Penalizes minority class misclassification more
   - Stratified splitting maintains distribution

2. 📊 Classification Metrics:
   - F1-Score for imbalanced data (not accuracy!)
   - Recall: don't miss churners
   - Precision: minimize false alarms
   - ROC-AUC: overall discriminative ability

3. 🏗️ Binary Classification Architecture:
   - Sigmoid activation for output (probability 0-1)
   - Binary crossentropy loss
   - Threshold 0.5 for binary prediction

4. 🎯 Deep Learning vs Traditional:
   - For tabular data: XGBoost/RF often better
   - DL shines with images, text, audio
   - DL needs more data and compute

📂 FILES GENERATED:
================================================================================

Models:
  ✅ models/best_classification_final.h5
  ✅ models/scaler_classification.pkl

Visualizations:
  ✅ outputs/01_target_distribution.png
  ✅ outputs/02_training_history.png
  ✅ outputs/03_model_comparison.png
  ✅ outputs/04_confusion_matrix.png
  ✅ outputs/05_roc_curves.png

================================================================================
© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering
================================================================================

✅ MODULE 23 - DEEP LEARNING CLASSIFICATION COMPLETE!

Happy Learning! 🚀🧠
================================================================================
""")
