"""
================================================================================
MODULE 17 - CLASSIFICATION COMPLETE
Complete Binary Classification Pipeline with Imbalanced Data Handling
================================================================================

© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering

Dataset: Sand Production Prediction (Oil & Gas Industry)
- 5,000 wells with 15 features
- Binary target: Sand Production (0 = No sand, 1 = Sand)
- Imbalanced: 85% No sand, 15% Sand (ratio 5.67:1)

Complete 10-FASE Pipeline:
1. Data Loading & Initial Exploration
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Train-Test Split & Baseline
5. Model Building (15 Algorithms)
5.5. Imbalanced Data Handling (SMOTE, ADASYN, Class Weights)
6. Cross-Validation
7. Hyperparameter Tuning
8. Model Evaluation & Comparison
9. Model Interpretation
10. Final Model Selection & Report

================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import pickle
from pathlib import Path

# Scikit-learn - Core
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)

# Scikit-learn - Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

# Imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier

# Gradient Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Hyperparameter Optimization
import optuna

# Model Interpretation
import shap

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print(" MODULE 17 - CLASSIFICATION COMPLETE ".center(80, "="))
print("=" * 80)
print(f"\n© Muhammad Ketsar Ali Abi Wahid")
print(f"Dataset: Sand Production Prediction (Imbalanced Binary Classification)")
print(f"\nLibraries loaded successfully!")
print(f"Random State: {RANDOM_STATE}")
print("=" * 80)

# Create necessary directories
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
df = pd.read_csv('datasets/sand_production_data.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Rows (samples): {df.shape[0]}")
print(f"   - Columns (features + target): {df.shape[1]}")

print(f"\n📝 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n💾 Memory Usage:")
print(f"   Total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n🔍 Data Types:")
print(df.dtypes)

print(f"\n👀 First 5 rows:")
print(df.head())

print(f"\n📊 Last 5 rows:")
print(df.tail())

print(f"\n🎯 Target Variable Distribution (CRITICAL for Classification!):")
target_counts = df['Sand_Production'].value_counts()
target_pct = df['Sand_Production'].value_counts(normalize=True) * 100

print(f"\nClass Distribution:")
print(f"   Class 0 (No Sand): {target_counts[0]} samples ({target_pct[0]:.2f}%)")
print(f"   Class 1 (Sand):    {target_counts[1]} samples ({target_pct[1]:.2f}%)")
print(f"\n⚖️ Imbalance Ratio: {target_counts[0] / target_counts[1]:.2f}:1")
print(f"\n⚠️ IMBALANCED DATA DETECTED!")
print(f"   This requires special handling techniques (SMOTE, class weights, etc.)")

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

# 2.3 Statistical Summary
print("\n📊 2.3 Statistical Summary:")
print(df.describe())

# 2.4 Correlation Analysis
print("\n📊 2.4 Correlation Analysis:")
print("\nGenerating correlation heatmap...")

plt.figure(figsize=(14, 10))
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
target_corr = correlation_matrix['Sand_Production'].abs().sort_values(ascending=False)
print(f"\n🎯 Top 5 Features Correlated with Sand Production:")
for i, (feature, corr) in enumerate(target_corr[1:6].items(), 1):
    print(f"   {i}. {feature}: {corr:.4f}")

# 2.5 Target Distribution Visualization
print("\n📊 2.5 Target Distribution Visualization:")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=df, x='Sand_Production', ax=axes[0], palette='Set2')
axes[0].set_title('Sand Production Class Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Class (0=No Sand, 1=Sand)', fontsize=10)
axes[0].set_ylabel('Count', fontsize=10)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%d')

# Pie chart
colors = ['#90EE90', '#FFB6C1']
axes[1].pie(target_counts, labels=['No Sand (0)', 'Sand (1)'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Class Distribution Percentage', fontsize=12, fontweight='bold')

plt.suptitle('Target Variable Distribution\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/02_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/02_target_distribution.png")

# 2.6 Feature Distributions by Class
print("\n📊 2.6 Feature Distributions by Class:")
print("   Generating distribution plots for all features...")

feature_cols = df.columns[:-1]  # All except target
n_features = len(feature_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    sns.histplot(data=df, x=col, hue='Sand_Production', kde=True, ax=axes[idx],
                 palette='Set2', alpha=0.6, bins=30)
    axes[idx].set_title(f'{col} Distribution by Class', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)
    axes[idx].legend(labels=['No Sand', 'Sand'], fontsize=8)

# Remove empty subplots
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Feature Distributions by Sand Production Class\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('outputs/03_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/03_feature_distributions.png")

# 2.7 Outlier Detection
print("\n📊 2.7 Outlier Detection (IQR Method):")

outlier_counts = {}
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_counts[col] = len(outliers)

print("\n   Outliers per feature:")
for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    if count > 0:
        print(f"   - {col}: {count} outliers ({count/len(df)*100:.2f}%)")

print("\n   ℹ️ Note: For classification, we typically keep outliers as they may")
print("   represent real minority class patterns (especially for imbalanced data).")

print("\n" + "=" * 80)

# ============================================================================
# FASE 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 3: DATA PREPROCESSING ".center(80, "="))
print("=" * 80)

print("\n📊 3.1 Separating Features and Target:")

X = df.drop('Sand_Production', axis=1)
y = df['Sand_Production']

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

print("\n📊 3.2 Feature Scaling (StandardScaler):")
print("   Why? Many algorithms (SVM, KNN, Logistic Regression) are sensitive to scale.")
print("   StandardScaler: (x - mean) / std → mean=0, std=1")

scaler = StandardScaler()

# Note: We'll fit scaler only on training data later to prevent data leakage
# For now, just create the scaler object

print("   ✅ StandardScaler initialized")
print("   ℹ️ Will fit on training data only (after split) to prevent data leakage!")

print("\n" + "=" * 80)

# ============================================================================
# FASE 4: TRAIN-TEST SPLIT & BASELINE
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 4: TRAIN-TEST SPLIT & BASELINE ".center(80, "="))
print("=" * 80)

print("\n📊 4.1 Stratified Train-Test Split:")
print("   Why STRATIFIED? To maintain class distribution in both train and test!")
print("   Split ratio: 80% train, 20% test")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\n   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

print(f"\n   Training class distribution:")
train_dist = y_train.value_counts()
print(f"   - Class 0: {train_dist[0]} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"   - Class 1: {train_dist[1]} ({train_dist[1]/len(y_train)*100:.2f}%)")

print(f"\n   Test class distribution:")
test_dist = y_test.value_counts()
print(f"   - Class 0: {test_dist[0]} ({test_dist[0]/len(y_test)*100:.2f}%)")
print(f"   - Class 1: {test_dist[1]} ({test_dist[1]/len(y_test)*100:.2f}%)")

print(f"\n   ✅ Class distribution preserved in both sets!")

# Now fit the scaler on training data only
print("\n📊 4.2 Fitting StandardScaler on Training Data:")

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ Scaler fitted on training data only")
print(f"   ✅ Applied to both train and test sets")

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\n📊 4.3 Baseline Model (Dummy Classifier):")
print("   Purpose: Establish performance floor - any model must beat this!")
print("   Strategy: 'most_frequent' → Always predict majority class")

baseline = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
baseline.fit(X_train_scaled, y_train)
y_pred_baseline = baseline.predict(X_test_scaled)

baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)

print(f"\n   Baseline Performance:")
print(f"   - Accuracy: {baseline_acc:.4f}")
print(f"   - F1-Score: {baseline_f1:.4f}")
print(f"   - Recall: {baseline_recall:.4f}")

print(f"\n   ⚠️ As expected, baseline has ~85% accuracy but ZERO F1-Score!")
print(f"   This is why accuracy is USELESS for imbalanced data!")

print("\n" + "=" * 80)

# ============================================================================
# FASE 5: MODEL BUILDING (15 ALGORITHMS)
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 5: MODEL BUILDING (15 ALGORITHMS) ".center(80, "="))
print("=" * 80)

print("\n🎯 We'll train 15 classification algorithms:")
print("   1. Logistic Regression")
print("   2. K-Nearest Neighbors (KNN)")
print("   3. Naive Bayes")
print("   4. Decision Tree")
print("   5. Random Forest")
print("   6. Gradient Boosting")
print("   7. XGBoost")
print("   8. LightGBM")
print("   9. CatBoost")
print("   10. Support Vector Machine (SVM)")
print("   11. Linear Discriminant Analysis (LDA)")
print("   12. Quadratic Discriminant Analysis (QDA)")
print("   13. AdaBoost")
print("   14. Extra Trees")
print("   15. Balanced Random Forest (handles imbalance!)")

print("\n⚖️ For algorithms that support it, we'll use class_weight='balanced'")
print("   to handle imbalanced data!")

# Store results
results = []

# ============================================================================
# 5.1 Logistic Regression
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.1 LOGISTIC REGRESSION")
print("-" * 80)

print("\n🔍 What is Logistic Regression?")
print("   Linear model that uses sigmoid function to predict probabilities.")
print("   Output: P(y=1|x) = 1 / (1 + e^(-wx + b))")

print("\n✅ When to use:")
print("   - Need probability estimates")
print("   - Linear decision boundary is sufficient")
print("   - Want interpretable model (coefficients)")
print("   - Need fast training and prediction")

print("\n❌ When NOT to use:")
print("   - Non-linear relationships in data")
print("   - Complex feature interactions")
print("   - Very high-dimensional sparse data")

print("\n⚖️ Handling Imbalance: class_weight='balanced'")

print("\n🚀 Training Logistic Regression...")
start_time = time.time()

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

train_time_lr = time.time() - start_time

acc_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_lr:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_lr:.4f}")
print(f"   - Precision: {precision_lr:.4f}")
print(f"   - Recall: {recall_lr:.4f}")
print(f"   - F1-Score: {f1_lr:.4f}")
print(f"   - ROC-AUC: {roc_auc_lr:.4f}")

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': acc_lr,
    'Precision': precision_lr,
    'Recall': recall_lr,
    'F1-Score': f1_lr,
    'ROC-AUC': roc_auc_lr,
    'Training Time (s)': train_time_lr
})

# ============================================================================
# 5.2 K-Nearest Neighbors (KNN)
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.2 K-NEAREST NEIGHBORS (KNN)")
print("-" * 80)

print("\n🔍 What is KNN?")
print("   Non-parametric algorithm that classifies based on k nearest neighbors.")
print("   Prediction = majority vote of k nearest neighbors")

print("\n✅ When to use:")
print("   - Small to medium datasets")
print("   - Non-linear decision boundaries")
print("   - No training required (lazy learning)")

print("\n❌ When NOT to use:")
print("   - Large datasets (slow prediction)")
print("   - High-dimensional data (curse of dimensionality)")
print("   - Need interpretable model")

print("\n⚖️ Note: KNN doesn't have class_weight parameter, but we can use")
print("   different distance metrics or post-processing")

print("\n🚀 Training K-Nearest Neighbors...")
start_time = time.time()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

train_time_knn = time.time() - start_time

acc_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_knn:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_knn:.4f}")
print(f"   - Precision: {precision_knn:.4f}")
print(f"   - Recall: {recall_knn:.4f}")
print(f"   - F1-Score: {f1_knn:.4f}")
print(f"   - ROC-AUC: {roc_auc_knn:.4f}")

results.append({
    'Model': 'K-Nearest Neighbors',
    'Accuracy': acc_knn,
    'Precision': precision_knn,
    'Recall': recall_knn,
    'F1-Score': f1_knn,
    'ROC-AUC': roc_auc_knn,
    'Training Time (s)': train_time_knn
})

# ============================================================================
# 5.3 Naive Bayes
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.3 NAIVE BAYES (Gaussian)")
print("-" * 80)

print("\n🔍 What is Naive Bayes?")
print("   Probabilistic classifier based on Bayes' Theorem.")
print("   Assumes features are independent (naive assumption).")
print("   P(y|x) = P(x|y) * P(y) / P(x)")

print("\n✅ When to use:")
print("   - Text classification (spam detection)")
print("   - Real-time prediction (very fast)")
print("   - Small training datasets")
print("   - Features are roughly independent")

print("\n❌ When NOT to use:")
print("   - Features are highly correlated")
print("   - Need high precision (tends to overestimate)")
print("   - Complex feature interactions")

print("\n🚀 Training Naive Bayes...")
start_time = time.time()

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

train_time_nb = time.time() - start_time

acc_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test, nb.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_nb:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_nb:.4f}")
print(f"   - Precision: {precision_nb:.4f}")
print(f"   - Recall: {recall_nb:.4f}")
print(f"   - F1-Score: {f1_nb:.4f}")
print(f"   - ROC-AUC: {roc_auc_nb:.4f}")

results.append({
    'Model': 'Naive Bayes',
    'Accuracy': acc_nb,
    'Precision': precision_nb,
    'Recall': recall_nb,
    'F1-Score': f1_nb,
    'ROC-AUC': roc_auc_nb,
    'Training Time (s)': train_time_nb
})

# ============================================================================
# 5.4 Decision Tree
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.4 DECISION TREE")
print("-" * 80)

print("\n🔍 What is Decision Tree?")
print("   Tree-based model that splits data based on feature values.")
print("   Easy to interpret and visualize.")

print("\n✅ When to use:")
print("   - Need interpretable model")
print("   - Mixed feature types (numerical + categorical)")
print("   - Non-linear relationships")
print("   - Don't need scaling")

print("\n❌ When NOT to use:")
print("   - Prone to overfitting")
print("   - Unstable (small data changes = big tree changes)")
print("   - Not best for high-dimensional data")

print("\n⚖️ Handling Imbalance: class_weight='balanced'")

print("\n🚀 Training Decision Tree...")
start_time = time.time()

dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE, class_weight='balanced')
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

train_time_dt = time.time() - start_time

acc_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_dt:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_dt:.4f}")
print(f"   - Precision: {precision_dt:.4f}")
print(f"   - Recall: {recall_dt:.4f}")
print(f"   - F1-Score: {f1_dt:.4f}")
print(f"   - ROC-AUC: {roc_auc_dt:.4f}")

results.append({
    'Model': 'Decision Tree',
    'Accuracy': acc_dt,
    'Precision': precision_dt,
    'Recall': recall_dt,
    'F1-Score': f1_dt,
    'ROC-AUC': roc_auc_dt,
    'Training Time (s)': train_time_dt
})

# ============================================================================
# 5.5 Random Forest
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.5 RANDOM FOREST")
print("-" * 80)

print("\n🔍 What is Random Forest?")
print("   Ensemble of Decision Trees with bagging.")
print("   Each tree trained on random subset of data and features.")
print("   Final prediction = majority vote of all trees")

print("\n✅ When to use:")
print("   - Need robust, accurate model")
print("   - Handle missing values and outliers well")
print("   - Feature importance needed")
print("   - Reduce overfitting compared to single tree")

print("\n❌ When NOT to use:")
print("   - Need very fast prediction (many trees = slower)")
print("   - Limited memory (stores many trees)")
print("   - Need simple, interpretable model")

print("\n⚖️ Handling Imbalance: class_weight='balanced'")

print("\n🚀 Training Random Forest...")
start_time = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

train_time_rf = time.time() - start_time

acc_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_rf:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_rf:.4f}")
print(f"   - Precision: {precision_rf:.4f}")
print(f"   - Recall: {recall_rf:.4f}")
print(f"   - F1-Score: {f1_rf:.4f}")
print(f"   - ROC-AUC: {roc_auc_rf:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': acc_rf,
    'Precision': precision_rf,
    'Recall': recall_rf,
    'F1-Score': f1_rf,
    'ROC-AUC': roc_auc_rf,
    'Training Time (s)': train_time_rf
})

# ============================================================================
# 5.6 Gradient Boosting
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.6 GRADIENT BOOSTING")
print("-" * 80)

print("\n🔍 What is Gradient Boosting?")
print("   Ensemble method that builds trees sequentially.")
print("   Each new tree corrects errors of previous trees.")
print("   Uses gradient descent to minimize loss function.")

print("\n✅ When to use:")
print("   - Need high accuracy")
print("   - Structured/tabular data")
print("   - Willing to tune hyperparameters")
print("   - Have enough computational resources")

print("\n❌ When NOT to use:")
print("   - Very large datasets (slower than RF)")
print("   - Real-time predictions needed")
print("   - Prone to overfitting if not tuned properly")

print("\n🚀 Training Gradient Boosting...")
start_time = time.time()

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                random_state=RANDOM_STATE)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)

train_time_gb = time.time() - start_time

acc_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
roc_auc_gb = roc_auc_score(y_test, gb.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_gb:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_gb:.4f}")
print(f"   - Precision: {precision_gb:.4f}")
print(f"   - Recall: {recall_gb:.4f}")
print(f"   - F1-Score: {f1_gb:.4f}")
print(f"   - ROC-AUC: {roc_auc_gb:.4f}")

results.append({
    'Model': 'Gradient Boosting',
    'Accuracy': acc_gb,
    'Precision': precision_gb,
    'Recall': recall_gb,
    'F1-Score': f1_gb,
    'ROC-AUC': roc_auc_gb,
    'Training Time (s)': train_time_gb
})

# ============================================================================
# 5.7 XGBoost
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.7 XGBOOST")
print("-" * 80)

print("\n🔍 What is XGBoost?")
print("   Extreme Gradient Boosting - optimized gradient boosting.")
print("   Includes regularization, parallel processing, handling missing values.")
print("   One of the most popular algorithms in competitions!")

print("\n✅ When to use:")
print("   - Need state-of-the-art performance")
print("   - Kaggle competitions")
print("   - Structured/tabular data")
print("   - Have missing values (handles automatically)")

print("\n❌ When NOT to use:")
print("   - Unstructured data (images, text, audio)")
print("   - Need simple interpretable model")
print("   - Very limited computational resources")

print("\n⚖️ Handling Imbalance: scale_pos_weight parameter")
print(f"   scale_pos_weight = {len(y_train[y_train==0]) / len(y_train[y_train==1]):.2f}")

print("\n🚀 Training XGBoost...")
start_time = time.time()

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                            scale_pos_weight=scale_pos_weight,
                            random_state=RANDOM_STATE, eval_metric='logloss')
xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

train_time_xgb = time.time() - start_time

acc_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, xgb_clf.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_xgb:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_xgb:.4f}")
print(f"   - Precision: {precision_xgb:.4f}")
print(f"   - Recall: {recall_xgb:.4f}")
print(f"   - F1-Score: {f1_xgb:.4f}")
print(f"   - ROC-AUC: {roc_auc_xgb:.4f}")

results.append({
    'Model': 'XGBoost',
    'Accuracy': acc_xgb,
    'Precision': precision_xgb,
    'Recall': recall_xgb,
    'F1-Score': f1_xgb,
    'ROC-AUC': roc_auc_xgb,
    'Training Time (s)': train_time_xgb
})

# ============================================================================
# 5.8 LightGBM
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.8 LIGHTGBM")
print("-" * 80)

print("\n🔍 What is LightGBM?")
print("   Light Gradient Boosting Machine by Microsoft.")
print("   Faster training than XGBoost, uses leaf-wise tree growth.")
print("   Excellent for large datasets!")

print("\n✅ When to use:")
print("   - Large datasets (> 10,000 rows)")
print("   - Need fast training")
print("   - High-dimensional data")
print("   - Limited memory")

print("\n❌ When NOT to use:")
print("   - Small datasets (< 10,000 rows) - may overfit")
print("   - Need highest accuracy (XGBoost/CatBoost might be better)")

print("\n⚖️ Handling Imbalance: is_unbalance=True")

print("\n🚀 Training LightGBM...")
start_time = time.time()

lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             is_unbalance=True, random_state=RANDOM_STATE, verbose=-1)
lgb_clf.fit(X_train_scaled, y_train)
y_pred_lgb = lgb_clf.predict(X_test_scaled)

train_time_lgb = time.time() - start_time

acc_lgb = accuracy_score(y_test, y_pred_lgb)
precision_lgb = precision_score(y_test, y_pred_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
roc_auc_lgb = roc_auc_score(y_test, lgb_clf.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_lgb:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_lgb:.4f}")
print(f"   - Precision: {precision_lgb:.4f}")
print(f"   - Recall: {recall_lgb:.4f}")
print(f"   - F1-Score: {f1_lgb:.4f}")
print(f"   - ROC-AUC: {roc_auc_lgb:.4f}")

results.append({
    'Model': 'LightGBM',
    'Accuracy': acc_lgb,
    'Precision': precision_lgb,
    'Recall': recall_lgb,
    'F1-Score': f1_lgb,
    'ROC-AUC': roc_auc_lgb,
    'Training Time (s)': train_time_lgb
})

# ============================================================================
# 5.9 CatBoost
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.9 CATBOOST")
print("-" * 80)

print("\n🔍 What is CatBoost?")
print("   Categorical Boosting by Yandex.")
print("   Handles categorical features automatically (no encoding needed!).")
print("   Robust, less overfitting than other boosting algorithms.")

print("\n✅ When to use:")
print("   - Have categorical features")
print("   - Need robust model (less prone to overfitting)")
print("   - Want good performance with default parameters")
print("   - Don't want to spend time on feature engineering")

print("\n❌ When NOT to use:")
print("   - Need fastest training (slower than LightGBM)")
print("   - Very small datasets")

print("\n⚖️ Handling Imbalance: auto_class_weights='Balanced'")

print("\n🚀 Training CatBoost...")
start_time = time.time()

cat_clf = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                             auto_class_weights='Balanced',
                             random_state=RANDOM_STATE, verbose=0)
cat_clf.fit(X_train_scaled, y_train)
y_pred_cat = cat_clf.predict(X_test_scaled)

train_time_cat = time.time() - start_time

acc_cat = accuracy_score(y_test, y_pred_cat)
precision_cat = precision_score(y_test, y_pred_cat)
recall_cat = recall_score(y_test, y_pred_cat)
f1_cat = f1_score(y_test, y_pred_cat)
roc_auc_cat = roc_auc_score(y_test, cat_clf.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_cat:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_cat:.4f}")
print(f"   - Precision: {precision_cat:.4f}")
print(f"   - Recall: {recall_cat:.4f}")
print(f"   - F1-Score: {f1_cat:.4f}")
print(f"   - ROC-AUC: {roc_auc_cat:.4f}")

results.append({
    'Model': 'CatBoost',
    'Accuracy': acc_cat,
    'Precision': precision_cat,
    'Recall': recall_cat,
    'F1-Score': f1_cat,
    'ROC-AUC': roc_auc_cat,
    'Training Time (s)': train_time_cat
})

# ============================================================================
# 5.10 Support Vector Machine (SVM)
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.10 SUPPORT VECTOR MACHINE (SVM)")
print("-" * 80)

print("\n🔍 What is SVM?")
print("   Finds optimal hyperplane that maximizes margin between classes.")
print("   Can use kernel trick for non-linear decision boundaries.")

print("\n✅ When to use:")
print("   - High-dimensional data")
print("   - Clear margin of separation")
print("   - Small to medium datasets")
print("   - Binary classification")

print("\n❌ When NOT to use:")
print("   - Large datasets (very slow)")
print("   - Overlapping classes")
print("   - Need probability estimates (needs calibration)")

print("\n⚖️ Handling Imbalance: class_weight='balanced'")

print("\n🚀 Training SVM (RBF kernel)...")
start_time = time.time()

svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
              class_weight='balanced', random_state=RANDOM_STATE)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)

train_time_svm = time.time() - start_time

acc_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, svm_clf.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_svm:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_svm:.4f}")
print(f"   - Precision: {precision_svm:.4f}")
print(f"   - Recall: {recall_svm:.4f}")
print(f"   - F1-Score: {f1_svm:.4f}")
print(f"   - ROC-AUC: {roc_auc_svm:.4f}")

results.append({
    'Model': 'SVM (RBF)',
    'Accuracy': acc_svm,
    'Precision': precision_svm,
    'Recall': recall_svm,
    'F1-Score': f1_svm,
    'ROC-AUC': roc_auc_svm,
    'Training Time (s)': train_time_svm
})

# ============================================================================
# 5.11 Linear Discriminant Analysis (LDA)
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.11 LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("-" * 80)

print("\n🔍 What is LDA?")
print("   Finds linear combinations of features that separate classes.")
print("   Assumes Gaussian distribution and equal covariance matrices.")

print("\n✅ When to use:")
print("   - Need dimensionality reduction + classification")
print("   - Gaussian distributed features")
print("   - Small to medium datasets")
print("   - Interpretable linear model")

print("\n❌ When NOT to use:")
print("   - Non-Gaussian features")
print("   - Different covariance matrices per class")
print("   - Non-linear decision boundary")

print("\n🚀 Training LDA...")
start_time = time.time()

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred_lda = lda.predict(X_test_scaled)

train_time_lda = time.time() - start_time

acc_lda = accuracy_score(y_test, y_pred_lda)
precision_lda = precision_score(y_test, y_pred_lda)
recall_lda = recall_score(y_test, y_pred_lda)
f1_lda = f1_score(y_test, y_pred_lda)
roc_auc_lda = roc_auc_score(y_test, lda.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_lda:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_lda:.4f}")
print(f"   - Precision: {precision_lda:.4f}")
print(f"   - Recall: {recall_lda:.4f}")
print(f"   - F1-Score: {f1_lda:.4f}")
print(f"   - ROC-AUC: {roc_auc_lda:.4f}")

results.append({
    'Model': 'LDA',
    'Accuracy': acc_lda,
    'Precision': precision_lda,
    'Recall': recall_lda,
    'F1-Score': f1_lda,
    'ROC-AUC': roc_auc_lda,
    'Training Time (s)': train_time_lda
})

# ============================================================================
# 5.12 Quadratic Discriminant Analysis (QDA)
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.12 QUADRATIC DISCRIMINANT ANALYSIS (QDA)")
print("-" * 80)

print("\n🔍 What is QDA?")
print("   Like LDA but allows different covariance matrices per class.")
print("   Can model quadratic decision boundaries.")

print("\n✅ When to use:")
print("   - Classes have different covariance structures")
print("   - Quadratic decision boundary")
print("   - Gaussian distributed features")

print("\n❌ When NOT to use:")
print("   - Small datasets (needs more parameters than LDA)")
print("   - Many features compared to samples")
print("   - Non-Gaussian features")

print("\n🚀 Training QDA...")
start_time = time.time()

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)
y_pred_qda = qda.predict(X_test_scaled)

train_time_qda = time.time() - start_time

acc_qda = accuracy_score(y_test, y_pred_qda)
precision_qda = precision_score(y_test, y_pred_qda)
recall_qda = recall_score(y_test, y_pred_qda)
f1_qda = f1_score(y_test, y_pred_qda)
roc_auc_qda = roc_auc_score(y_test, qda.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_qda:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_qda:.4f}")
print(f"   - Precision: {precision_qda:.4f}")
print(f"   - Recall: {recall_qda:.4f}")
print(f"   - F1-Score: {f1_qda:.4f}")
print(f"   - ROC-AUC: {roc_auc_qda:.4f}")

results.append({
    'Model': 'QDA',
    'Accuracy': acc_qda,
    'Precision': precision_qda,
    'Recall': recall_qda,
    'F1-Score': f1_qda,
    'ROC-AUC': roc_auc_qda,
    'Training Time (s)': train_time_qda
})

# ============================================================================
# 5.13 AdaBoost
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.13 ADABOOST")
print("-" * 80)

print("\n🔍 What is AdaBoost?")
print("   Adaptive Boosting - combines multiple weak learners.")
print("   Each iteration focuses on misclassified samples.")
print("   Adjusts sample weights adaptively.")

print("\n✅ When to use:")
print("   - Need interpretable ensemble")
print("   - Want to boost weak learners")
print("   - Binary classification")
print("   - Small to medium datasets")

print("\n❌ When NOT to use:")
print("   - Noisy data or outliers (sensitive to them)")
print("   - Large datasets (slower)")
print("   - Multiclass with many classes")

print("\n🚀 Training AdaBoost...")
start_time = time.time()

ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
ada.fit(X_train_scaled, y_train)
y_pred_ada = ada.predict(X_test_scaled)

train_time_ada = time.time() - start_time

acc_ada = accuracy_score(y_test, y_pred_ada)
precision_ada = precision_score(y_test, y_pred_ada)
recall_ada = recall_score(y_test, y_pred_ada)
f1_ada = f1_score(y_test, y_pred_ada)
roc_auc_ada = roc_auc_score(y_test, ada.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_ada:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_ada:.4f}")
print(f"   - Precision: {precision_ada:.4f}")
print(f"   - Recall: {recall_ada:.4f}")
print(f"   - F1-Score: {f1_ada:.4f}")
print(f"   - ROC-AUC: {roc_auc_ada:.4f}")

results.append({
    'Model': 'AdaBoost',
    'Accuracy': acc_ada,
    'Precision': precision_ada,
    'Recall': recall_ada,
    'F1-Score': f1_ada,
    'ROC-AUC': roc_auc_ada,
    'Training Time (s)': train_time_ada
})

# ============================================================================
# 5.14 Extra Trees
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.14 EXTRA TREES")
print("-" * 80)

print("\n🔍 What is Extra Trees?")
print("   Extremely Randomized Trees - like Random Forest but more random.")
print("   Uses random thresholds for splits (not optimal ones).")
print("   Faster training, more randomization reduces overfitting.")

print("\n✅ When to use:")
print("   - Need faster training than Random Forest")
print("   - Want to reduce overfitting")
print("   - Large feature space")

print("\n❌ When NOT to use:")
print("   - Need highest accuracy (RF might be better)")
print("   - Very small datasets")

print("\n⚖️ Handling Imbalance: class_weight='balanced'")

print("\n🚀 Training Extra Trees...")
start_time = time.time()

et = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
                         class_weight='balanced', n_jobs=-1)
et.fit(X_train_scaled, y_train)
y_pred_et = et.predict(X_test_scaled)

train_time_et = time.time() - start_time

acc_et = accuracy_score(y_test, y_pred_et)
precision_et = precision_score(y_test, y_pred_et)
recall_et = recall_score(y_test, y_pred_et)
f1_et = f1_score(y_test, y_pred_et)
roc_auc_et = roc_auc_score(y_test, et.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_et:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_et:.4f}")
print(f"   - Precision: {precision_et:.4f}")
print(f"   - Recall: {recall_et:.4f}")
print(f"   - F1-Score: {f1_et:.4f}")
print(f"   - ROC-AUC: {roc_auc_et:.4f}")

results.append({
    'Model': 'Extra Trees',
    'Accuracy': acc_et,
    'Precision': precision_et,
    'Recall': recall_et,
    'F1-Score': f1_et,
    'ROC-AUC': roc_auc_et,
    'Training Time (s)': train_time_et
})

# ============================================================================
# 5.15 Balanced Random Forest
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.15 BALANCED RANDOM FOREST")
print("-" * 80)

print("\n🔍 What is Balanced Random Forest?")
print("   Random Forest specifically designed for imbalanced data!")
print("   Automatically balances classes by under-sampling majority class")
print("   in each bootstrap sample.")

print("\n✅ When to use:")
print("   - Imbalanced classification (perfect for our case!)")
print("   - Want automatic balancing without SMOTE")
print("   - Need robust ensemble model")

print("\n❌ When NOT to use:")
print("   - Balanced datasets (use regular RF)")
print("   - Very small minority class (< 50 samples)")

print("\n⚖️ This model is SPECIFICALLY designed for imbalanced data!")

print("\n🚀 Training Balanced Random Forest...")
start_time = time.time()

brf = BalancedRandomForestClassifier(n_estimators=100, max_depth=10,
                                     random_state=RANDOM_STATE, n_jobs=-1)
brf.fit(X_train_scaled, y_train)
y_pred_brf = brf.predict(X_test_scaled)

train_time_brf = time.time() - start_time

acc_brf = accuracy_score(y_test, y_pred_brf)
precision_brf = precision_score(y_test, y_pred_brf)
recall_brf = recall_score(y_test, y_pred_brf)
f1_brf = f1_score(y_test, y_pred_brf)
roc_auc_brf = roc_auc_score(y_test, brf.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_brf:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_brf:.4f}")
print(f"   - Precision: {precision_brf:.4f}")
print(f"   - Recall: {recall_brf:.4f}")
print(f"   - F1-Score: {f1_brf:.4f}")
print(f"   - ROC-AUC: {roc_auc_brf:.4f}")

results.append({
    'Model': 'Balanced Random Forest',
    'Accuracy': acc_brf,
    'Precision': precision_brf,
    'Recall': recall_brf,
    'F1-Score': f1_brf,
    'ROC-AUC': roc_auc_brf,
    'Training Time (s)': train_time_brf
})

print("\n" + "=" * 80)
print(" ✅ ALL 15 MODELS TRAINED! ".center(80, "="))
print("=" * 80)

# ============================================================================
# FASE 5.5: IMBALANCED DATA HANDLING WITH SMOTE & ADASYN
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 5.5: IMBALANCED DATA HANDLING (SMOTE & ADASYN) ".center(80, "="))
print("=" * 80)

print("\n🎯 Now we'll apply resampling techniques and compare with best model!")
print("   We'll use XGBoost (one of the best performers) for comparison.")

# ============================================================================
# 5.5.1 SMOTE (Synthetic Minority Over-sampling Technique)
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.5.1 SMOTE (Synthetic Minority Over-sampling Technique)")
print("-" * 80)

print("\n🔍 What is SMOTE?")
print("   Creates synthetic samples for minority class by interpolating")
print("   between existing minority class samples.")
print("   Formula: x_new = x + lambda * (x_neighbor - x)")

print("\n✅ Advantages:")
print("   - Reduces overfitting compared to random over-sampling")
print("   - Creates diverse synthetic samples")
print("   - Most popular resampling technique")

print("\n❌ Disadvantages:")
print("   - Can create noisy samples in overlapping regions")
print("   - Increases training time")

print(f"\n📊 Before SMOTE:")
print(f"   Class 0: {len(y_train[y_train==0])} samples")
print(f"   Class 1: {len(y_train[y_train==1])} samples")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\n📊 After SMOTE:")
print(f"   Class 0: {len(y_train_smote[y_train_smote==0])} samples")
print(f"   Class 1: {len(y_train_smote[y_train_smote==1])} samples")
print(f"   ✅ Perfectly balanced!")

print("\n🚀 Training XGBoost with SMOTE...")
start_time = time.time()

xgb_smote = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                              random_state=RANDOM_STATE, eval_metric='logloss')
xgb_smote.fit(X_train_smote, y_train_smote)
y_pred_xgb_smote = xgb_smote.predict(X_test_scaled)

train_time_xgb_smote = time.time() - start_time

acc_xgb_smote = accuracy_score(y_test, y_pred_xgb_smote)
precision_xgb_smote = precision_score(y_test, y_pred_xgb_smote)
recall_xgb_smote = recall_score(y_test, y_pred_xgb_smote)
f1_xgb_smote = f1_score(y_test, y_pred_xgb_smote)
roc_auc_xgb_smote = roc_auc_score(y_test, xgb_smote.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_xgb_smote:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_xgb_smote:.4f}")
print(f"   - Precision: {precision_xgb_smote:.4f}")
print(f"   - Recall: {recall_xgb_smote:.4f}")
print(f"   - F1-Score: {f1_xgb_smote:.4f}")
print(f"   - ROC-AUC: {roc_auc_xgb_smote:.4f}")

results.append({
    'Model': 'XGBoost + SMOTE',
    'Accuracy': acc_xgb_smote,
    'Precision': precision_xgb_smote,
    'Recall': recall_xgb_smote,
    'F1-Score': f1_xgb_smote,
    'ROC-AUC': roc_auc_xgb_smote,
    'Training Time (s)': train_time_xgb_smote
})

# ============================================================================
# 5.5.2 ADASYN (Adaptive Synthetic Sampling)
# ============================================================================

print("\n" + "-" * 80)
print("📊 5.5.2 ADASYN (Adaptive Synthetic Sampling)")
print("-" * 80)

print("\n🔍 What is ADASYN?")
print("   Like SMOTE but focuses more on hard-to-learn samples.")
print("   Generates more synthetic data for minority samples that are")
print("   harder to classify (closer to decision boundary).")

print("\n✅ Advantages:")
print("   - Adaptive - focuses on difficult regions")
print("   - Better for complex decision boundaries")
print("   - Reduces bias better than SMOTE")

print("\n❌ Disadvantages:")
print("   - Can amplify noise")
print("   - Slightly slower than SMOTE")

print(f"\n📊 Before ADASYN:")
print(f"   Class 0: {len(y_train[y_train==0])} samples")
print(f"   Class 1: {len(y_train[y_train==1])} samples")

adasyn = ADASYN(random_state=RANDOM_STATE)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_scaled, y_train)

print(f"\n📊 After ADASYN:")
print(f"   Class 0: {len(y_train_adasyn[y_train_adasyn==0])} samples")
print(f"   Class 1: {len(y_train_adasyn[y_train_adasyn==1])} samples")
print(f"   ✅ Nearly balanced (may not be exactly 50-50)!")

print("\n🚀 Training XGBoost with ADASYN...")
start_time = time.time()

xgb_adasyn = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=RANDOM_STATE, eval_metric='logloss')
xgb_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_xgb_adasyn = xgb_adasyn.predict(X_test_scaled)

train_time_xgb_adasyn = time.time() - start_time

acc_xgb_adasyn = accuracy_score(y_test, y_pred_xgb_adasyn)
precision_xgb_adasyn = precision_score(y_test, y_pred_xgb_adasyn)
recall_xgb_adasyn = recall_score(y_test, y_pred_xgb_adasyn)
f1_xgb_adasyn = f1_score(y_test, y_pred_xgb_adasyn)
roc_auc_xgb_adasyn = roc_auc_score(y_test, xgb_adasyn.predict_proba(X_test_scaled)[:, 1])

print(f"\n✅ Trained in {train_time_xgb_adasyn:.4f} seconds")
print(f"\n📊 Performance:")
print(f"   - Accuracy: {acc_xgb_adasyn:.4f}")
print(f"   - Precision: {precision_xgb_adasyn:.4f}")
print(f"   - Recall: {recall_xgb_adasyn:.4f}")
print(f"   - F1-Score: {f1_xgb_adasyn:.4f}")
print(f"   - ROC-AUC: {roc_auc_xgb_adasyn:.4f}")

results.append({
    'Model': 'XGBoost + ADASYN',
    'Accuracy': acc_xgb_adasyn,
    'Precision': precision_xgb_adasyn,
    'Recall': recall_xgb_adasyn,
    'F1-Score': f1_xgb_adasyn,
    'ROC-AUC': roc_auc_xgb_adasyn,
    'Training Time (s)': train_time_xgb_adasyn
})

print("\n" + "=" * 80)

# ============================================================================
# FASE 6: CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 6: CROSS-VALIDATION ".center(80, "="))
print("=" * 80)

print("\n🎯 Cross-Validation helps us:")
print("   - Get more reliable performance estimates")
print("   - Detect overfitting")
print("   - Use all data for both training and validation")

print("\n📊 We'll use STRATIFIED K-Fold Cross-Validation")
print("   Why STRATIFIED? To preserve class distribution in each fold!")
print("   K = 5 folds")

print("\n🚀 Running 5-Fold Stratified Cross-Validation on top 5 models...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Select top 5 models based on F1-Score
results_df_temp = pd.DataFrame(results)
top_5_models = results_df_temp.nlargest(5, 'F1-Score')['Model'].tolist()

print(f"\n📋 Top 5 models for CV:")
for i, model_name in enumerate(top_5_models, 1):
    print(f"   {i}. {model_name}")

# Define models for CV
models_cv = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, is_unbalance=True, random_state=RANDOM_STATE, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, auto_class_weights='Balanced', random_state=RANDOM_STATE, verbose=0),
    'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost + SMOTE': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric='logloss'),
}

cv_results = {}

for model_name in top_5_models:
    if model_name in models_cv:
        print(f"\n   Running CV for {model_name}...")
        model = models_cv[model_name]

        # Handle SMOTE separately
        if 'SMOTE' in model_name or 'ADASYN' in model_name:
            # For SMOTE/ADASYN, we need to apply resampling in each fold
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train_scaled, y_train):
                X_train_fold = X_train_scaled.iloc[train_idx]
                y_train_fold = y_train.iloc[train_idx]
                X_val_fold = X_train_scaled.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]

                # Apply SMOTE
                smote_fold = SMOTE(random_state=RANDOM_STATE)
                X_train_fold_resampled, y_train_fold_resampled = smote_fold.fit_resample(X_train_fold, y_train_fold)

                # Train and evaluate
                model.fit(X_train_fold_resampled, y_train_fold_resampled)
                y_pred_fold = model.predict(X_val_fold)
                cv_scores.append(f1_score(y_val_fold, y_pred_fold))

            cv_scores = np.array(cv_scores)
        else:
            # Standard CV
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                       cv=skf, scoring='f1', n_jobs=-1)

        cv_results[model_name] = {
            'CV Mean F1': cv_scores.mean(),
            'CV Std F1': cv_scores.std(),
            'CV Scores': cv_scores
        }

        print(f"      ✅ CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n📊 Cross-Validation Summary:")
print(f"\n{'Model':<30} {'CV Mean F1':<15} {'CV Std F1':<15}")
print("-" * 60)
for model_name, scores in cv_results.items():
    print(f"{model_name:<30} {scores['CV Mean F1']:<15.4f} {scores['CV Std F1']:<15.4f}")

print("\n" + "=" * 80)

# ============================================================================
# FASE 7: HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 7: HYPERPARAMETER TUNING ".center(80, "="))
print("=" * 80)

print("\n🎯 We'll demonstrate 3 hyperparameter tuning techniques:")
print("   1. Grid Search CV - Exhaustive search")
print("   2. Random Search CV - Random sampling")
print("   3. Bayesian Optimization (Optuna) - Smart search")

# ============================================================================
# 7.1 Grid Search CV (Random Forest)
# ============================================================================

print("\n" + "-" * 80)
print("📊 7.1 GRID SEARCH CV (Random Forest)")
print("-" * 80)

print("\n🔍 Grid Search:")
print("   - Tests ALL combinations of parameters")
print("   - Guarantees finding best combination in search space")
print("   - Can be slow for large parameter grids")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print(f"\n📋 Parameter Grid:")
for param, values in param_grid_rf.items():
    print(f"   - {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid_rf.values()])
print(f"\n   Total combinations: {total_combinations}")

print("\n🚀 Running Grid Search (this may take a while)...")
start_time = time.time()

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    param_grid_rf,
    cv=3,  # Use 3-fold to speed up
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search_rf.fit(X_train_scaled, y_train)
grid_search_time = time.time() - start_time

print(f"\n✅ Grid Search completed in {grid_search_time:.2f} seconds")
print(f"\n🎯 Best Parameters:")
for param, value in grid_search_rf.best_params_.items():
    print(f"   - {param}: {value}")

print(f"\n📊 Best CV F1-Score: {grid_search_rf.best_score_:.4f}")

# Evaluate on test set
y_pred_grid_rf = grid_search_rf.predict(X_test_scaled)
f1_grid_rf = f1_score(y_test, y_pred_grid_rf)
print(f"   Test F1-Score: {f1_grid_rf:.4f}")

results.append({
    'Model': 'Random Forest (Grid Search)',
    'Accuracy': accuracy_score(y_test, y_pred_grid_rf),
    'Precision': precision_score(y_test, y_pred_grid_rf),
    'Recall': recall_score(y_test, y_pred_grid_rf),
    'F1-Score': f1_grid_rf,
    'ROC-AUC': roc_auc_score(y_test, grid_search_rf.predict_proba(X_test_scaled)[:, 1]),
    'Training Time (s)': grid_search_time
})

# ============================================================================
# 7.2 Random Search CV (XGBoost)
# ============================================================================

print("\n" + "-" * 80)
print("📊 7.2 RANDOM SEARCH CV (XGBoost)")
print("-" * 80)

print("\n🔍 Random Search:")
print("   - Samples random combinations of parameters")
print("   - Faster than Grid Search")
print("   - Can explore larger parameter space")
print("   - Often finds good solutions quickly")

param_dist_xgb = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [scale_pos_weight]
}

print(f"\n📋 Parameter Distributions:")
for param, values in param_dist_xgb.items():
    print(f"   - {param}: {values}")

n_iter = 20
print(f"\n   Number of random samples: {n_iter}")

print("\n🚀 Running Random Search...")
start_time = time.time()

random_search_xgb = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
    param_dist_xgb,
    n_iter=n_iter,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)

random_search_xgb.fit(X_train_scaled, y_train)
random_search_time = time.time() - start_time

print(f"\n✅ Random Search completed in {random_search_time:.2f} seconds")
print(f"\n🎯 Best Parameters:")
for param, value in random_search_xgb.best_params_.items():
    print(f"   - {param}: {value}")

print(f"\n📊 Best CV F1-Score: {random_search_xgb.best_score_:.4f}")

# Evaluate on test set
y_pred_random_xgb = random_search_xgb.predict(X_test_scaled)
f1_random_xgb = f1_score(y_test, y_pred_random_xgb)
print(f"   Test F1-Score: {f1_random_xgb:.4f}")

results.append({
    'Model': 'XGBoost (Random Search)',
    'Accuracy': accuracy_score(y_test, y_pred_random_xgb),
    'Precision': precision_score(y_test, y_pred_random_xgb),
    'Recall': recall_score(y_test, y_pred_random_xgb),
    'F1-Score': f1_random_xgb,
    'ROC-AUC': roc_auc_score(y_test, random_search_xgb.predict_proba(X_test_scaled)[:, 1]),
    'Training Time (s)': random_search_time
})

# ============================================================================
# 7.3 Bayesian Optimization with Optuna (LightGBM)
# ============================================================================

print("\n" + "-" * 80)
print("📊 7.3 BAYESIAN OPTIMIZATION WITH OPTUNA (LightGBM)")
print("-" * 80)

print("\n🔍 Bayesian Optimization:")
print("   - Builds probabilistic model of objective function")
print("   - Uses past trials to inform next parameter selection")
print("   - Most efficient search method")
print("   - Finds good solutions with fewer trials")

print("\n🚀 Running Bayesian Optimization with Optuna...")
print("   (Suppressing Optuna output for cleaner logs)")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_lgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'is_unbalance': True,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**param)
    score = cross_val_score(model, X_train_scaled, y_train,
                           cv=3, scoring='f1', n_jobs=-1).mean()
    return score

start_time = time.time()

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=False)

optuna_time = time.time() - start_time

print(f"\n✅ Bayesian Optimization completed in {optuna_time:.2f} seconds")
print(f"\n🎯 Best Parameters:")
for param, value in study_lgb.best_params.items():
    print(f"   - {param}: {value}")

print(f"\n📊 Best CV F1-Score: {study_lgb.best_value:.4f}")

# Train final model with best parameters
best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, is_unbalance=True,
                              random_state=RANDOM_STATE, verbose=-1)
best_lgb.fit(X_train_scaled, y_train)
y_pred_optuna_lgb = best_lgb.predict(X_test_scaled)
f1_optuna_lgb = f1_score(y_test, y_pred_optuna_lgb)

print(f"   Test F1-Score: {f1_optuna_lgb:.4f}")

results.append({
    'Model': 'LightGBM (Optuna)',
    'Accuracy': accuracy_score(y_test, y_pred_optuna_lgb),
    'Precision': precision_score(y_test, y_pred_optuna_lgb),
    'Recall': recall_score(y_test, y_pred_optuna_lgb),
    'F1-Score': f1_optuna_lgb,
    'ROC-AUC': roc_auc_score(y_test, best_lgb.predict_proba(X_test_scaled)[:, 1]),
    'Training Time (s)': optuna_time
})

print("\n" + "=" * 80)

# ============================================================================
# FASE 8: MODEL EVALUATION & COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 8: MODEL EVALUATION & COMPARISON ".center(80, "="))
print("=" * 80)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

print("\n📊 Complete Model Comparison (Sorted by F1-Score):")
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('outputs/model_results.csv', index=False)
print("\n💾 Results saved to: outputs/model_results.csv")

# ============================================================================
# 8.1 Model Comparison Visualizations
# ============================================================================

print("\n📊 8.1 Generating Model Comparison Visualizations...")

# Top 10 models for visualization
top_10_models = results_df.head(10)

# Figure 1: F1-Score Comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# F1-Score
ax1 = axes[0, 0]
bars1 = ax1.barh(top_10_models['Model'], top_10_models['F1-Score'], color='skyblue')
ax1.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax1.set_title('F1-Score Comparison (Top 10 Models)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1)
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
             ha='left', va='center', fontsize=9)

# Recall
ax2 = axes[0, 1]
bars2 = ax2.barh(top_10_models['Model'], top_10_models['Recall'], color='lightcoral')
ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax2.set_title('Recall Comparison (Top 10 Models)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 1)
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
             ha='left', va='center', fontsize=9)

# Precision
ax3 = axes[1, 0]
bars3 = ax3.barh(top_10_models['Model'], top_10_models['Precision'], color='lightgreen')
ax3.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax3.set_title('Precision Comparison (Top 10 Models)', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 1)
for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
             ha='left', va='center', fontsize=9)

# ROC-AUC
ax4 = axes[1, 1]
bars4 = ax4.barh(top_10_models['Model'], top_10_models['ROC-AUC'], color='plum')
ax4.set_xlabel('ROC-AUC', fontsize=11, fontweight='bold')
ax4.set_title('ROC-AUC Comparison (Top 10 Models)', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 1)
for i, bar in enumerate(bars4):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
             ha='left', va='center', fontsize=9)

plt.suptitle('Model Performance Comparison\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/04_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/04_model_comparison.png")

# Figure 2: Training Time Comparison
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(top_10_models['Model'], top_10_models['Training Time (s)'], color='orange', alpha=0.7)
ax.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
ax.set_title('Training Time Comparison (Top 10 Models)\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=12, fontweight='bold')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}s',
            ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/05_training_time.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/05_training_time.png")

# Figure 3: Metrics Radar Chart for Top 5
print("\n📊 8.2 Creating Radar Chart for Top 5 Models...")

from math import pi

top_5_for_radar = results_df.head(5)
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.xticks(angles[:-1], categories, size=11)
ax.set_ylim(0, 1)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for idx, (i, row) in enumerate(top_5_for_radar.iterrows()):
    values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.title('Top 5 Models - Performance Radar Chart\n© Muhammad Ketsar Ali Abi Wahid',
          size=13, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('outputs/06_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/06_radar_chart.png")

# ============================================================================
# 8.3 Confusion Matrix for Best Model
# ============================================================================

print("\n📊 8.3 Confusion Matrix for Best Model...")

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")

# Get predictions from best model
# Find which model variable corresponds to best model
if 'SMOTE' in best_model_name:
    y_pred_best = y_pred_xgb_smote
elif 'ADASYN' in best_model_name:
    y_pred_best = y_pred_xgb_adasyn
elif 'Grid Search' in best_model_name:
    y_pred_best = y_pred_grid_rf
elif 'Random Search' in best_model_name:
    y_pred_best = y_pred_random_xgb
elif 'Optuna' in best_model_name:
    y_pred_best = y_pred_optuna_lgb
elif 'Balanced Random Forest' in best_model_name:
    y_pred_best = y_pred_brf
elif 'XGBoost' in best_model_name:
    y_pred_best = y_pred_xgb
elif 'LightGBM' in best_model_name:
    y_pred_best = y_pred_lgb
elif 'CatBoost' in best_model_name:
    y_pred_best = y_pred_cat
elif 'Random Forest' in best_model_name:
    y_pred_best = y_pred_rf
elif 'Gradient Boosting' in best_model_name:
    y_pred_best = y_pred_gb
else:
    # Default to XGBoost
    y_pred_best = y_pred_xgb

cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            xticklabels=['No Sand', 'Sand'], yticklabels=['No Sand', 'Sand'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name}\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=13, fontweight='bold', pad=15)

# Add text annotations
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, 1.8, f'TN={tn}', ha='center', fontsize=10, color='darkblue')
ax.text(1.5, 1.8, f'FP={fp}', ha='center', fontsize=10, color='darkred')
ax.text(0.5, 2.8, f'FN={fn}', ha='center', fontsize=10, color='darkred')
ax.text(1.5, 2.8, f'TP={tp}', ha='center', fontsize=10, color='darkblue')

plt.tight_layout()
plt.savefig('outputs/07_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/07_confusion_matrix.png")

print(f"\n📊 Confusion Matrix Breakdown:")
print(f"   True Negatives (TN):  {tn}")
print(f"   False Positives (FP): {fp}")
print(f"   False Negatives (FN): {fn}")
print(f"   True Positives (TP):  {tp}")

# ============================================================================
# 8.4 ROC Curve
# ============================================================================

print("\n📊 8.4 ROC Curve for Top 5 Models...")

fig, ax = plt.subplots(figsize=(10, 8))

# Get top 5 models and their predictions
models_for_roc = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb_clf,
    'LightGBM': lgb_clf,
    'CatBoost': cat_clf,
    'Balanced Random Forest': brf,
    'XGBoost + SMOTE': xgb_smote,
}

colors_roc = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']

for idx, (model_name, model) in enumerate(list(models_for_roc.items())[:5]):
    if model_name in top_5_for_radar['Model'].values:
        try:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc_score:.4f})',
                   color=colors_roc[idx])
        except:
            pass

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC=0.5000)')

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Top 5 Models\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/08_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/08_roc_curves.png")

# ============================================================================
# 8.5 Precision-Recall Curve
# ============================================================================

print("\n📊 8.5 Precision-Recall Curve for Top 5 Models...")

fig, ax = plt.subplots(figsize=(10, 8))

for idx, (model_name, model) in enumerate(list(models_for_roc.items())[:5]):
    if model_name in top_5_for_radar['Model'].values:
        try:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            ax.plot(recall, precision, linewidth=2,
                   label=f'{model_name} (AP={ap_score:.4f})',
                   color=colors_roc[idx])
        except:
            pass

# Baseline (no skill)
no_skill = len(y_test[y_test==1]) / len(y_test)
ax.plot([0, 1], [no_skill, no_skill], 'k--', linewidth=1,
        label=f'No Skill (AP={no_skill:.4f})')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Top 5 Models\n© Muhammad Ketsar Ali Abi Wahid',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/09_precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: outputs/09_precision_recall_curves.png")

print("\n" + "=" * 80)

# ============================================================================
# FASE 9: MODEL INTERPRETATION
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 9: MODEL INTERPRETATION ".center(80, "="))
print("=" * 80)

# ============================================================================
# 9.1 Feature Importance (Best Tree-based Model)
# ============================================================================

print("\n📊 9.1 Feature Importance Analysis...")

# Find best tree-based model from top 5
tree_based_models = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost',
                     'Gradient Boosting', 'Balanced Random Forest', 'Extra Trees']

best_tree_model = None
best_tree_model_name = None

for model_name in results_df.head(10)['Model']:
    for tree_name in tree_based_models:
        if tree_name in model_name:
            best_tree_model_name = model_name
            # Get the corresponding model object
            if 'Random Forest' in model_name and 'Balanced' not in model_name:
                if 'Grid Search' in model_name:
                    best_tree_model = grid_search_rf.best_estimator_
                else:
                    best_tree_model = rf
            elif 'Balanced Random Forest' in model_name:
                best_tree_model = brf
            elif 'XGBoost' in model_name:
                if 'SMOTE' in model_name:
                    best_tree_model = xgb_smote
                elif 'Random Search' in model_name:
                    best_tree_model = random_search_xgb.best_estimator_
                else:
                    best_tree_model = xgb_clf
            elif 'LightGBM' in model_name:
                if 'Optuna' in model_name:
                    best_tree_model = best_lgb
                else:
                    best_tree_model = lgb_clf
            elif 'CatBoost' in model_name:
                best_tree_model = cat_clf
            elif 'Gradient Boosting' in model_name:
                best_tree_model = gb
            elif 'Extra Trees' in model_name:
                best_tree_model = et
            break
    if best_tree_model is not None:
        break

if best_tree_model is None:
    # Default to XGBoost
    best_tree_model = xgb_clf
    best_tree_model_name = 'XGBoost'

print(f"\n🌳 Using {best_tree_model_name} for feature importance analysis")

# Get feature importance
if hasattr(best_tree_model, 'feature_importances_'):
    importances = best_tree_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(f"\n📊 Top 10 Most Important Features:")
    for i, row in enumerate(feature_importance_df.head(10).itertuples(), 1):
        print(f"   {i:2d}. {row.Feature:30s}: {row.Importance:.6f}")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'],
                   color='steelblue', alpha=0.8)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance - {best_tree_model_name}\n© Muhammad Ketsar Ali Abi Wahid',
                 fontsize=13, fontweight='bold', pad=15)
    ax.invert_yaxis()

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {width:.4f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/10_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n   ✅ Saved: outputs/10_feature_importance.png")

# ============================================================================
# 9.2 SHAP Analysis
# ============================================================================

print("\n📊 9.2 SHAP (SHapley Additive exPlanations) Analysis...")

print("\n🔍 What is SHAP?")
print("   - Unified framework for interpreting model predictions")
print("   - Based on game theory (Shapley values)")
print("   - Shows feature contribution to each prediction")
print("   - Works for any model!")

print(f"\n🚀 Computing SHAP values for {best_tree_model_name}...")
print("   (This may take a few minutes...)")

# Use a sample for faster computation
n_samples_shap = min(200, len(X_test_scaled))
X_test_sample = X_test_scaled.sample(n=n_samples_shap, random_state=RANDOM_STATE)

try:
    # TreeExplainer for tree-based models (faster)
    explainer = shap.TreeExplainer(best_tree_model)
    shap_values = explainer.shap_values(X_test_sample)

    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # For binary classification, we want positive class
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    print("\n   ✅ SHAP values computed!")

    # SHAP Summary Plot
    print("\n   Creating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_plot, X_test_sample, show=False)
    plt.title(f'SHAP Summary Plot - {best_tree_model_name}\n© Muhammad Ketsar Ali Abi Wahid',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/11_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/11_shap_summary.png")

    # SHAP Importance Plot
    print("\n   Creating SHAP importance plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_plot, X_test_sample, plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance - {best_tree_model_name}\n© Muhammad Ketsar Ali Abi Wahid',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/12_shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/12_shap_importance.png")

    print("\n📊 SHAP Interpretation:")
    print("   - Red = High feature value")
    print("   - Blue = Low feature value")
    print("   - Right = Positive impact (increases sand production probability)")
    print("   - Left = Negative impact (decreases sand production probability)")

except Exception as e:
    print(f"\n   ⚠️ SHAP analysis failed: {e}")
    print("   Continuing with other analyses...")

print("\n" + "=" * 80)

# ============================================================================
# FASE 10: FINAL MODEL SELECTION & REPORT
# ============================================================================

print("\n" + "=" * 80)
print(" FASE 10: FINAL MODEL SELECTION & REPORT ".center(80, "="))
print("=" * 80)

# ============================================================================
# 10.1 Final Model Selection
# ============================================================================

print("\n📊 10.1 Final Model Selection Criteria:")

print("\n🎯 Selection Criteria:")
print("   1. F1-Score (PRIMARY) - Balance between Precision and Recall")
print("   2. Recall - Critical for sand production (don't miss cases!)")
print("   3. ROC-AUC - Overall discriminative ability")
print("   4. Training Time - Practical considerations")
print("   5. Model Complexity - Interpretability vs Performance")

print("\n🏆 Top 5 Models by F1-Score:")
top_5_final = results_df.head(5)
print("\n" + top_5_final.to_string(index=False))

best_model_final = results_df.iloc[0]
print(f"\n✅ SELECTED BEST MODEL: {best_model_final['Model']}")
print(f"\n📊 Performance Metrics:")
print(f"   - Accuracy:  {best_model_final['Accuracy']:.4f}")
print(f"   - Precision: {best_model_final['Precision']:.4f}")
print(f"   - Recall:    {best_model_final['Recall']:.4f}")
print(f"   - F1-Score:  {best_model_final['F1-Score']:.4f}")
print(f"   - ROC-AUC:   {best_model_final['ROC-AUC']:.4f}")
print(f"   - Training Time: {best_model_final['Training Time (s)']:.2f} seconds")

# ============================================================================
# 10.2 Classification Report
# ============================================================================

print("\n📊 10.2 Detailed Classification Report:")

print(f"\n{classification_report(y_test, y_pred_best)}")

# ============================================================================
# 10.3 Additional Metrics
# ============================================================================

print("\n📊 10.3 Additional Metrics:")

mcc = matthews_corrcoef(y_test, y_pred_best)
kappa = cohen_kappa_score(y_test, y_pred_best)

print(f"\n   Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"   Cohen's Kappa: {kappa:.4f}")

print("\n   ℹ️ MCC: Correlation between predicted and actual labels (-1 to +1)")
print("   ℹ️ Cohen's Kappa: Agreement between raters, accounting for chance")

# ============================================================================
# 10.4 Business Impact Analysis
# ============================================================================

print("\n📊 10.4 Business Impact Analysis:")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

print(f"\n   Out of {len(y_test)} wells:")
print(f"   ✅ Correctly identified NO sand: {tn} wells")
print(f"   ✅ Correctly identified SAND: {tp} wells")
print(f"   ❌ False alarms (predicted sand, but no sand): {fp} wells")
print(f"   ❌ Missed sand cases (predicted no sand, but has sand): {fn} wells")

print(f"\n💡 Business Insights:")
print(f"   - Detection Rate: {tp/(tp+fn)*100:.2f}% of sand cases detected")
print(f"   - False Alarm Rate: {fp/(fp+tn)*100:.2f}% of no-sand wells misclassified")
print(f"   - Missed Cases: {fn} critical cases missed (requires attention!)")

if fn > 0:
    print(f"\n⚠️ RECOMMENDATION:")
    print(f"   Consider adjusting threshold to increase recall and reduce missed cases.")
    print(f"   Missing sand production = potential equipment damage!")

# ============================================================================
# 10.5 Save Final Model
# ============================================================================

print("\n📊 10.5 Saving Final Model and Scaler...")

# Determine which model to save
if 'SMOTE' in best_model_final['Model']:
    final_model_to_save = xgb_smote
elif 'ADASYN' in best_model_final['Model']:
    final_model_to_save = xgb_adasyn
elif 'Grid Search' in best_model_final['Model']:
    final_model_to_save = grid_search_rf.best_estimator_
elif 'Random Search' in best_model_final['Model']:
    final_model_to_save = random_search_xgb.best_estimator_
elif 'Optuna' in best_model_final['Model']:
    final_model_to_save = best_lgb
elif 'Balanced Random Forest' in best_model_final['Model']:
    final_model_to_save = brf
elif 'XGBoost' in best_model_final['Model']:
    final_model_to_save = xgb_clf
elif 'LightGBM' in best_model_final['Model']:
    final_model_to_save = lgb_clf
elif 'CatBoost' in best_model_final['Model']:
    final_model_to_save = cat_clf
elif 'Random Forest' in best_model_final['Model']:
    final_model_to_save = rf
else:
    final_model_to_save = xgb_clf

# Save model
model_filename = f"models/best_model_{best_model_final['Model'].replace(' ', '_').replace('+', 'plus')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(final_model_to_save, f)

print(f"   ✅ Model saved: {model_filename}")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"   ✅ Scaler saved: models/scaler.pkl")

# ============================================================================
# 10.6 Usage Example
# ============================================================================

print("\n📊 10.6 How to Use Saved Model:")

print("""
# Load model and scaler
import pickle

with open('models/best_model_*.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data (must have same features as training data)
new_data = pd.DataFrame({...})  # Your 15 features

# Scale data
new_data_scaled = scaler.transform(new_data)

# Predict
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)

print(f"Prediction: {predictions[0]}")  # 0 or 1
print(f"Probability of sand: {probabilities[0][1]:.4f}")
""")

# ============================================================================
# 10.7 Final Summary Report
# ============================================================================

print("\n" + "=" * 80)
print(" FINAL SUMMARY REPORT ".center(80, "="))
print("=" * 80)

print(f"""
🎯 PROJECT SUMMARY
================================================================================

Dataset: Sand Production Prediction (Oil & Gas Industry)
  - Total samples: 5,000 wells
  - Features: 15 (geological and operational parameters)
  - Target: Binary (0 = No sand, 1 = Sand)
  - Imbalance ratio: 5.67:1 (85% no sand, 15% sand)

📊 BEST MODEL: {best_model_final['Model']}

Performance Metrics:
  - Accuracy:  {best_model_final['Accuracy']:.4f} ({best_model_final['Accuracy']*100:.2f}%)
  - Precision: {best_model_final['Precision']:.4f} (of predicted sand, {best_model_final['Precision']*100:.2f}% are correct)
  - Recall:    {best_model_final['Recall']:.4f} (detected {best_model_final['Recall']*100:.2f}% of actual sand cases)
  - F1-Score:  {best_model_final['F1-Score']:.4f} (harmonic mean of Precision & Recall)
  - ROC-AUC:   {best_model_final['ROC-AUC']:.4f} (discriminative ability)

Training Time: {best_model_final['Training Time (s)']:.2f} seconds

🎓 KEY LEARNINGS:
================================================================================

1. ⚖️ Imbalanced Data Handling:
   - Class weights, SMOTE, ADASYN are CRITICAL for imbalanced classification
   - Accuracy is USELESS metric - use F1-Score, Recall instead
   - Always use stratified splitting and cross-validation

2. 📊 Metrics Matter:
   - F1-Score balances Precision and Recall
   - Recall is critical when missing positive cases is costly
   - ROC-AUC shows overall discriminative ability
   - Always check confusion matrix for business insights

3. 🔧 Hyperparameter Tuning:
   - Grid Search: exhaustive but slow
   - Random Search: good balance of speed and performance
   - Bayesian Optimization (Optuna): most efficient for large search spaces

4. 🌳 Model Selection:
   - Tree-based models (XGBoost, LightGBM, RF) excel at tabular data
   - Ensemble methods generally outperform single models
   - Balance performance vs interpretability vs training time

5. 🔍 Model Interpretation:
   - SHAP provides unified framework for any model
   - Feature importance helps understand key drivers
   - Always validate findings with domain knowledge

📂 GENERATED FILES:
================================================================================

Models:
  ✅ {model_filename}
  ✅ models/scaler.pkl

Results:
  ✅ outputs/model_results.csv

Visualizations (12 plots):
  ✅ outputs/01_correlation_heatmap.png
  ✅ outputs/02_target_distribution.png
  ✅ outputs/03_feature_distributions.png
  ✅ outputs/04_model_comparison.png
  ✅ outputs/05_training_time.png
  ✅ outputs/06_radar_chart.png
  ✅ outputs/07_confusion_matrix.png
  ✅ outputs/08_roc_curves.png
  ✅ outputs/09_precision_recall_curves.png
  ✅ outputs/10_feature_importance.png
  ✅ outputs/11_shap_summary.png
  ✅ outputs/12_shap_importance.png

🚀 NEXT STEPS:
================================================================================

1. ✅ Deploy model as REST API (Module 30 - FastAPI)
2. ✅ Set up monitoring dashboard (Module 32 - Model Monitoring)
3. ✅ Implement A/B testing (Module 34 - Experimentation)
4. ✅ Create automated retraining pipeline (Module 28 - MLflow)
5. ✅ Build real-time prediction service (Module 31 - Deployment)

💡 BUSINESS RECOMMENDATIONS:
================================================================================

1. Model Deployment:
   - Deploy {best_model_final['Model']} for production use
   - Set threshold based on business cost of false negatives vs false positives
   - Monitor model performance continuously

2. Feature Engineering:
   - Top features identified: {', '.join(feature_importance_df.head(3)['Feature'].tolist()) if 'feature_importance_df' in locals() else 'See feature importance plot'}
   - Consider collecting more data on these critical features
   - Explore domain-specific feature combinations

3. Risk Management:
   - Current model misses {fn} out of {tp+fn} sand cases ({fn/(tp+fn)*100:.1f}%)
   - Consider ensemble with multiple models for critical predictions
   - Implement human-in-the-loop for borderline cases

4. Continuous Improvement:
   - Collect feedback on model predictions
   - Retrain model quarterly with new data
   - Monitor for data drift and model degradation

================================================================================
© Muhammad Ketsar Ali Abi Wahid
Data Science Zero to Hero: Complete MLOps & Production ML Engineering
================================================================================

✅ MODULE 17 - CLASSIFICATION COMPLETE!

Thank you for using this comprehensive classification pipeline!
For questions or feedback, please refer to the documentation.

Happy Learning! 🚀
================================================================================
""")
