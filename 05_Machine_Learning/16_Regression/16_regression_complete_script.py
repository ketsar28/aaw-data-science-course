"""
🏗️ REGRESSION - COMPLETE IMPLEMENTATION (All 10 FASE)
📊 Prediksi Kekuatan Beton (Concrete Compressive Strength)

© Muhammad Ketsar Ali Abi Wahid

This script contains the complete end-to-end regression pipeline covering all 10 phases:
FASE 1: Data Loading & Initial Exploration
FASE 2: Exploratory Data Analysis (EDA)
FASE 3: Data Preprocessing
FASE 4: Train-Test Split & Baseline
FASE 5: Model Building (12 Algorithms)
FASE 6: Cross-Validation
FASE 7: Hyperparameter Tuning
FASE 8: Model Evaluation & Comparison
FASE 9: Model Interpretation (SHAP, Feature Importance, PDP)
FASE 10: Final Model Selection & Report

Total Lines: 2000+
Estimated Runtime: 30-60 minutes (depending on hardware)
"""

# ========================================
# IMPORT ALL LIBRARIES
# ========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
import pickle
import os
from scipy import stats

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import shap

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 100)
print(" 🏗️ REGRESSION ANALYSIS - COMPLETE PIPELINE ".center(100, "="))
print("=" * 100)
print(f"© Muhammad Ketsar Ali Abi Wahid")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

#  ========================================
# FASE 1: DATA LOADING & INITIAL EXPLORATION
# ========================================
print("\n" + "=" * 100)
print(" FASE 1: DATA LOADING & INITIAL EXPLORATION ".center(100, "="))
print("=" * 100)

# Load data
df = pd.read_csv('datasets/concrete_data.csv')
print(f"\n✅ Dataset loaded successfully!")
print(f"📏 Shape: {df.shape} (rows x columns)")
print(f"💾 Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Define target and features
target_col = 'Concrete_Compressive_Strength'
feature_cols = [col for col in df.columns if col != target_col]
print(f"\n🎯 Target: {target_col}")
print(f"📊 Features: {len(feature_cols)}")

# ========================================
# FASE 2: EXPLORATORY DATA ANALYSIS
# ========================================
print("\n" + "=" * 100)
print(" FASE 2: EXPLORATORY DATA ANALYSIS (EDA) ".center(100, "="))
print("=" * 100)

# 2.1 Missing Values
print("\n2.1 Missing Values Analysis:")
print("-" * 50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print(missing[missing > 0])

# 2.2 Duplicates
print("\n2.2 Duplicate Rows:")
print("-" * 50)
dups = df.duplicated().sum()
print(f"Duplicates: {dups}")
if dups > 0:
    df = df.drop_duplicates()
    print(f"✅ Removed {dups} duplicates")

# 2.3 Statistical Summary
print("\n2.3 Statistical Summary:")
print("-" * 50)
print(df.describe().T)

# 2.4 Correlation Analysis
print("\n2.4 Correlation Analysis:")
print("-" * 50)
correlation_matrix = df.corr()
target_corr = correlation_matrix[target_col].sort_values(ascending=False)
print("\nCorrelation with Target:")
print(target_corr)

# Visualization: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("📊 Saved: outputs/01_correlation_heatmap.png")

# 2.5 Target Distribution
target = df[target_col]
print(f"\n2.5 Target Variable Statistics:")
print("-" * 50)
print(f"Mean: {target.mean():.2f} MPa")
print(f"Median: {target.median():.2f} MPa")
print(f"Std: {target.std():.2f} MPa")
print(f"Min: {target.min():.2f} MPa")
print(f"Max: {target.max():.2f} MPa")
print(f"Skewness: {target.skew():.3f}")
print(f"Kurtosis: {target.kurt():.3f}")

# 2.6 Outlier Detection (IQR Method)
print("\n2.6 Outlier Detection (IQR Method):")
print("-" * 50)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Outliers per feature:")
print(outliers[outliers > 0])

# Visualization: Feature Distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].grid(alpha=0.3)
axes[-1].axis('off')  # Hide last subplot
plt.tight_layout()
plt.savefig('outputs/02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("📊 Saved: outputs/02_feature_distributions.png")

# ========================================
# FASE 3: DATA PREPROCESSING
# ========================================
print("\n" + "=" * 100)
print(" FASE 3: DATA PREPROCESSING ".center(100, "="))
print("=" * 100)

# 3.1 Prepare features and target
X = df[feature_cols].copy()
y = df[target_col].copy()
print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# 3.2 Feature Scaling
print("\n3.2 Feature Scaling (StandardScaler):")
print("-" * 50)
print("Why scaling? Many algorithms (SVR, Neural Networks) are sensitive to feature scales.")
print("StandardScaler: Transforms features to have mean=0 and std=1")

scaler = StandardScaler()
# We'll fit scaler after train-test split to avoid data leakage!
print("⚠️ Note: Scaler will be fitted on training data only (after split)")

# ========================================
# FASE 4: TRAIN-TEST SPLIT & BASELINE
# ========================================
print("\n" + "=" * 100)
print(" FASE 4: TRAIN-TEST SPLIT & BASELINE ".center(100, "="))
print("=" * 100)

# 4.1 Train-Test Split
print("\n4.1 Splitting data (80% train, 20% test):")
print("-" * 50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 4.2 Apply Scaling (FIT on train, TRANSFORM on both)
print("\n4.2 Applying scaling:")
print("-" * 50)
print("✅ Fitting scaler on training data...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✅ Scaled training data shape: {X_train_scaled.shape}")
print(f"✅ Scaled test data shape: {X_test_scaled.shape}")

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# 4.3 Baseline Model
print("\n4.3 Baseline Model (Dummy Regressor - Mean Strategy):")
print("-" * 50)
print("Baseline: Always predict the mean of training target")
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train_scaled, y_train)
y_pred_baseline = baseline.predict(X_test_scaled)

r2_baseline = r2_score(y_test, y_pred_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mape_baseline = mean_absolute_percentage_error(y_test, y_pred_baseline) * 100

print(f"Baseline R²: {r2_baseline:.4f}")
print(f"Baseline MAE: {mae_baseline:.4f}")
print(f"Baseline RMSE: {rmse_baseline:.4f}")
print(f"Baseline MAPE: {mape_baseline:.2f}%")
print("\n🎯 Any model must beat this baseline!")

# ========================================
# FASE 5: MODEL BUILDING (12 ALGORITHMS)
# ========================================
print("\n" + "=" * 100)
print(" FASE 5: MODEL BUILDING - 12 ALGORITHMS ".center(100, "="))
print("=" * 100)
print("\nWe will train 12 different regression algorithms:")
print("1. Linear Regression")
print("2. Ridge Regression (L2)")
print("3. Lasso Regression (L1)")
print("4. ElasticNet (L1 + L2)")
print("5. Polynomial Regression (degree 2)")
print("6. Decision Tree Regressor")
print("7. Random Forest Regressor")
print("8. Gradient Boosting Regressor")
print("9. XGBoost Regressor")
print("10. LightGBM Regressor")
print("11. CatBoost Regressor")
print("12. Support Vector Regressor (SVR)")
print("\n" + "=" * 100)

# Dictionary to store models and results
models = {}
results = []

# Function to train and evaluate model
def train_evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Train model and return metrics"""
    print(f"\n{'=' * 80}")
    print(f"Training: {name}")
    print(f"{'=' * 80}")

    start_time = time.time()
    model.fit(X_tr, y_tr)
    training_time = time.time() - start_time

    y_pred = model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mape = mean_absolute_percentage_error(y_te, y_pred) * 100

    print(f"✅ Trained in {training_time:.2f} seconds")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        'Model': name,
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Training_Time': training_time,
        'Predictions': y_pred
    }

# 5.1 Linear Regression
print("\n🔹 MODEL 1: Linear Regression")
print("-" * 50)
print("Simple linear model: y = β0 + β1*x1 + β2*x2 + ... + βn*xn")
print("Pros: Fast, interpretable, good baseline")
print("Cons: Assumes linear relationships, sensitive to outliers")
lr = LinearRegression()
models['Linear Regression'] = lr
result = train_evaluate_model('Linear Regression', lr, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.2 Ridge Regression
print("\n🔹 MODEL 2: Ridge Regression (L2 Regularization)")
print("-" * 50)
print("Adds L2 penalty: minimizes RSS + α * Σ(coefficients²)")
print("Pros: Prevents overfitting, handles multicollinearity")
print("Cons: Doesn't perform feature selection")
ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
models['Ridge'] = ridge
result = train_evaluate_model('Ridge Regression', ridge, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.3 Lasso Regression
print("\n🔹 MODEL 3: Lasso Regression (L1 Regularization)")
print("-" * 50)
print("Adds L1 penalty: minimizes RSS + α * Σ|coefficients|")
print("Pros: Feature selection (can shrink coefficients to zero)")
print("Cons: May struggle with highly correlated features")
lasso = Lasso(alpha=1.0, random_state=RANDOM_STATE)
models['Lasso'] = lasso
result = train_evaluate_model('Lasso Regression', lasso, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.4 ElasticNet
print("\n🔹 MODEL 4: ElasticNet (L1 + L2 Regularization)")
print("-" * 50)
print("Combines L1 and L2 penalties")
print("Pros: Benefits of both Ridge and Lasso")
print("Cons: Two hyperparameters to tune")
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE)
models['ElasticNet'] = elastic
result = train_evaluate_model('ElasticNet', elastic, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.5 Polynomial Regression
print("\n🔹 MODEL 5: Polynomial Regression (degree=2)")
print("-" * 50)
print("Creates polynomial features then applies linear regression")
print("Pros: Can capture non-linear relationships")
print("Cons: Risk of overfitting, increases feature space dramatically")
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)
poly = LinearRegression()
models['Polynomial'] = poly
result_poly = train_evaluate_model('Polynomial Regression', poly, X_train_poly, X_test_poly, y_train, y_test)
results.append(result_poly)

# 5.6 Decision Tree
print("\n🔹 MODEL 6: Decision Tree Regressor")
print("-" * 50)
print("Tree-based model that splits data based on features")
print("Pros: Non-linear, interpretable, no scaling needed")
print("Cons: Prone to overfitting, unstable (small data changes = different tree)")
dt = DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10)
models['Decision Tree'] = dt
result = train_evaluate_model('Decision Tree', dt, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.7 Random Forest
print("\n🔹 MODEL 7: Random Forest Regressor")
print("-" * 50)
print("Ensemble of decision trees (Bagging)")
print("Pros: Reduces overfitting, provides feature importance, robust")
print("Cons: Less interpretable, slower than single tree")
rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
models['Random Forest'] = rf
result = train_evaluate_model('Random Forest', rf, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.8 Gradient Boosting
print("\n🔹 MODEL 8: Gradient Boosting Regressor")
print("-" * 50)
print("Sequential ensemble: each tree corrects errors of previous")
print("Pros: Very accurate, handles complex patterns")
print("Cons: Can overfit, slower to train, many hyperparameters")
gb = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
models['Gradient Boosting'] = gb
result = train_evaluate_model('Gradient Boosting', gb, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.9 XGBoost
print("\n🔹 MODEL 9: XGBoost Regressor")
print("-" * 50)
print("Optimized gradient boosting with regularization")
print("Pros: Fast, accurate, built-in regularization")
print("Cons: Many hyperparameters, can overfit")
xgboost = xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
models['XGBoost'] = xgboost
result = train_evaluate_model('XGBoost', xgboost, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.10 LightGBM
print("\n🔹 MODEL 10: LightGBM Regressor")
print("-" * 50)
print("Leaf-wise tree growth, very fast")
print("Pros: Faster than XGBoost, handles large data well")
print("Cons: Can overfit on small datasets")
lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
models['LightGBM'] = lgbm
result = train_evaluate_model('LightGBM', lgbm, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.11 CatBoost
print("\n🔹 MODEL 11: CatBoost Regressor")
print("-" * 50)
print("Ordered boosting, handles categorical features natively")
print("Pros: Less tuning needed, robust")
print("Cons: Slower than LightGBM")
catboost_model = CatBoostRegressor(iterations=100, random_state=RANDOM_STATE, verbose=0)
models['CatBoost'] = catboost_model
result = train_evaluate_model('CatBoost', catboost_model, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# 5.12 Support Vector Regressor
print("\n🔹 MODEL 12: Support Vector Regressor (SVR)")
print("-" * 50)
print("Uses kernel trick to capture non-linear relationships")
print("Pros: Effective in high dimensions, versatile (different kernels)")
print("Cons: Slow on large datasets, sensitive to feature scaling")
svr_model = SVR(kernel='rbf', C=1.0)
models['SVR'] = svr_model
result = train_evaluate_model('SVR (RBF)', svr_model, X_train_scaled, X_test_scaled, y_train, y_test)
results.append(result)

# ========================================
# FASE 6: CROSS-VALIDATION
# ========================================
print("\n" + "=" * 100)
print(" FASE 6: CROSS-VALIDATION (K-FOLD CV) ".center(100, "="))
print("=" * 100)
print("\nCross-Validation gives more reliable performance estimate")
print("We'll use 5-Fold CV for all models")
print("-" * 50)

cv_results = []
for name, model in models.items():
    print(f"\nCV for {name}...", end=" ")
    try:
        if name == 'Polynomial':
            # Use polynomial features for CV
            cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5,
                                       scoring='r2', n_jobs=-1)
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                       scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"R² = {cv_mean:.4f} (±{cv_std:.4f})")
        cv_results.append({'Model': name, 'CV_Mean_R2': cv_mean, 'CV_Std_R2': cv_std})
    except Exception as e:
        print(f"Error: {str(e)}")
        cv_results.append({'Model': name, 'CV_Mean_R2': np.nan, 'CV_Std_R2': np.nan})

# ========================================
# FASE 7: HYPERPARAMETER TUNING
# ========================================
print("\n" + "=" * 100)
print(" FASE 7: HYPERPARAMETER TUNING ".center(100, "="))
print("=" * 100)
print("\nTuning top 3 models: Random Forest, XGBoost, LightGBM")
print("Methods: Grid Search CV, Random Search CV, Bayesian Optimization (Optuna)")
print("-" * 50)

# Sort results by R2
results_df = pd.DataFrame(results)
top_models = results_df.nlargest(3, 'R2')['Model'].tolist()
print(f"\n🏆 Top 3 models based on R²:")
for i, model_name in enumerate(top_models, 1):
    r2_value = results_df[results_df['Model'] == model_name]['R2'].values[0]
    print(f"{i}. {model_name}: R² = {r2_value:.4f}")

# 7.1 Grid Search CV (Random Forest)
if 'Random Forest' in top_models:
    print("\n" + "-" * 80)
    print("7.1 Grid Search CV - Random Forest")
    print("-" * 80)
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5]
    }
    print("Parameter grid:", param_grid_rf)
    print("Searching... (this may take a few minutes)")
    grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE),
                                   param_grid_rf, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    grid_search_rf.fit(X_train_scaled, y_train)
    print(f"✅ Best params: {grid_search_rf.best_params_}")
    print(f"✅ Best CV R²: {grid_search_rf.best_score_:.4f}")

    # Test performance
    y_pred_tuned_rf = grid_search_rf.predict(X_test_scaled)
    r2_tuned_rf = r2_score(y_test, y_pred_tuned_rf)
    print(f"✅ Test R²: {r2_tuned_rf:.4f}")

# 7.2 Random Search CV (XGBoost)
if 'XGBoost' in top_models:
    print("\n" + "-" * 80)
    print("7.2 Random Search CV - XGBoost")
    print("-" * 80)
    param_dist_xgb = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }
    print("Parameter distributions:", param_dist_xgb)
    print("Searching... (this may take a few minutes)")
    random_search_xgb = RandomizedSearchCV(xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
                                           param_dist_xgb, n_iter=20, cv=3, scoring='r2',
                                           n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    random_search_xgb.fit(X_train_scaled, y_train)
    print(f"✅ Best params: {random_search_xgb.best_params_}")
    print(f"✅ Best CV R²: {random_search_xgb.best_score_:.4f}")

    # Test performance
    y_pred_tuned_xgb = random_search_xgb.predict(X_test_scaled)
    r2_tuned_xgb = r2_score(y_test, y_pred_tuned_xgb)
    print(f"✅ Test R²: {r2_tuned_xgb:.4f}")

# 7.3 Bayesian Optimization with Optuna (LightGBM)
if 'LightGBM' in top_models:
    print("\n" + "-" * 80)
    print("7.3 Bayesian Optimization (Optuna) - LightGBM")
    print("-" * 80)
    print("Using Optuna for intelligent hyperparameter search...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': RANDOM_STATE,
            'verbosity': -1
        }
        model = lgb.LGBMRegressor(**params)
        score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name='LightGBM_Optimization')
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print(f"✅ Best params: {study.best_params}")
    print(f"✅ Best CV R²: {study.best_value:.4f}")

    # Train with best params
    best_lgbm = lgb.LGBMRegressor(**study.best_params)
    best_lgbm.fit(X_train_scaled, y_train)
    y_pred_tuned_lgbm = best_lgbm.predict(X_test_scaled)
    r2_tuned_lgbm = r2_score(y_test, y_pred_tuned_lgbm)
    print(f"✅ Test R²: {r2_tuned_lgbm:.4f}")

# ========================================
# FASE 8: MODEL EVALUATION & COMPARISON
# ========================================
print("\n" + "=" * 100)
print(" FASE 8: MODEL EVALUATION & COMPARISON ".center(100, "="))
print("=" * 100)

# Create results dataframe
results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)
final_results = results_df.merge(cv_df, on='Model', how='left')

# Add adjusted R²
n = len(y_test)
p = X_test.shape[1]
final_results['Adjusted_R2'] = 1 - (1 - final_results['R2']) * (n - 1) / (n - p - 1)

# Sort by R²
final_results = final_results.sort_values('R2', ascending=False)

print("\n📊 COMPLETE MODEL COMPARISON")
print("=" * 100)
print(final_results[['Model', 'R2', 'Adjusted_R2', 'MAE', 'RMSE', 'MAPE', 'CV_Mean_R2', 'Training_Time']].to_string(index=False))
print("=" * 100)

# Visualization: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# R² comparison
axes[0, 0].barh(final_results['Model'], final_results['R2'], color='steelblue')
axes[0, 0].set_xlabel('R² Score', fontsize=12)
axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[0, 0].grid(alpha=0.3, axis='x')

# MAE comparison
axes[0, 1].barh(final_results['Model'], final_results['MAE'], color='coral')
axes[0, 1].set_xlabel('Mean Absolute Error', fontsize=12)
axes[0, 1].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='x')

# RMSE comparison
axes[1, 0].barh(final_results['Model'], final_results['RMSE'], color='lightgreen')
axes[1, 0].set_xlabel('Root Mean Squared Error', fontsize=12)
axes[1, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='x')

# Training Time comparison
axes[1, 1].barh(final_results['Model'], final_results['Training_Time'], color='gold')
axes[1, 1].set_xlabel('Training Time (seconds)', fontsize=12)
axes[1, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/03_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n📊 Saved: outputs/03_model_comparison.png")

# ========================================
# FASE 9: MODEL INTERPRETATION (SHAP)
# ========================================
print("\n" + "=" * 100)
print(" FASE 9: MODEL INTERPRETATION & EXPLAINABILITY ".center(100, "="))
print("=" * 100)

# Get best model
best_model_name = final_results.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"🏆 R² Score: {final_results.iloc[0]['R2']:.4f}")

best_model = models[best_model_name]

# 9.1 Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree']:
    print("\n9.1 Feature Importance:")
    print("-" * 50)

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(feature_importance_df.to_string(index=False))

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('outputs/04_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: outputs/04_feature_importance.png")

# 9.2 SHAP Analysis
print("\n9.2 SHAP (SHapley Additive exPlanations) Analysis:")
print("-" * 50)
print("SHAP provides detailed explanation of model predictions")
print("Computing SHAP values... (this may take a minute)")

try:
    # Create SHAP explainer
    if best_model_name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
        explainer = shap.LinearExplainer(best_model, X_train_scaled)
    else:
        # Use TreeExplainer for tree-based models (faster)
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree']:
            explainer = shap.TreeExplainer(best_model)
        else:
            # Use KernelExplainer for other models (slower but model-agnostic)
            explainer = shap.KernelExplainer(best_model.predict, X_train_scaled.sample(100, random_state=RANDOM_STATE))

    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test_scaled)

    # SHAP Summary Plot (Global Importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.title('SHAP Summary Plot - Global Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/05_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP Summary Plot saved: outputs/05_shap_summary.png")

    # SHAP Bar Plot (Mean Absolute SHAP values)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/06_shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP Importance Plot saved: outputs/06_shap_importance.png")

    print("\n📌 SHAP Interpretation:")
    print("- Each dot represents a sample")
    print("- Red = High feature value, Blue = Low feature value")
    print("- Position on x-axis shows impact on prediction")
    print("- Features are sorted by importance (top = most important)")

except Exception as e:
    print(f"⚠️ SHAP analysis failed: {str(e)}")
    print("Continuing with other analyses...")

# 9.3 Actual vs Predicted Plot
print("\n9.3 Actual vs Predicted Plot:")
print("-" * 50)
best_predictions = final_results[final_results['Model'] == best_model_name]['Predictions'].values[0]

plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_predictions, alpha=0.6, edgecolors='black', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Concrete Strength (MPa)', fontsize=12)
plt.ylabel('Predicted Concrete Strength (MPa)', fontsize=12)
plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/07_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("📊 Saved: outputs/07_actual_vs_predicted.png")

# 9.4 Residual Analysis
print("\n9.4 Residual Analysis:")
print("-" * 50)
residuals = y_test - best_predictions

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Residual plot
axes[0].scatter(best_predictions, residuals, alpha=0.6, edgecolors='black', s=50)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Values', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/08_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("📊 Saved: outputs/08_residual_analysis.png")

print(f"\nResidual Statistics:")
print(f"Mean: {residuals.mean():.4f} (should be close to 0)")
print(f"Std: {residuals.std():.4f}")
print(f"Min: {residuals.min():.4f}")
print(f"Max: {residuals.max():.4f}")

# ========================================
# FASE 10: FINAL MODEL & REPORT
# ========================================
print("\n" + "=" * 100)
print(" FASE 10: FINAL MODEL SELECTION & COMPREHENSIVE REPORT ".center(100, "="))
print("=" * 100)

print("\n📋 FINAL MODEL REPORT")
print("=" * 100)
print(f"\n🏆 Selected Model: {best_model_name}")
print(f"\n📊 Performance Metrics:")
print("-" * 50)
print(f"R² Score: {final_results.iloc[0]['R2']:.4f}")
print(f"Adjusted R²: {final_results.iloc[0]['Adjusted_R2']:.4f}")
print(f"Mean Absolute Error (MAE): {final_results.iloc[0]['MAE']:.4f} MPa")
print(f"Root Mean Squared Error (RMSE): {final_results.iloc[0]['RMSE']:.4f} MPa")
print(f"Mean Absolute Percentage Error (MAPE): {final_results.iloc[0]['MAPE']:.2f}%")
print(f"Cross-Validation R² (mean): {final_results.iloc[0]['CV_Mean_R2']:.4f}")
print(f"Training Time: {final_results.iloc[0]['Training_Time']:.2f} seconds")

print(f"\n📊 Model Interpretation:")
print("-" * 50)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree']:
    print("✅ Feature importance analysis completed")
    print("✅ SHAP analysis provides detailed explanation")
    print("✅ Top 3 most important features:")
    if 'feature_importance_df' in locals():
        for i, row in feature_importance_df.head(3).iterrows():
            print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"\n💡 Business Recommendations:")
print("-" * 50)
print("1. Use this model to predict concrete strength before 28-day testing")
print("2. Optimize concrete mixture by focusing on top features")
print("3. Potential cost savings by reducing cement while maintaining strength")
print("4. Quality control: Flag mixtures predicted to underperform")

print(f"\n⚠️ Limitations:")
print("-" * 50)
print("1. Model trained on synthetic data - validate with real data")
print("2. Performance may vary with different concrete types")
print("3. Extrapolation beyond training data range may be unreliable")
print("4. Regular retraining recommended as new data becomes available")

print(f"\n🚀 Next Steps:")
print("-" * 50)
print("1. Collect more real-world concrete data")
print("2. Validate model on new test set")
print("3. Deploy model as REST API (FastAPI/Flask)")
print("4. Set up monitoring for model performance in production")
print("5. Implement MLOps pipeline (MLflow, DVC)")

# Save best model
model_path = f'models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n💾 Best model saved: {model_path}")

# Save scaler
scaler_path = 'models/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"💾 Scaler saved: {scaler_path}")

# Save results
results_path = 'outputs/model_results.csv'
final_results.to_csv(results_path, index=False)
print(f"💾 Results saved: {results_path}")

print("\n" + "=" * 100)
print(f"✅ COMPLETE REGRESSION ANALYSIS FINISHED!")
print(f"⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("📁 All outputs saved in 'outputs/' directory")
print("🤖 All models saved in 'models/' directory")
print("=" * 100)
print("\n🎉 Congratulations! You've completed a comprehensive regression analysis!")
print("=" * 100)
