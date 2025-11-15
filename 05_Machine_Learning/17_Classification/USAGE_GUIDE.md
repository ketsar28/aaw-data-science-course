# 📖 Module 17 - Usage Guide

**© Muhammad Ketsar Ali Abi Wahid**

---

## 🎯 Quick Start

Module 17 provides a complete classification pipeline with special focus on **imbalanced data handling**:

### **Option 1: Automated Script** (Recommended for Quick Results)
```bash
cd 05_Machine_Learning/17_Classification
python 17_classification_complete_script.py
```
- 🚀 Runs all 10 FASE automatically
- 📊 Trains 15+ classification algorithms
- ⚖️ Applies SMOTE, ADASYN for imbalanced data
- 💾 Saves all models and results
- ⏱️ Estimated time: 45-75 minutes

### **Option 2: Interactive Notebook** (Coming Soon)
```bash
jupyter notebook 17_classification_complete.ipynb
```
- 📝 Step-by-step explanations
- 🎨 Interactive visualizations
- 🧪 Experiment with different techniques

---

## 📂 File Structure

```
17_Classification/
├── README.md                               # Main documentation
├── QUICKSTART.md                           # 5-minute quick start
├── USAGE_GUIDE.md                          # This file
├── 17_classification_complete_script.py   # Automated script (2477 lines!)
├── datasets/
│   ├── sand_production_data.csv           # Main dataset (imbalanced)
│   └── create_sand_dataset.py             # Dataset generator
├── models/
│   └── (trained models will be saved here)
└── outputs/
    └── (12 plots and results will be saved here)
```

---

## 🔟 What's Covered (10 FASE + Bonus)

### **FASE 1: Data Loading & Initial Exploration**
- Load imbalanced dataset
- Check shape, dtypes, memory usage
- **CRITICAL**: Analyze class distribution and imbalance ratio

### **FASE 2: Exploratory Data Analysis (EDA)**
- Missing values and duplicate check
- Statistical summary
- Correlation analysis
- **Feature distributions BY CLASS** (key for classification!)
- Outlier detection (handled differently for classification)

### **FASE 3: Data Preprocessing**
- Feature scaling (StandardScaler)
- Preparation for modeling

### **FASE 4: Train-Test Split & Baseline**
- **STRATIFIED** split (maintains class distribution)
- Dummy Classifier baseline
- Why accuracy is USELESS for imbalanced data

### **FASE 5: Model Building (15 Algorithms)**
1. Logistic Regression (with class_weight='balanced')
2. K-Nearest Neighbors (KNN)
3. Naive Bayes (Gaussian)
4. Decision Tree (with class_weight='balanced')
5. Random Forest (with class_weight='balanced')
6. Gradient Boosting
7. XGBoost (with scale_pos_weight)
8. LightGBM (with is_unbalance=True)
9. CatBoost (with auto_class_weights='Balanced')
10. Support Vector Machine (SVM, with class_weight='balanced')
11. Linear Discriminant Analysis (LDA)
12. Quadratic Discriminant Analysis (QDA)
13. AdaBoost
14. Extra Trees (with class_weight='balanced')
15. **Balanced Random Forest** (specialized for imbalanced data!)

**Each model includes:**
- Detailed algorithm explanation
- When to use / when NOT to use
- Pros & Cons
- Imbalance handling strategy
- Full implementation & evaluation

### **FASE 5.5: Imbalanced Data Handling** ⚖️
**This is the CRITICAL difference from Module 16!**

Techniques demonstrated:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- Class weights (built into algorithms)
- Comparison: XGBoost vs XGBoost+SMOTE vs XGBoost+ADASYN

### **FASE 6: Cross-Validation**
- **STRATIFIED** K-Fold Cross-Validation (maintains class distribution)
- Applied to top 5 models
- Compare single split vs CV scores

### **FASE 7: Hyperparameter Tuning**
Three methods demonstrated:
- **Grid Search CV** (Random Forest)
- **Random Search CV** (XGBoost)
- **Bayesian Optimization with Optuna** (LightGBM)

### **FASE 8: Model Evaluation & Comparison**
**Metrics** (DIFFERENT from regression!):
- Accuracy (with caveats for imbalanced data)
- Precision (minimize false alarms)
- Recall (minimize missed cases - CRITICAL!)
- **F1-Score** (PRIMARY METRIC for imbalanced data)
- ROC-AUC (overall discriminative ability)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa

**Visualizations** (12 plots total):
1. Correlation heatmap
2. Target distribution (pie + bar chart)
3. Feature distributions by class
4. Model comparison (F1, Recall, Precision, ROC-AUC)
5. Training time comparison
6. Radar chart (top 5 models)
7. Confusion matrix (best model)
8. ROC curves (top 5 models)
9. Precision-Recall curves (top 5 models)
10. Feature importance
11. SHAP summary plot
12. SHAP importance plot

### **FASE 9: Model Interpretation**
- **Feature Importance** (best tree-based model)
- **SHAP Analysis** (SHapley Additive exPlanations)
  - Summary plots showing feature contributions
  - Importance plots
  - Detailed interpretation guide

### **FASE 10: Final Model Selection & Report**
- Multi-criteria model selection (F1, Recall, ROC-AUC, training time)
- Classification report (precision, recall, F1 per class)
- **Business impact analysis** (cost of false negatives vs false positives)
- Model & scaler saving
- Usage example
- Comprehensive final report

---

## 📊 Expected Outputs

After running the script, you'll get:

### **Visualizations** (in `outputs/` folder):
1. `01_correlation_heatmap.png` - Feature correlations
2. `02_target_distribution.png` - Class imbalance visualization
3. `03_feature_distributions.png` - All features by class
4. `04_model_comparison.png` - F1, Precision, Recall, ROC-AUC
5. `05_training_time.png` - Training time per model
6. `06_radar_chart.png` - Top 5 models radar chart
7. `07_confusion_matrix.png` - Best model confusion matrix
8. `08_roc_curves.png` - ROC curves comparison
9. `09_precision_recall_curves.png` - PR curves comparison
10. `10_feature_importance.png` - Feature importance
11. `11_shap_summary.png` - SHAP summary plot
12. `12_shap_importance.png` - SHAP importance plot

### **Models** (in `models/` folder):
- `best_model_*.pkl` - Best performing model
- `scaler.pkl` - Fitted StandardScaler

### **Results** (in `outputs/` folder):
- `model_results.csv` - Complete comparison table (15+ models)

---

## 🎯 Learning Outcomes

After completing this module, you will be able to:

✅ Understand the imbalanced data problem and why it matters

✅ Apply appropriate metrics for imbalanced classification (F1-Score, Recall)

✅ Implement 15 different classification algorithms

✅ Handle imbalanced data with SMOTE, ADASYN, and class weights

✅ Use stratified splitting and cross-validation correctly

✅ Interpret confusion matrix for business insights

✅ Understand trade-off between Precision and Recall

✅ Apply hyperparameter tuning for classification

✅ Interpret models with SHAP for classification

✅ Select best model based on business requirements

✅ Save and deploy classification models

---

## 💡 Tips for Success

### **For Beginners:**
1. ⏱️ **Understand Metrics First** - Classification metrics are VERY different from regression!
2. 📊 **Focus on F1-Score** - Not accuracy! Accuracy is misleading for imbalanced data.
3. 🔍 **Study Confusion Matrix** - It tells the real story (TP, TN, FP, FN)
4. ⚖️ **Learn Imbalance Handling** - This is critical for real-world problems
5. 📚 **Read All Explanations** - Each algorithm has specific use cases

### **For Advanced Learners:**
1. 🎯 **Optimize Threshold** - Don't stick to 0.5! Adjust based on business cost
2. 🔬 **Try Ensemble of Resampling** - Combine SMOTE with different algorithms
3. 🏗️ **Build Custom Metrics** - Create cost-sensitive metrics for your use case
4. 🚀 **Calibrate Probabilities** - Use CalibratedClassifierCV for better probabilities
5. 📊 **Experiment with Thresholds** - Plot Precision-Recall curve, find optimal point

---

## ⚠️ Common Issues & Solutions

### **Issue 1: "Module not found: imbalanced-learn"**
**Solution:**
```bash
pip install imbalanced-learn
# OR
pip install -r requirements.txt
```

### **Issue 2: Script runs very slow**
**Solutions:**
- Reduce number of Optuna trials (30 → 15)
- Reduce number of CV folds (5 → 3)
- Reduce n_estimators for ensemble models (100 → 50)
- Skip Grid Search (most time-consuming)
- Comment out SHAP analysis if needed

### **Issue 3: All models get ~85% accuracy but 0% F1-Score**
**Solutions:**
- This means models are predicting all samples as majority class!
- MUST use imbalance handling (SMOTE, class weights, etc.)
- Check if you're using F1-Score for optimization, not accuracy
- Verify stratified splitting is used

### **Issue 4: SMOTE creates too many synthetic samples**
**Solutions:**
- Use SMOTE with sampling_strategy parameter: `SMOTE(sampling_strategy=0.5)`
- Try ADASYN instead (more conservative)
- Use combination methods: SMOTEENN or SMOTETomek
- Consider under-sampling majority class instead

### **Issue 5: Model has high precision but low recall (or vice versa)**
**Solutions:**
- This is the Precision-Recall trade-off!
- For sand production (safety-critical): Prioritize RECALL (don't miss cases)
- For spam detection (user experience): Prioritize PRECISION (avoid false alarms)
- Adjust threshold to balance based on business needs
- Use F1-Score for balanced optimization

---

## 🔄 Customization

Want to use your own imbalanced dataset? Follow these steps:

### **Step 1: Prepare your CSV file**
- Must have numerical features
- Binary target column (0 and 1)
- Check your imbalance ratio: `target.value_counts()`
- Imbalance ratio > 1.5:1 → Use imbalance handling techniques!

### **Step 2: Update script**
```python
# Change dataset path
df = pd.read_csv('path/to/your/dataset.csv')

# Update target column name
# Find this in FASE 1:
# y = df['Sand_Production']  # Change 'Sand_Production' to your target column
```

### **Step 3: Adjust imbalance handling**
```python
# In SMOTE section, adjust sampling strategy if needed
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.7)  # 0.7 = 70% of majority class

# Or use different technique:
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=RANDOM_STATE)
```

### **Step 4: Run analysis**
- Everything else should work automatically!
- Check outputs for insights specific to your data

---

## 📊 Understanding Metrics (Simplified!)

### **Confusion Matrix:**
```
                Predicted
              No Sand  Sand
Actual  No      850     100    = 950 total (majority class)
        Sand     30      20    =  50 total (minority class)
```

### **From this matrix:**

**Accuracy** = (850 + 20) / 1000 = **87%**
- Looks good, but misleading!
- Only catches 20 out of 50 sand cases (40% recall!)

**Precision** = 20 / (20 + 100) = **16.7%**
- "Of all predicted sand, only 16.7% actually had sand"
- Many false alarms!

**Recall** = 20 / (20 + 30) = **40%**
- "Of all actual sand cases, we caught only 40%"
- Missed 30 critical cases! (BAD for safety!)

**F1-Score** = 2 * (0.167 * 0.40) / (0.167 + 0.40) = **23.5%**
- Very low! This model is NOT good despite 87% accuracy!

### **Better Model After SMOTE:**
```
                Predicted
              No Sand  Sand
Actual  No      830     120    = 950 total
        Sand     10      40    =  50 total
```

**Recall** = 40 / (40 + 10) = **80%**
- Much better! Catches 80% of sand cases!

**Precision** = 40 / (40 + 120) = **25%**
- Still has false alarms, but acceptable trade-off

**F1-Score** = 2 * (0.25 * 0.80) / (0.25 + 0.80) = **38%**
- Much better than 23.5%!

---

## 🎓 Key Differences: Module 16 vs Module 17

| Aspect | Module 16 (Regression) | Module 17 (Classification) |
|--------|----------------------|------------------------------|
| **Problem Type** | Predict continuous values | Predict categories/classes |
| **Output** | Numbers (e.g., 45.3 MPa) | Classes (e.g., 0 or 1) |
| **Main Challenge** | Outliers, scaling | **Imbalanced data** ⚖️ |
| **Primary Metric** | R² | **F1-Score** |
| **Secondary Metrics** | MAE, RMSE, MAPE | Precision, Recall, ROC-AUC |
| **Splitting** | Random split | **Stratified split** |
| **Cross-Validation** | K-Fold | **Stratified K-Fold** |
| **Special Techniques** | Polynomial features | **SMOTE, ADASYN, class weights** |
| **Evaluation Plot** | Actual vs Predicted | **Confusion Matrix, ROC curve** |
| **Model Selection** | Highest R², lowest RMSE | **Highest F1, Recall (business-driven)** |
| **Algorithms** | 12 regression algorithms | 15 classification algorithms |
| **Imbalance Handling** | Not applicable | **CRITICAL! FASE 5.5** |

---

## 📞 Support & Questions

If you encounter issues or have questions:

1. 📖 **Read README.md** - Comprehensive 300+ line guide
2. 🔍 **Check QUICKSTART.md** - Quick 5-minute overview
3. 📚 **Study code comments** - 2477 lines with detailed explanations
4. 💬 **Refer to classification metrics guide** - In README.md
5. 🔧 **Check troubleshooting** - Common issues section above

---

## 🎓 Next Steps

After mastering Module 17:

1. ✅ **Compare with Module 16** - Understand regression vs classification
2. ✅ **Module 18**: Advanced Ensemble Methods
3. ✅ **Module 19**: Unsupervised Learning (Clustering, Dimensionality Reduction)
4. ✅ **Module 23**: Deep Learning for Classification
5. ✅ **Module 26**: Advanced Training Techniques
6. ✅ **Module 27**: Deep Dive into Model Explainability
7. ✅ **Module 28**: Experiment Tracking with MLflow
8. ✅ **Module 30**: Deploy classification model as API

---

## 🚀 Real-World Applications

This classification pipeline can be applied to:

### **Safety-Critical (Prioritize RECALL):**
- Disease detection (don't miss cancer cases!)
- Fraud detection (catch all fraud attempts)
- Equipment failure prediction (prevent downtime)
- **Sand production** (avoid equipment damage)
- Credit card fraud
- Churn prediction (retain customers)

### **User Experience (Balance Precision-Recall):**
- Spam detection (balance false alarms vs spam)
- Recommendation systems
- Customer segmentation
- Sentiment analysis

### **Cost-Sensitive (Optimize Threshold):**
- Insurance claim approval
- Loan default prediction
- Marketing campaign targeting

---

## 📜 License

**© Muhammad Ketsar Ali Abi Wahid**

Part of "Data Science Zero to Hero: Complete MLOps & Production ML Engineering" course.

---

**Happy Learning! 🚀**

> "In classification, it's not about being right most of the time. It's about being right when it matters most! Understanding your metrics is the key to success."

---

## 📌 Quick Reference Card

**Most Important Concepts:**
1. ⚖️ **Imbalanced Data** - Majority class >> Minority class
2. 📊 **F1-Score** - Harmonic mean of Precision & Recall (PRIMARY METRIC)
3. 🎯 **Recall** - Don't miss positive cases (critical for safety)
4. 🔍 **Precision** - Avoid false alarms (important for UX)
5. ⚡ **SMOTE** - Create synthetic minority samples
6. 🌳 **Stratified Split** - Maintain class distribution
7. 📈 **ROC-AUC** - Overall discriminative ability
8. 🎯 **Confusion Matrix** - Shows the real story (TP, TN, FP, FN)

**When to use what:**
- **High Recall**: Medical diagnosis, fraud detection, safety systems
- **High Precision**: Spam detection, recommendation systems
- **Balanced (F1)**: Most real-world problems
- **Threshold tuning**: When costs of FP and FN are different

---

**© Muhammad Ketsar Ali Abi Wahid**
**Module 17 - Classification Complete**
