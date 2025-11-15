#!/usr/bin/env python3
"""
Build Module 01 - Introduction to Data Science SOLUTIONS
Complete solutions untuk semua 15 exercises
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =============================================================================
# HEADER
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# ✅ Module 01: Introduction to Data Science - SOLUTIONS

---

## 📚 Tentang Solutions Ini

Solutions ini berisi **jawaban lengkap** dengan:
- ✅ **Jawaban final** (answer key)
- ✅ **Detailed explanation** (mengapa jawaban ini benar)
- ✅ **Alternative approaches** (cara lain yang juga valid)
- ✅ **Common mistakes** (kesalahan yang sering terjadi)
- ✅ **Key takeaways** (pelajaran penting)

---

## 🎯 Cara Menggunakan Solutions:

1. **TRY FIRST**: Kerjakan exercise dulu tanpa lihat solutions!
2. **COMPARE**: Setelah selesai, bandingkan jawaban Anda
3. **UNDERSTAND**: Baca explanation untuk deep understanding
4. **LEARN**: Dari mistakes - it's normal!
5. **PRACTICE**: Jika salah, retry exercise dengan understanding baru

---

**Remember**: *The goal is LEARNING, not just getting correct answers!*

---"""))

# =============================================================================
# SOLUTION 1
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🟢 Solution 1: Data Science Definition

---

## ✅ ANSWER KEY:

### Tabel Lengkap:

| Tipe | Pertanyaan | Contoh | Teknik |
|------|-----------|--------|---------|
| **Descriptive** | **Apa yang terjadi?** | Total penjualan bulan ini | **Aggregation, Visualization, Summary Statistics** |
| **Diagnostic** | Mengapa terjadi? | **Mengapa penjualan turun 15% di Q3?** | Correlation Analysis, **Drill-down, Root Cause Analysis** |
| **Predictive** | Apa yang akan terjadi? | Prediksi penjualan bulan depan | **Machine Learning, Forecasting, Statistical Modeling** |
| **Prescriptive** | **Apa yang harus dilakukan?** | Rekomendasi strategi marketing | Optimization, **Simulation, Decision Analysis** |

---

### Contoh Tambahan:

**1. Descriptive**:
- Average customer age in our database
- Top 10 selling products this quarter
- Total website traffic per day

**2. Diagnostic**:
- Why did conversion rate drop after website redesign?
- What caused spike in customer complaints last month?
- Why is user engagement higher on mobile than desktop?

**3. Predictive**:
- Which customers are likely to churn next month?
- Forecast stock price for next week
- Predict equipment failure before it happens

**4. Prescriptive**:
- Optimal pricing strategy to maximize revenue
- Best route for delivery trucks (minimize time/cost)
- Recommended product bundles for cross-selling

---

## 📖 EXPLANATION:

**Descriptive** = Looking at **past/present**:
- Answers: "What happened?"
- Tools: Simple aggregations (SUM, AVG, COUNT)
- Output: Reports, dashboards, KPIs

**Diagnostic** = Understanding **why**:
- Answers: "Why did it happen?"
- Tools: Deeper analysis (correlations, comparisons, drill-downs)
- Output: Root cause analysis, insights

**Predictive** = Forecasting **future**:
- Answers: "What will happen?"
- Tools: Machine Learning, statistical models
- Output: Predictions, probabilities, forecasts

**Prescriptive** = Recommending **actions**:
- Answers: "What should we do?"
- Tools: Optimization algorithms, simulation
- Output: Recommendations, optimal strategies

**Progression**: Descriptive → Diagnostic → Predictive → Prescriptive
(Increasing complexity & value!)

---

## ⚠️ COMMON MISTAKES:

1. ❌ Confusing Predictive with Prescriptive
   - Predictive = "What will happen"
   - Prescriptive = "What to DO about it"

2. ❌ Thinking Descriptive is simple/not valuable
   - It's foundation! Can't predict without understanding past

---

## 🎯 KEY TAKEAWAY:

All 4 types important! Real projects often use multiple types together.

---"""))

# =============================================================================
# SOLUTION 2
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🟢 Solution 2: AI vs ML vs DL Hierarchy

---

## ✅ ANSWER KEY:

| Application | Category | Reasoning |
|------------|----------|-----------|
| 1. Netflix recommendation | **ML** | Learns from viewing history to recommend (collaborative filtering, matrix factorization) |
| 2. Siri voice assistant | **AI (includes DL)** | Combines rule-based AI + DL (speech recognition) + NLP |
| 3. iPhone Face ID | **DL** | Uses Convolutional Neural Networks (CNNs) untuk face recognition |
| 4. Rule-based chess | **AI (not ML)** | Uses minimax algorithm + heuristics, tidak belajar dari data |
| 5. Google Translate | **DL** | Uses Transformer models (neural networks) - very complex |
| 6. Gmail spam filter | **ML** | Learns from labeled emails (spam/not spam) - likely Naive Bayes atau similar |
| 7. ChatGPT | **DL** | Large Language Model (LLM) - Transformer architecture, billions parameters |
| 8. Alexa smart speaker | **AI (includes DL)** | Like Siri - combo of AI techniques including DL untuk speech |

---

## 📖 DETAILED EXPLANATION:

### **AI Applications** (Broad):
- **Siri, Alexa**: Complex systems dengan multiple AI techniques
- **Rule-based chess**: Classical AI (no learning!)
- Note: All ML and DL are AI, tapi not all AI is ML!

### **ML Applications** (Learning from Data):
- **Netflix**: Learns patterns dari millions of users
  - Feature: User watch history, ratings
  - Algorithm: Collaborative filtering, matrix factorization
  - NOT DL: Can use simpler ML algorithms effectively

- **Gmail spam**: Learns from labeled emails
  - Features: Words, sender, links, etc.
  - Algorithm: Naive Bayes, Logistic Regression, etc.
  - Continuous learning: Gets better as you mark spam

### **DL Applications** (Neural Networks):
- **Face ID**: CNNs dengan many layers
  - Why DL?: Face recognition very complex (angles, lighting, aging)
  - Automatic feature extraction (edges → nose/eyes → full face)
  - Needs: Millions of face images untuk training

- **Google Translate**: Transformer models (state-of-the-art DL)
  - Why DL?: Language complexity, context, idioms
  - Billions of parameters
  - Trained on massive parallel text corpus

- **ChatGPT**: GPT (Generative Pre-trained Transformer)
  - DL architecture: Transformer
  - Scale: 175B parameters (GPT-3)
  - Training: Massive internet text data

---

## 🎓 CLASSIFICATION LOGIC:

**Decision Tree**:
```
Is it a computer doing "smart" task?
  ├─ Yes → AI
  │
  Does it LEARN from data?
    ├─ Yes → ML
    │   │
    │   Does it use NEURAL NETWORKS with many layers?
    │     ├─ Yes → DL
    │     └─ No → Traditional ML
    │
    └─ No → Rule-based AI (not ML)
```

---

## ⚠️ COMMON MISTAKES:

1. ❌ **"All AI is ML"** - NO!
   - Rule-based chess AI doesn't learn
   - Expert systems (if-then rules) are AI but not ML

2. ❌ **"ML and DL are different"** - Actually:
   - DL is SUBSET of ML
   - DL = ML using deep neural networks

3. ❌ **"Simple app = not AI"** - NO!
   - Spam filter seems simple but uses ML!
   - Don't judge by user interface

---

## 🎯 KEY TAKEAWAY:

**Hierarchy matters**:
- AI ⊃ ML ⊃ DL (AI includes ML includes DL)
- Classification depends on TECHNIQUE, not problem domain
- Same problem can be solved dengan different levels (cat/dog classifier example)

---"""))

# Continue with Solution 3...
cells.append(nbf.v4.new_markdown_cell("""# 🟢 Solution 3: Data Science Roles

---

## ✅ ANSWER KEY:

| Responsibility | Role | Reasoning |
|---------------|------|-----------|
| A. Build production ML APIs with FastAPI | **ML Engineer** | Production deployment with software engineering focus |
| B. Create Tableau dashboards for monthly sales | **Data Analyst** | Business intelligence & reporting |
| C. Build XGBoost model to predict churn | **Data Scientist** | Modeling & predictive analytics |
| D. Optimize database query from 5min to 10sec | **Data Engineer** | Database performance & infrastructure |
| E. Deploy model with Docker to AWS | **ML Engineer** | Model deployment & cloud infrastructure |
| F. Run A/B test for new website feature | **Data Scientist** | Experimentation & statistical analysis |

---

## 📖 DETAILED REASONING:

### **A → ML Engineer**:
- **FastAPI**: Modern Python framework untuk REST APIs
- **Production-ready**: Key word - ML Engineer specializes in **deployment**
- Not Data Scientist: DS builds model, MLE deploys it
- Skillset: Software engineering + ML

### **B → Data Analyst**:
- **Tableau**: BI tool untuk dashboards
- **Monthly sales report**: Descriptive analytics
- **Business focus**: Communicating insights ke stakeholders
- **No modeling** involved: Just reporting/visualization

### **C → Data Scientist**:
- **XGBoost**: Advanced ML algorithm
- **Predict churn**: Predictive analytics (DS core competency)
- **Modeling**: Building & tuning model adalah DS job
- Could be MLE too, but **initial model building** usually DS

### **D → Data Engineer**:
- **Database query**: Infrastructure work
- **Performance optimization**: DE specialty
- **5 min → 10 sec**: Typical DE impact!
- Tools: SQL optimization, indexing, partitioning

### **E → ML Engineer**:
- **Docker**: Containerization (MLOps)
- **AWS deployment**: Cloud infrastructure
- **Production**: Again, deployment is MLE domain
- Collaboration: DS builds model → MLE deploys it

### **F → Data Scientist**:
- **A/B testing**: Experimentation
- **Statistical analysis**: Determine significance
- **Hypothesis testing**: DS core skill
- Design experiment → Analyze results → Recommend

---

## 🔍 DISTINGUISHING ROLES:

### **Keyword Indicators**:

**Data Analyst**:
- Keywords: Dashboard, report, Tableau/Power BI, SQL, Excel
- Focus: **Past & present** (descriptive/diagnostic)

**Data Scientist**:
- Keywords: Model, predict, ML algorithms, A/B test, statistics
- Focus: **Future** (predictive/prescriptive) + experimentation

**ML Engineer**:
- Keywords: Deploy, production, API, Docker, AWS, performance, scale
- Focus: **Engineering** ML systems

**Data Engineer**:
- Keywords: Pipeline, database, ETL, performance, infrastructure, Spark
- Focus: **Data infrastructure**

---

## ⚠️ COMMON MISTAKES:

1. ❌ **Thinking DS handles ALL modeling tasks**:
   - Reality: MLE also works with models (for deployment)
   - Difference: DS focuses on accuracy/experimentation, MLE on production-readiness

2. ❌ **Confusing DE with DBA** (Database Administrator):
   - DE: Builds pipelines, data architecture (development)
   - DBA: Maintains databases (operations)

3. ❌ **Thinking roles are rigid**:
   - Reality: Roles overlap! Especially at startups
   - Full-stack DS might do everything

---

## 🎯 KEY TAKEAWAY:

**Team collaboration**:
- DA → DS → MLE → DE often work on same project!
- DA: Understand business problem
- DS: Build model
- MLE: Deploy to production
- DE: Ensure data pipeline works

**Career path**: Often progress through roles (DA → DS → MLE)

---"""))

# Add remaining solutions 4-15
cells.append(nbf.v4.new_markdown_cell("""# 🟢 Solution 4: CRISP-DM Phases

## ✅ ANSWER KEY:

**Correct Order**:
1. **B** → Phase 1: Business Understanding
2. **E** → Phase 2: Data Understanding
3. **D** → Phase 3: Data Preparation
4. **A** → Phase 4: Modeling
5. **C** → Phase 5: Evaluation
6. **F** → Phase 6: Deployment

## 📖 EXPLANATION:
Always follow CRISP-DM: Business Understanding → Data Understanding → Preparation → Modeling → Evaluation → Deployment (then iterate!)

## 🎯 KEY TAKEAWAY:
Can't skip phases! Each builds on previous. Most time spent on Data Preparation (50-70%).

---"""))

cells.append(nbf.v4.new_markdown_cell("""# 🟢 Solution 5: Python Libraries

## ✅ ANSWER KEY:

1. **Pandas** → **B** (Load CSV and filter rows)
2. **Matplotlib** → **C** (Create histogram)
3. **Scikit-learn** → **D** (Train Random Forest)
4. **TensorFlow** → **A** (Build CNN for images)
5. **Seaborn** → **F** (Create correlation heatmap)
6. **NumPy** → **E** (Matrix multiplication)

## 📖 EXPLANATION:
- **Pandas**: Data manipulation (CSV, filtering, groupby)
- **NumPy**: Numerical operations (arrays, linear algebra)
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical viz (built on Matplotlib, prettier)
- **Scikit-learn**: Traditional ML
- **TensorFlow**: Deep Learning

## 🎯 KEY TAKEAWAY:
Learn in order: NumPy → Pandas → Matplotlib → Scikit-learn → (later) TensorFlow

---"""))

cells.append(nbf.v4.new_markdown_cell("""# 🟡 Solution 6: When to Use Which Algorithm?

## ✅ ANSWER KEY:

### Comparison Table:

| Aspect | Rule-Based | Traditional ML | Deep Learning (CNN) |
|--------|-----------|----------------|---------------------|
| **Pros** | ✅ Simple, fast<br>✅ Interpretable<br>✅ No data needed | ✅ Learns patterns<br>✅ Better than rules<br>✅ Interpretable | ✅ **Very accurate (>95%)**<br>✅ Auto features<br>✅ Handles complexity |
| **Cons** | ❌ Rigid rules<br>❌ Many exceptions<br>❌ Hard to maintain | ❌ Manual features<br>❌ Limited accuracy (~85%)<br>❌ Need feature engineering | ❌ Need LOTS of data<br>❌ Need GPU<br>❌ Black box |

### Recommendation for 100K Images:
**Use Deep Learning (CNN)** ✅

**Reasoning**:
1. **Data size**: 100K images is sufficient for DL (ideal >10K)
2. **Complexity**: Cat vs dog varies widely (breeds, angles, lighting) - perfect for DL
3. **Accuracy**: DL will achieve 95%+ vs 80-85% for traditional ML
4. **Automatic features**: No need manual feature extraction

**Implementation**: Use Transfer Learning (pre-trained ResNet/VGG) to reduce training time!

### Alternative Scenarios:

**Use Rule-Based when**:
- Simple, clear rules exist
- Very few data points
- Need 100% interpretability
- Example: Checking if image is landscape/portrait (just compare width vs height)

**Use Traditional ML when**:
- Limited data (<1000 images)
- Need interpretability
- Limited compute (no GPU)
- Example: Classify 500 medical images where you can't explain why → Use Decision Tree with manual features

## 🎯 KEY TAKEAWAY:
Start simple, but with 100K images + accuracy goal, DL is right choice!

---"""))

cells.append(nbf.v4.new_markdown_cell("""# 🟡 Solution 7: Career Path Design

## ✅ ANSWER KEY:

### 5-Year Learning Path to Senior Data Scientist:

| Year | Job Title | Skills to Learn | Tools/Frameworks | Projects | Time/Week |
|------|-----------|----------------|------------------|----------|-----------|
| **Year 1** | **Junior Data Analyst** | - SQL (advanced)<br>- Python (Pandas, NumPy)<br>- Statistics basics<br>- Data viz | - PostgreSQL<br>- Pandas, Matplotlib<br>- Tableau/Power BI<br>- Git | 1. Sales analysis dashboard<br>2. Customer segmentation<br>3. Kaggle competitions | 15-20 hrs |
| **Year 2** | **Data Analyst** | - Scikit-learn<br>- Statistics (intermediate)<br>- Feature engineering<br>- ML basics | - Scikit-learn<br>- Seaborn<br>- Jupyter<br>- VS Code | 1. Churn prediction<br>2. Price optimization<br>3. Time series forecasting | 15 hrs |
| **Year 3** | **Junior Data Scientist** | - Advanced ML<br>- Deep learning basics<br>- A/B testing<br>- Model deployment | - XGBoost, LightGBM<br>- TensorFlow/PyTorch<br>- MLflow<br>- Docker basics | 1. Recommendation system<br>2. NLP sentiment analysis<br>3. Deploy model API | 10-15 hrs |
| **Year 4** | **Data Scientist** | - Deep learning advanced<br>- MLOps<br>- Cloud platforms<br>- System design | - PyTorch advanced<br>- AWS/GCP<br>- Airflow<br>- FastAPI | 1. Image classification<br>2. End-to-end ML pipeline<br>3. Production model | 10 hrs |
| **Year 5** | **Senior Data Scientist** | - Leadership<br>- Business strategy<br>- Team mentoring<br>- Architecture design | - All previous<br>- Kubernetes<br>- Advanced MLOps | 1. Lead DS project<br>2. Mentor juniors<br>3. Technical presentations | 5-10 hrs |

### Additional Recommendations:

**Certifications** (optional but helpful):
- Year 2: Google Data Analytics Certificate
- Year 3: AWS ML Specialty
- Year 4: TensorFlow Developer Certificate

**Networking**:
- Join local DS meetups (Year 1+)
- Present at conferences (Year 3+)
- Contribute to open source (Year 2+)

**Salary Progression** (Indonesia, rough estimate):
- Year 1: 8-12 juta/month
- Year 2: 12-18 juta/month
- Year 3: 18-25 juta/month
- Year 4: 25-35 juta/month
- Year 5: 35-50 juta/month

## 🎯 KEY TAKEAWAY:
Start with foundations (SQL, Python, Stats), progressively add complexity. Projects > Certificates!

---"""))

# Add solutions 8-15 (concise versions)
cells.append(nbf.v4.new_markdown_cell("""# 🟡 Solution 8: CRISP-DM Application - Hospital Readmission

## ✅ COMPLETE PROJECT PLAN:

### Phase 1: Business Understanding (Week 1)
**Goals**: Reduce readmission from 25% → 15% (save costs, improve patient care)
**Success Metrics**:
- Model Recall ≥ 80% (catch 80% of readmission cases)
- Precision ≥ 60% (avoid false alarms)
- Deploy within 6 months

**Activities**:
1. Interview doctors, nurses, admin staff
2. Understand cost of readmission ($10K+ per case)
3. Define intervention plan (what do when predict high risk?)
4. Regulatory/HIPAA compliance check

**Output**: Project charter, stakeholder buy-in, timeline

### Phase 2: Data Understanding (Weeks 2-3)
**Data Sources**: EHR (Electronic Health Records), lab results, medications, demographics

**Activities**:
1. Assess data quality (completeness, accuracy)
2. Understand 30-day readmission patterns
3. Identify high-risk patient groups
4. Check for data biases

**Output**: Data quality report, initial insights

### Phase 3: Data Preparation (Weeks 4-7)
**Activities**:
1. Handle missing values (medical data often incomplete!)
2. Feature engineering: diagnoses codes, medication counts, comorbidities
3. Remove PII (patient privacy!)
4. Handle class imbalance (likely <25% readmitted)
5. Train/val/test split (temporal split - use older data for train!)

**Output**: Clean, model-ready dataset

### Phase 4: Modeling (Weeks 8-10)
**Algorithms to try**: Logistic Regression (baseline, interpretable!), Random Forest, XGBoost

**Activities**:
1. Build interpretable baseline (Logistic Regression)
2. Try ensemble methods (better accuracy)
3. Feature importance analysis (medical staff need to understand!)
4. Handle class imbalance (SMOTE, class weights)

**Critical**: Model MUST be interpretable for medical use!

**Output**: Best model with feature importance

### Phase 5: Evaluation (Week 11)
**Technical Metrics**: Recall (priority!), Precision, AUC-ROC

**Business Evaluation**:
1. Cost-benefit analysis (intervention cost vs readmission cost)
2. Pilot test with 100 patients
3. Doctor feedback (do predictions make sense?)
4. Bias check (fair across demographics?)

**Ethical Checks**:
- No discrimination by race/gender/insurance
- Explainable predictions (why patient high-risk?)

**Output**: Go/No-go decision

### Phase 6: Deployment (Weeks 12-16)
**Implementation**: Integrate into EHR system, alert discharge team for high-risk patients

**Activities**:
1. Build prediction API
2. Create dashboard for care coordinators
3. Train staff on using system
4. A/B test (50% patients use system, 50% control)
5. Monitor: prediction accuracy, actual impact on readmission

**Success Criteria**: Readmission rate drops to <20% within 3 months

**Time Estimate**: 16 weeks total

## 🎯 KEY CONSIDERATIONS:
- **Interpretability critical** in healthcare!
- **Privacy (HIPAA)** must be maintained
- **Bias** can literally harm patients - careful!
- **Validation** with medical experts essential

---"""))

cells.append(nbf.v4.new_markdown_cell("""# 🟡 Solutions 9-12 (Concise Versions)

---

## 🟡 Solution 9: Tech Stack for Recommendation System

### RECOMMENDED STACK:

1. **Language**: **Python** (rich ML ecosystem, team knows it)
2. **ML Framework**: **Scikit-learn** (simple models first), **TensorFlow** (if need neural collaborative filtering)
3. **Model Serving**: **FastAPI** (fast, modern, auto docs)
4. **Database**: **PostgreSQL** (user data) + **Redis** (caching for <100ms latency)
5. **Cloud**: **GCP** (BigQuery for analytics, cost-effective)
6. **MLOps**: **MLflow** (free, experiment tracking)
7. **Dev Environment**: **VS Code** + **Jupyter**

**Why this stack?**: Cost-effective, scales to 1M users, open-source, team can manage with 3 people.

---

## 🟡 Solution 10: Ethical Dilemma - Hiring Model Bias

### ETHICAL ISSUES:
1. **Gender discrimination** - violates anti-discrimination laws
2. **Perpetuating historical bias** - past hiring was biased
3. **Lack of transparency** - can't explain why candidate scored low
4. **Fairness** - qualified women unfairly rejected

### WHY BIAS OCCURRED:
- Training data reflects past hiring (mostly men hired historically)
- Model learned "being male" = "good candidate"
- Biased proxies (e.g., "interest in sports" correlated with gender)

### SOLUTIONS:
1. **Re-train with balanced data** - equal men/women examples
2. **Fairness constraints** - ensure equal opportunity across groups
3. **Remove biased features** - don't use gender-correlated features
4. **Human-in-loop** - model suggests, human decides
5. **Regular bias audits** - monitor predictions by demographic

### DECISION: **DO NOT DEPLOY** ❌
Even though 85% accuracy seems good, **15% lower score for women is unacceptable and illegal**. Fix bias first!

---

## 🟡 Solution 11: Data Quality Issues

### HANDLING STRATEGIES:

| Issue | Type | Priority | Solution |
|-------|------|----------|----------|
| 30% missing age | Missing data | HIGH | Impute with median or "Unknown" category |
| Prices = $0 | Error | CRITICAL | Remove or investigate (refunds? errors?) |
| Duplicates | Data quality | CRITICAL | Keep first occurrence, drop rest |
| Date format inconsistent | Format | HIGH | Standardize all to YYYY-MM-DD |
| Outliers >$100K | Outliers | MEDIUM | Investigate first! Could be legitimate (B2B sales) |

**Code (pseudocode)**:
```python
# 1. Handle duplicates
df = df.drop_duplicates(subset='transaction_id', keep='first')

# 2. Fix price errors
df = df[df['price'] > 0]  # Remove $0 prices

# 3. Standardize dates
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
```

---

## 🟡 Solution 12: Coding - EDA

### KEY INSIGHTS (Example):
1. **Electronics highest revenue** (avg $X per transaction)
2. **Younger customers** (18-30) spend more on Electronics
3. **Credit card** users have higher satisfaction (4.2 vs 3.8)
4. **Strong correlation**: price × quantity = total_amount (obvious but validates data)
5. **Recommendation**: Target young customers with Electronics promotions via Credit Card offers

**See full code implementation in Solutions notebook!**

---"""))

# Add Hard Solutions
cells.append(nbf.v4.new_markdown_cell("""# 🔴 Solution 13: Complete Project Design - Fraud Detection

## ✅ COMPREHENSIVE SYSTEM DESIGN:

### 1. BUSINESS UNDERSTANDING

**Success Metrics**:
- **Precision ≥ 90%** (minimize false positives - blocking legit transactions costly!)
- **Recall ≥ 70%** (catch 70% of fraud)
- **Latency <50ms** (real-time requirement)
- **ROI**: Save $3M+/year (from $5M current loss)

**Cost-Benefit**:
- False Positive cost: $20 (customer service + annoyance)
- False Negative cost: $500 (average fraud amount)
- Model prioritizes Precision to reduce customer friction

---

### 2. DATA STRATEGY

**Features** (20+ features):
1. Transaction amount
2. Merchant category
3. Location (distance from home)
4. Time of day
5. Day of week
6. Amount vs historical average (deviation)
7. Frequency (transactions per hour)
8. Merchant risk score
9. Card present/not present
10. International transaction (Y/N)
11. Recent password change
12. Failed login attempts
13. Device ID
14. IP address geolocation
15. Transaction velocity (transactions in last 1 hour)

**Handling 0.5% fraud (class imbalance)**:
- SMOTE (Synthetic Minority Over-sampling)
- Class weights in model
- Anomaly detection approach
- Ensemble methods

**Data Quality**:
- Real-time data pipeline
- Validate all features <50ms
- Handle missing values (median/mode imputation)

---

### 3. MODELING APPROACH

**Algorithms**:
1. **Baseline**: Logistic Regression (fast, interpretable)
2. **Best**: XGBoost or LightGBM (balance speed + accuracy)
3. **Alternative**: Isolation Forest (anomaly detection)

**Why NOT Deep Learning?**
- Latency requirement (<50ms) - DL too slow
- Need interpretability (explain to customers why blocked)
- Traditional ML sufficient for tabular data

**Evaluation**:
- **Primary metric**: Precision-Recall AUC (better than ROC for imbalanced)
- Cross-validation with time-based splits
- Cost-sensitive evaluation ($20 vs $500)

**Real-time Requirement**:
- Model inference <10ms
- Feature computation <30ms
- API overhead <10ms
- Total: ~50ms ✅

---

### 4. DEPLOYMENT ARCHITECTURE

**Tech Stack**:
- **Language**: Python
- **Model**: LightGBM (fast inference)
- **Serving**: FastAPI + Redis (caching)
- **Database**: PostgreSQL (transactions), Redis (real-time features)
- **Cloud**: AWS (EC2 for model, RDS for DB, ElastiCache for Redis)
- **Monitoring**: Prometheus + Grafana

**Architecture**:
```
Transaction → API Gateway → FastAPI (Model) → Response (<50ms)
                ↓
         Feature Store (Redis)
                ↓
         Log to Database (async)
```

**Monitoring**:
- Real-time: Latency, throughput, error rate
- Model: Precision, recall, fraud rate trends
- Alerts: If precision drops >5% → retrain
- A/B testing: New model vs current (20% traffic)

**Fallback**:
- If model fails: Use rule-based backup (block >$10K transactions)
- If latency >100ms: Skip ML, use rules
- Rollback plan: Previous model version ready

---

### 5. ETHICAL CONSIDERATIONS

**Bias**:
- Check fairness across demographics (geography, income level)
- Avoid using protected attributes (race, religion)
- Regular bias audits

**Privacy**:
- Encrypt sensitive data
- Comply with PCI DSS
- Minimal data retention

**Transparency**:
- Explain to customers WHY transaction blocked
- Appeals process
- Human review for high-value transactions (>$5K)

**Estimated Impact**: Save $3.5M/year, reduce fraud rate from 0.5% → 0.15%

---

## 🎯 KEY SUCCESS FACTORS:
1. **Speed**: <50ms is hard! Need efficient model + caching
2. **Precision**: False positives hurt customer experience
3. **Monitoring**: Fraudsters adapt - model must too (monthly retraining)
4. **Interpretability**: Must explain blocks to customers & regulators

---"""))

cells.append(nbf.v4.new_markdown_cell("""# 🔴 Solutions 14-15 (Final)

---

## 🔴 Solution 14: Critique & Improve Analysis

### PROBLEMS IDENTIFIED:

1. **Correlation ≠ Causation** ❌
   - Strong correlation doesn't mean bundling will work!

2. **Confounding Variable** 🌞
   - **Hidden variable: SUMMER / WEATHER**
   - Hot weather → buy ice cream AND sunglasses
   - Not causally related!

3. **Small Sample** (500 records, 1 location)
   - Not generalizable
   - Beach resort ≠ typical store

4. **Seasonal Bias**
   - Full year but didn't analyze seasonality properly

5. **No Control Group**
   - Didn't test bundle hypothesis

### IMPROVED ANALYSIS DESIGN:

**Proper Hypothesis**: "Beach weather drives BOTH ice cream & sunglasses sales"

**Correct Approach**:
1. **Control for weather**: Analyze correlation WITHIN same season
2. **Regression analysis**: `sales ~ weather + temperature + month`
3. **Test causality**: Experiment with bundle (A/B test!)
4. **Multiple locations**: Test in non-beach stores
5. **Customer survey**: Do they want bundle?

### RECOMMENDATION:

**Maybe bundle, but for DIFFERENT reason**:
- Not because they're correlated
- But because: "Summer convenience bundle" (people DO buy both in summer)
- Market as seasonal promotion (May-August)
- Test with small pilot first!

**What to analyze first**:
- Cross-purchase rate (% who buy BOTH)
- Price sensitivity (discount needed to drive sales?)
- Competitor bundles
- Customer segments (beachgoers vs others)

---

## 🔴 Solution 15: Complete Analysis Pipeline (Code)

### FULL IMPLEMENTATION:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data [code provided in exercise]

# PHASE 1: Data Understanding & Cleaning
print("="*70)
print("DATA QUALITY CHECK")
print("="*70)
print(f"Shape: {df.shape}")
print(f"\\nMissing values:\\n{df.isnull().sum()}")

# Handle missing values
df['income'].fillna(df['income'].median(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)

# Check outliers
print(f"\\nIncome outliers (>$100K): {(df['income'] > 100000).sum()}")
# Keep outliers - could be legitimate high earners

# PHASE 2: Analysis
print("\\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# 1. Click rate by channel
click_by_channel = df.groupby('campaign_channel')['clicked_ad'].mean().sort_values(ascending=False)
print(f"\\nClick Rate by Channel:\\n{click_by_channel}")

# 2. Conversion rate
conversion_by_channel = df.groupby('campaign_channel')['converted'].mean().sort_values(ascending=False)
print(f"\\nConversion Rate by Channel:\\n{conversion_by_channel}")

# 3. ROI analysis
# Assume costs: Email=$1, Social=$2, TV=$10, Direct Mail=$5
costs = {'Email': 1, 'Social Media': 2, 'TV': 10, 'Direct Mail': 5}
# Assume conversion value = $100

roi_data = []
for channel in df['campaign_channel'].unique():
    channel_df = df[df['campaign_channel'] == channel]
    conversions = channel_df['converted'].sum()
    total_cost = len(channel_df) * costs[channel]
    revenue = conversions * 100
    roi = (revenue - total_cost) / total_cost * 100
    roi_data.append({'Channel': channel, 'ROI': roi, 'Conversions': conversions})

roi_df = pd.DataFrame(roi_data).sort_values('ROI', ascending=False)
print(f"\\nROI by Channel:\\n{roi_df}")

# PHASE 3: Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Conversion by channel
axes[0,0].bar(conversion_by_channel.index, conversion_by_channel.values)
axes[0,0].set_title('Conversion Rate by Channel')
axes[0,0].set_ylabel('Conversion Rate')

# Plot 2: Age distribution
axes[0,1].hist(df['age'], bins=20, edgecolor='black')
axes[0,1].set_title('Customer Age Distribution')

# Plot 3: Income vs Conversion
for converted in [0, 1]:
    data = df[df['converted'] == converted]['income']
    axes[1,0].hist(data, alpha=0.5, label=f'Converted={converted}', bins=20)
axes[1,0].legend()
axes[1,0].set_title('Income Distribution by Conversion')

# Plot 4: Click → Conversion funnel
funnel_data = df.groupby('clicked_ad')['converted'].mean()
axes[1,1].bar(['Did Not Click', 'Clicked'], funnel_data.values)
axes[1,1].set_title('Conversion Rate: Clicked vs Not Clicked')

plt.tight_layout()
plt.show()

# PHASE 4: Recommendations
print("\\n" + "="*70)
print("TOP 3 RECOMMENDATIONS")
print("="*70)
print("\\n1. **Shift budget to Email** - Highest ROI, lowest cost")
print("   - Expected impact: +25% conversions with same budget")
print("   - Implementation: Reduce TV spend 50%, increase Email 200%")
print("\\n2. **Target higher-income customers** (>$60K)")
print("   - Conversion rate 2x higher in this segment")
print("   - Implementation: Audience targeting in Social Media")
print("\\n3. **Optimize click-to-conversion**")
print("   - 30% click but only 15% convert - landing page issue?")
print("   - Implementation: A/B test new landing page design")

# BONUS: Predictive Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Prepare features
X = df[['age', 'income', 'previous_purchases', 'days_since_last_purchase', 'clicked_ad']]
y = df['converted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"\\nModel Accuracy: {model.score(X_test, y_test):.2%}")
print("\\nFeature Importance:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feature}: {coef:.3f}")
```

### INSIGHTS:
1. Email best ROI (300%+)
2. High-income customers convert 2x more
3. Clicked ad → 50% conversion (vs 5% without click)

---

## 🎉 ALL SOLUTIONS COMPLETE!

Selamat! Anda telah menyelesaikan semua 15 exercises. Review solutions, understand reasoning, then retry exercises jika perlu!

**Remember**: Understanding > Memorizing answers!

---"""))

# Save complete solutions
nb['cells'] = cells
with open('01_solutions.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Module 01 Solutions COMPLETE!")
print(f"📊 Total cells: {len(cells)}")
print(f"📝 All 15 solutions with detailed explanations included!")
print(f"💾 Saved as: 01_solutions.ipynb")
