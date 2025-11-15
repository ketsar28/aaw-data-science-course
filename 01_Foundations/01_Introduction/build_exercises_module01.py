#!/usr/bin/env python3
"""
Build Module 01 - Introduction to Data Science EXERCISES
15 exercises covering all 6 parts with progressive difficulty
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =============================================================================
# HEADER
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📝 Module 01: Introduction to Data Science - EXERCISES

---

## 🎯 Tujuan Exercises

Exercises ini dirancang untuk **menguji pemahaman Anda** terhadap semua konsep yang dipelajari di Module 01!

### 📚 Coverage:
- ✅ **Part 1**: What is Data Science, AI vs ML vs DL
- ✅ **Part 2**: Roles in Data Science
- ✅ **Part 3**: Data Science Workflow (CRISP-DM)
- ✅ **Part 4**: Tools & Ecosystem
- ✅ **Part 5**: Data Scientist Mindset
- ✅ **Part 6**: Hands-on Analysis

---

## 📊 Exercise Breakdown:

| Difficulty | Count | Description |
|-----------|-------|-------------|
| 🟢 **Easy** | 5 | Recall & understanding (definitions, concepts) |
| 🟡 **Medium** | 7 | Application & analysis (use concepts, compare) |
| 🔴 **Hard** | 3 | Synthesis & evaluation (create, design, critique) |
| **TOTAL** | **15** | **Complete coverage** |

---

## 💡 Tips:

1. **Read carefully**: Setiap soal punya instruksi spesifik
2. **Use course material**: Refer back ke notebook jika perlu
3. **Think critically**: Beberapa soal butuh reasoning, bukan hafalan
4. **Practice coding**: Soal coding harus dijalankan!
5. **Check solutions**: Setelah selesai, compare dengan solution notebook

---

**Let's begin!** 💪

---"""))

# =============================================================================
# EASY EXERCISES (5)
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🟢 EASY EXERCISES (1-5)

Soal-soal ini menguji **basic understanding** dari konsep fundamental.

---"""))

# Exercise 1
cells.append(nbf.v4.new_markdown_cell("""## Exercise 1: Data Science Definition (Part 1) 🟢

**Pertanyaan**:

Lengkapi tabel berikut yang menjelaskan **4 Tipe Data Analytics**:

| Tipe | Pertanyaan | Contoh | Teknik |
|------|-----------|--------|---------|
| Descriptive | ??? | Total penjualan bulan ini | ??? |
| Diagnostic | Mengapa terjadi? | ??? | Correlation Analysis |
| ??? | Apa yang akan terjadi? | Prediksi penjualan bulan depan | ??? |
| Prescriptive | ??? | Rekomendasi strategi marketing | Optimization |

**Tasks**:
1. Isi semua yang bertanda `???`
2. Berikan 1 contoh tambahan untuk setiap tipe analytics

**Expected Output**: Tabel lengkap + 4 contoh tambahan

**Hint**: Review Part 1 section 1.1

---"""))

# Exercise 2
cells.append(nbf.v4.new_markdown_cell("""## Exercise 2: AI vs ML vs DL Hierarchy (Part 1) 🟢

**Pertanyaan**:

Klasifikasikan aplikasi berikut ke dalam kategori: **AI**, **ML**, atau **DL**

List of applications:
1. Netflix recommendation system
2. Siri voice assistant
3. Face recognition di iPhone (Face ID)
4. Rule-based chess program
5. Google Translate
6. Spam email filter di Gmail
7. ChatGPT
8. Alexa smart speaker

**Tasks**:
Buat tabel dengan kolom: Application, Category (AI/ML/DL), Reasoning

**Expected Output**: Tabel 8 baris dengan reasoning untuk setiap classification

**Hint**:
- AI = broad (bisa include rule-based)
- ML = learning from data
- DL = neural networks

---"""))

# Exercise 3
cells.append(nbf.v4.new_markdown_cell("""## Exercise 3: Data Science Roles (Part 2) 🟢

**Pertanyaan**:

Match each **responsibility** dengan **role** yang sesuai:

**Responsibilities**:
A. Build production-ready ML APIs with FastAPI
B. Create Tableau dashboards untuk monthly sales report
C. Build XGBoost model untuk predict customer churn
D. Optimize database query dari 5 menit ke 10 detik
E. Deploy model dengan Docker ke AWS
F. Run A/B test untuk new website feature

**Roles**:
1. Data Analyst
2. Data Scientist
3. ML Engineer
4. Data Engineer

**Tasks**:
Create mapping (contoh: A → ML Engineer) dan jelaskan reasoning

**Expected Output**:
- 6 mappings (A-F → role)
- Brief reasoning untuk each

**Hint**: Review Part 2 comparison table

---"""))

# Exercise 4
cells.append(nbf.v4.new_markdown_cell("""## Exercise 4: CRISP-DM Phases (Part 3) 🟢

**Pertanyaan**:

Berikut adalah activities dari sebuah Data Science project. **Sort them** sesuai urutan CRISP-DM phases:

**Activities** (acak):
- A. Train Random Forest dan XGBoost model, compare performance
- B. Interview stakeholders untuk understand problem
- C. Check test set performance, get stakeholder approval
- D. Handle missing values, create new features, encode categorical
- E. Explore data quality, check distributions, find correlations
- F. Containerize dengan Docker, deploy ke AWS, setup monitoring

**Tasks**:
1. Sort activities A-F sesuai CRISP-DM order
2. Name each phase (contoh: Phase 1 - Business Understanding)

**Expected Output**:
Ordered list dengan phase names

**Hint**: CRISP-DM punya 6 phases

---"""))

# Exercise 5
cells.append(nbf.v4.new_markdown_cell("""## Exercise 5: Python Libraries (Part 4) 🟢

**Pertanyaan**:

Match setiap **library** dengan **use case** yang paling sesuai:

**Libraries**:
1. Pandas
2. Matplotlib
3. Scikit-learn
4. TensorFlow
5. Seaborn
6. NumPy

**Use Cases**:
A. Build Convolutional Neural Network untuk image classification
B. Load CSV file dan filter rows based on condition
C. Create histogram untuk visualize sales distribution
D. Train Random Forest classifier
E. Perform matrix multiplication dan linear algebra operations
F. Create beautiful correlation heatmap

**Tasks**:
Create mappings (1-6 → A-F)

**Expected Output**: 6 mappings dengan brief explanation

**Hint**: Review Part 4 section 4.1

---"""))

# =============================================================================
# MEDIUM EXERCISES (7)
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🟡 MEDIUM EXERCISES (6-12)

Soal-soal ini menguji **application & analysis** - menggunakan konsep untuk solve problems.

---"""))

# Exercise 6
cells.append(nbf.v4.new_markdown_cell("""## Exercise 6: When to Use Which Algorithm? (Part 1) 🟡

**Scenario**:

Anda punya dataset dengan **100,000 images** kucing dan anjing. Anda ingin build classifier.

**Option 1**: Rule-based (if has_whiskers and pointy_ears → cat)
**Option 2**: Traditional ML (Decision Tree dengan manual features)
**Option 3**: Deep Learning (CNN)

**Tasks**:
1. **Compare** ketiga approaches: List 3 pros dan 3 cons untuk each
2. **Recommend** approach terbaik untuk scenario ini dengan detailed reasoning
3. **Explain** kapan Anda akan pilih approach lain (different scenario)

**Expected Output**:
- Comparison table (3x3 pros/cons)
- Recommendation (1 paragraph)
- Alternative scenarios (2-3 examples)

**Hint**: Consider: data size, complexity, accuracy needs, interpretability

---"""))

# Exercise 7
cells.append(nbf.v4.new_markdown_cell("""## Exercise 7: Career Path Design (Part 2) 🟡

**Scenario**:

Anda fresh graduate Computer Science dengan skills:
- Python programming (intermediate)
- Basic SQL
- No ML experience
- No statistics background

**Goal**: Menjadi **Senior Data Scientist** dalam 5 tahun

**Tasks**:
1. **Design learning path** tahun per tahun (Year 1-5)
2. **Specify**:
   - Skills to learn each year
   - Tools/frameworks to master
   - Recommended projects
   - Target job title progression
3. **Estimate** time investment per week

**Expected Output**:
Detailed 5-year plan (table format recommended)

**Hint**: Review Part 2 career progression paths

---"""))

# Exercise 8
cells.append(nbf.v4.new_markdown_cell("""## Exercise 8: CRISP-DM Application (Part 3) 🟡

**Scenario**:

Hospital ingin predict **patient readmission risk** (apakah pasien akan kembali dalam 30 hari setelah discharge). Goal: Reduce readmission rate dari 25% ke 15%.

**Tasks**:

Design complete project plan mengikuti **CRISP-DM**, include untuk setiap phase:
1. **Goals & key questions**
2. **Activities** (at least 3 per phase)
3. **Expected outputs**
4. **Time estimate**
5. **Success criteria**

**Expected Output**:
Complete project plan covering all 6 CRISP-DM phases

**Hint**: Think medical domain specifics (ethics, interpretability critical!)

---"""))

# Exercise 9
cells.append(nbf.v4.new_markdown_cell("""## Exercise 9: Tech Stack Selection (Part 4) 🟡

**Scenario**:

Startup ingin build **recommendation system** dengan requirements:
- 1 million users
- Real-time recommendations (<100ms latency)
- Budget: Limited (prefer open-source)
- Team: 2 ML engineers, 1 data scientist

**Tasks**:

Design complete **tech stack** dan justify choices:
1. **Programming language**
2. **ML framework** (training)
3. **Model serving** (deployment)
4. **Database**
5. **Cloud platform**
6. **MLOps tools** (experiment tracking, monitoring)
7. **Development environment**

**Expected Output**:
Tech stack with justification untuk each choice (why this over alternatives?)

**Hint**: Consider: performance, cost, team skills, scalability

---"""))

# Exercise 10
cells.append(nbf.v4.new_markdown_cell("""## Exercise 10: Ethical Dilemma (Part 5) 🟡

**Scenario**:

Company ingin build ML model untuk **screen job applicants**. Model akan predict "good candidate" probability based on:
- Resume text
- Years of experience
- Education background
- Previous employers

Historical data shows model achieves 85% accuracy pada test set.

**However**: Analysis shows model has **bias** - systematically rates women candidates 15% lower than men with same qualifications.

**Tasks**:

1. **Identify** ethical issues (at least 4)
2. **Explain** why bias occurred (possible reasons)
3. **Propose** solutions to mitigate bias (at least 3 approaches)
4. **Decide**: Should company deploy this model as-is? Why/why not?

**Expected Output**:
Comprehensive ethical analysis dengan actionable recommendations

**Hint**: Review Part 5 section 5.2 (Ethical & Responsible)

---"""))

# Exercise 11
cells.append(nbf.v4.new_markdown_cell("""## Exercise 11: Data Quality Issues (Part 5) 🟡

**Scenario**:

Anda analyze dataset e-commerce sales dengan findings:
- 30% missing values di column "customer_age"
- Beberapa prices = $0 (clearly errors)
- Duplicate transaction IDs (500 rows)
- Column "date" format inconsistent (mix of DD/MM/YYYY dan MM/DD/YYYY)
- Outliers: Some transactions >$100,000 (normal range: $10-$500)

**Tasks**:

1. **Classify** each issue: Data quality problem type
2. **Propose** handling strategy untuk each (minimum 2 options per issue)
3. **Prioritize**: Which issues MUST be fixed before modeling? Which bisa di-tolerate?
4. **Write code** (pseudocode ok) untuk fix top 3 priority issues

**Expected Output**:
- Issue classification table
- Handling strategies
- Prioritization dengan reasoning
- Code/pseudocode

---"""))

# Exercise 12
cells.append(nbf.v4.new_markdown_cell("""## Exercise 12: Coding - Exploratory Data Analysis (Part 6) 🟡

**Task**:

Given sales dataset, perform **complete EDA** dan generate insights!

**Code & Analyze**:

```python
import pandas as pd
import numpy as np

# Sample dataset: Online retail sales
np.random.seed(42)
data = {
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 200),
    'price': np.random.uniform(10, 500, 200),
    'quantity': np.random.randint(1, 10, 200),
    'customer_age': np.random.randint(18, 70, 200),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Cash'], 200),
    'customer_satisfaction': np.random.randint(1, 6, 200)  # 1-5 scale
}
df = pd.DataFrame(data)
df['total_amount'] = df['price'] * df['quantity']

# Add some missing values
df.loc[np.random.choice(df.index, 20), 'customer_age'] = np.nan
```

**Requirements**:
1. Basic info (shape, dtypes, missing values)
2. Descriptive statistics
3. Distribution analysis (histogram untuk numerical columns)
4. Categorical analysis (value counts, percentages)
5. Correlation analysis (which factors relate to customer_satisfaction?)
6. Group analysis (average total_amount by product_category)
7. **Generate 5 business insights** from analysis

**Expected Output**:
- Code cells dengan analysis
- Visualizations (at least 3)
- 5 actionable business insights

---"""))

# =============================================================================
# HARD EXERCISES (3)
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🔴 HARD EXERCISES (13-15)

Soal-soal ini menguji **synthesis & evaluation** - create solutions, design systems, critique approaches.

---"""))

# Exercise 13
cells.append(nbf.v4.new_markdown_cell("""## Exercise 13: Complete Project Design (All Parts) 🔴

**Scenario**:

Bank ingin reduce **credit card fraud losses** (currently $5M/year). Build complete ML system untuk detect fraudulent transactions in real-time.

**Constraints**:
- Must detect fraud <50ms (real-time)
- False positives costly (block legitimate transactions → angry customers)
- Fraud is rare (~0.5% of transactions)
- Available data: 2 years historical transactions (10M records)

**Tasks**:

Design **END-TO-END ML system** covering:

**1. Business Understanding**:
- Success metrics (be specific!)
- ROI calculation
- Risk assessment

**2. Data Strategy**:
- Features needed (list at least 10)
- Data quality concerns
- Handling class imbalance (0.5% fraud)

**3. Modeling Approach**:
- Algorithm selection (justify!)
- Evaluation metrics (which one primary? why?)
- How to handle real-time requirement?

**4. Deployment Architecture**:
- Tech stack (infrastructure, frameworks, databases)
- Monitoring strategy
- Fallback/rollback plan

**5. Ethical Considerations**:
- Bias concerns
- Privacy/security
- Transparency

**Expected Output**:
Comprehensive system design document (2-3 pages)

**Evaluation Criteria**:
- Completeness
- Feasibility
- Technical depth
- Business alignment

---"""))

# Exercise 14
cells.append(nbf.v4.new_markdown_cell("""## Exercise 14: Critique & Improve Analysis (Part 3 & 5) 🔴

**Scenario**:

Junior Data Scientist presents analysis:

> "I analyzed 500 customer records and found strong correlation (r=0.8) between ice cream sales and sunglasses sales in our store. Therefore, we should bundle ice cream + sunglasses together to increase sales!"

**Additional context**:
- Data: 1 year (Jan-Dec)
- Location: Beach resort store
- Both products show seasonal pattern

**Tasks**:

**1. Identify Problems** (at least 5):
- Statistical issues
- Logical fallacies
- Missing considerations

**2. Critique Methodology**:
- What should have been done differently?
- What analyses are missing?

**3. Design Better Analysis**:
- Proper hypothesis
- Correct statistical approach
- Additional data/features needed
- Appropriate visualization

**4. Recommendation**:
- Is bundling a good idea? (maybe yes, maybe no - justify!)
- What ELSE to analyze first?

**Expected Output**:
- Detailed critique (1 page)
- Improved analysis design (1 page)
- Recommendation dengan evidence

**Hint**: Correlation ≠ Causation! Think confounding variables.

---"""))

# Exercise 15
cells.append(nbf.v4.new_markdown_cell("""## Exercise 15: Coding - Complete Analysis Pipeline (Part 6) 🔴

**Challenge**:

Build **complete data analysis pipeline** untuk marketing campaign dataset!

**Dataset**:
```python
# Marketing Campaign Dataset
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = {
    'customer_id': range(1, n+1),
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(50000, 20000, n),
    'campaign_channel': np.random.choice(['Email', 'Social Media', 'TV', 'Direct Mail'], n),
    'previous_purchases': np.random.randint(0, 20, n),
    'days_since_last_purchase': np.random.randint(1, 365, n),
    'clicked_ad': np.random.choice([0, 1], n, p=[0.7, 0.3]),  # 30% click rate
    'converted': np.random.choice([0, 1], n, p=[0.85, 0.15])  # 15% conversion rate
}

df = pd.DataFrame(data)

# Add missing values
df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
df.loc[np.random.choice(df.index, 30), 'age'] = np.nan

# Add some realistic patterns
df.loc[df['clicked_ad'] == 1, 'converted'] = np.random.choice([0, 1], (df['clicked_ad'] == 1).sum(), p=[0.5, 0.5])
```

**Requirements**:

**Phase 1: Data Understanding & Cleaning**
1. Comprehensive EDA
2. Handle missing values (justify approach)
3. Check for outliers (handle appropriately)
4. Validate data quality

**Phase 2: Analysis**
1. What factors influence **ad clicks**?
2. What factors influence **conversion**?
3. Which campaign channel most effective?
4. Customer segmentation (at least 2 segments)

**Phase 3: Visualization & Communication**
1. Create dashboard (4+ plots)
2. Clear insights for each visualization
3. Professional formatting

**Phase 4: Recommendations**
1. Top 3 actionable recommendations untuk marketing team
2. Each recommendation: Include expected impact & implementation plan

**Expected Output**:
- Fully executable notebook
- Professional visualizations
- Business insights
- Actionable recommendations

**Bonus** (+10 points):
- Build simple predictive model (logistic regression) untuk predict conversion
- Evaluate model performance
- Feature importance analysis

**Evaluation**:
- Code quality: 25%
- Analysis depth: 25%
- Visualizations: 25%
- Business insights: 25%

---"""))

# =============================================================================
# SUBMISSION & NEXT STEPS
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""---

# 🎯 Selesai!

## ✅ What to Do Next:

1. **Complete ALL exercises** (don't skip!)
2. **Check solutions**: Compare your answers dengan solution notebook
3. **Reflect**: Which topics perlu di-review?
4. **Practice more**: Create your own mini projects!

---

## 📊 Self-Assessment:

Setelah selesai, evaluate yourself:

| Difficulty | Target Correct | Your Score |
|-----------|----------------|------------|
| 🟢 Easy (1-5) | 5/5 (100%) | ___/5 |
| 🟡 Medium (6-12) | 5/7 (70%+) | ___/7 |
| 🔴 Hard (13-15) | 2/3 (65%+) | ___/3 |
| **TOTAL** | **12/15 (80%+)** | **___/15** |

**Passing Score**: 12/15 (80%)

**If you scored**:
- ✅ **13-15**: Excellent! Ready for Module 02
- ✅ **10-12**: Good! Review topics you struggled with
- ⚠️ **<10**: Need more practice - review course material & retry

---

## 📚 Additional Resources:

**Want more practice?**
- Kaggle Learn (free courses)
- DataCamp exercises
- LeetCode (for coding)
- Personal projects!

---

**Great job completing these exercises!** 🎉

**Next**: Module 02 - Python for Data Science (Deep Dive)

---"""))

# Save notebook
nb['cells'] = cells
with open('01_exercises.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Module 01 Exercises created!")
print(f"📊 Total cells: {len(cells)}")
print(f"📝 Total exercises: 15 (5 Easy + 7 Medium + 3 Hard)")
print(f"💾 Saved as: 01_exercises.ipynb")
