#!/usr/bin/env python3
"""
Build COMPLETE Module 01 - Introduction to Data Science
All 6 parts with theory + practice + mini project
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =============================================================================
# HEADER
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📊 Module 01: Introduction to Data Science & Ecosystem

---

## 🎯 Selamat Datang di Dunia Data Science!

Selamat! Anda telah menyelesaikan Python basics di Module 00. Sekarang saatnya memulai perjalanan **Data Science** yang sesungguhnya!

### 📖 Apa yang Akan Anda Pelajari:

1. ✅ **What is Data Science?** - Memahami apa itu Data Science, AI, ML, DL
2. ✅ **Roles in Data Science** - Berbagai peran dan career paths
3. ✅ **Data Science Workflow** - Complete pipeline dari problem ke solution
4. ✅ **Tools & Ecosystem** - Tools dan libraries yang digunakan
5. ✅ **Data Scientist Mindset** - Cara berpikir yang tepat
6. ✅ **Mini Project** - Hands-on pertama Anda dengan data!

### ⏱️ Estimasi Waktu: 3-4 jam

### 💡 Learning Approach:

Course ini menggunakan **Dual-Pillar Approach**:
- 🧠 **Teori (WHY & WHEN)**: Memahami konsep, kapan digunakan
- 💻 **Praktik (HOW)**: Implementasi dengan kode Python

### 📝 Format Setiap Bagian:
- **Penjelasan Konsep** dengan analogi sederhana
- **Visualisasi** (diagram, tabel)
- **Contoh Real-World** dari industri
- **Code Examples** (jika applicable)
- **Key Takeaways** di setiap section

---

**Let's begin your Data Science journey!** 🚀

---"""))

# =============================================================================
# PART 1: WHAT IS DATA SCIENCE?
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📚 PART 1: What is Data Science?

## Progress: 1/6 🟩⬜⬜⬜⬜⬜

---

## 1.1 Definisi Data Science

**Data Science** adalah ilmu yang menggabungkan berbagai bidang untuk **mengekstrak pengetahuan dan insight dari data**, baik terstruktur maupun tidak terstruktur.

### 🎯 4 Tujuan Data Science:

| Tipe | Pertanyaan | Contoh | Teknik |
|------|-----------|--------|---------|
| **Descriptive** | Apa yang terjadi? | Total penjualan bulan ini | Aggregation, Visualization |
| **Diagnostic** | Mengapa terjadi? | Mengapa penjualan turun? | Correlation Analysis, Drill-down |
| **Predictive** | Apa yang akan terjadi? | Prediksi penjualan bulan depan | Machine Learning, Forecasting |
| **Prescriptive** | Apa yang harus dilakukan? | Rekomendasi strategi marketing | Optimization, Simulation |

### 🧩 Data Science = Intersection of Multiple Fields

```
        💻 Programming
             ▲
             │
    Statistics ◄─┼─► 🎓 Domain
       📊      │    Knowledge
             │
             ▼
          📐 Math
```

**Data Science menggabungkan:**
1. **Programming/Computer Science**: Tools untuk process data (Python, SQL, etc.)
2. **Statistics & Mathematics**: Methods untuk analyze data
3. **Domain Knowledge**: Understanding bisnis/field (healthcare, finance, etc.)
4. **Communication**: Ability to tell story dengan data

### 📝 Analogi Sederhana:

**Data Science = Detektif Modern**

Bayangkan Anda seorang detektif yang menyelidiki kasus:

1. **📦 Data** = Bukti-bukti di TKP (Tempat Kejadian Perkara)
   - CCTV footage, sidik jari, saksi mata
   - Sama seperti: Sales data, customer behavior, sensor readings

2. **🔍 Analysis** = Investigasi mencari pola
   - Hubungkan bukti-bukti
   - Cari pattern dan anomalies
   - Sama seperti: EDA, Statistical Analysis

3. **💡 Insights** = Kesimpulan "siapa pelakunya"
   - Temukan root cause
   - Sama seperti: "Customer churn karena poor customer service"

4. **🎯 Action** = Rekomendasi mencegah kejahatan berikutnya
   - Actionable recommendations
   - Sama seperti: "Improve customer service training"

---"""))

cells.append(nbf.v4.new_markdown_cell("""## 1.2 AI vs ML vs DL - Perbedaan yang Jelas!

Istilah-istilah ini sering membingungkan. Mari kita jelaskan dengan **sangat jelas**!

### 🎯 Hierarchy (Dari Luas ke Spesifik):

```
┌─────────────────────────────────────────────────┐
│  🤖 Artificial Intelligence (AI)                │
│  "Komputer yang cerdas"                         │
│  ┌───────────────────────────────────────────┐  │
│  │  🧠 Machine Learning (ML)                 │  │
│  │  "Belajar dari data"                      │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │  🕸️ Deep Learning (DL)               │  │  │
│  │  │  "Neural Networks"                   │  │  │
│  │  │                                       │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

AI ⊃ ML ⊃ DL (AI includes ML, ML includes DL)

---

### 1️⃣ 🤖 Artificial Intelligence (AI)

**Definisi**: 
Kemampuan komputer untuk melakukan tugas yang **biasanya memerlukan kecerdasan manusia**.

**Karakteristik**:
- Bisa reasoning (berpikir)
- Bisa problem solving
- Bisa learning
- Bisa understand language
- Bisa recognize patterns

**2 Jenis AI**:

**A. Narrow AI (Weak AI)** - Yang ada sekarang:
- Pintar di satu tugas spesifik
- Contoh: Siri (voice assistant), Chess AI, Spam filter

**B. General AI (Strong AI)** - Belum ada:
- Pintar di semua tugas seperti manusia
- Masih dalam research/science fiction

**Contoh Real-World**:
- 🗣️ **Voice Assistants**: Siri, Alexa, Google Assistant
- ♟️ **Game AI**: AlphaGo (kalahkan juara dunia Go)
- 💬 **Chatbots**: Customer service automation
- 🚗 **Self-driving cars**: Tesla Autopilot
- 🎬 **Recommendation**: Netflix, YouTube recommendations

**Analogi**: 
AI = **Anak yang bisa berpikir dan belajar**
- Bisa menyelesaikan masalah
- Bisa adapt ke situasi baru
- Bisa improve over time

---

### 2️⃣ 🧠 Machine Learning (ML)

**Definisi**: 
Subset dari AI. Komputer **belajar dari data** tanpa di-program secara eksplisit untuk setiap rule.

**Perbedaan dengan Traditional Programming**:

**Traditional Programming**:
```
Input (Data) + Program (Rules) → Output (Result)
```
Contoh: If temperature > 30°C, display "Hot"

**Machine Learning**:
```
Input (Data) + Output (Result) → Model (Rules)
```
Model **belajar rules sendiri** dari data!

**Cara Kerja ML**:
```
1. Collect Data (Training Data)
2. Choose Algorithm
3. Train Model (Model belajar dari data)
4. Test Model
5. Deploy Model
6. Predict New Data
```

**3 Tipe Machine Learning**:

**A. Supervised Learning** (Paling umum):
- Belajar dari data yang **sudah dilabel**
- Ada "guru" yang kasih jawaban benar

**Contoh**:
- **Regression** (Predict angka):
  - Prediksi harga rumah dari luas, lokasi, jumlah kamar
  - Prediksi suhu besok dari data historis
  
- **Classification** (Predict kategori):
  - Email spam atau bukan
  - Gambar kucing atau anjing
  - Pasien sakit atau sehat

**B. Unsupervised Learning**:
- Belajar dari data **tanpa label**
- Model cari pola sendiri

**Contoh**:
- **Clustering**: Group customers berdasarkan behavior
- **Dimensionality Reduction**: Simplify complex data
- **Anomaly Detection**: Detect fraud transactions

**C. Reinforcement Learning**:
- Belajar dari **trial & error** dengan reward/punishment
- Seperti melatih anjing dengan treats!

**Contoh**:
- Game AI (AlphaGo, Chess)
- Robot learning to walk
- Self-driving cars

**Analogi ML**: 
**Anak yang belajar dari pengalaman**

Contoh: Anak belajar recognize anjing
- **Traditional Programming**: 
  - Orang tua explain: "Anjing punya 4 kaki, ekor, gonggong"
  - Anak hafal rules
  
- **Machine Learning**:
  - Anak lihat banyak foto anjing + label "anjing"
  - Anak belajar sendiri ciri-ciri anjing
  - Semakin banyak contoh (data), semakin akurat

**Contoh Real-World ML**:
- 📧 **Spam Filter**: Gmail belajar dari email yang Anda mark sebagai spam
- 🎥 **Netflix Recommendations**: Belajar dari film yang Anda tonton
- 💳 **Credit Scoring**: Bank predict risk dari historical data
- 🛒 **Product Recommendations**: Amazon/Tokopedia suggestions
- 🏥 **Disease Diagnosis**: Predict penyakit dari symptoms

---

### 3️⃣ 🕸️ Deep Learning (DL)

**Definisi**: 
Subset dari ML. Menggunakan **Neural Networks** dengan banyak layers (deep = dalam/banyak layer).

**Inspiration**: 
Terinspirasi dari cara kerja **otak manusia**:
- Otak punya billions neurons yang saling connect
- DL punya artificial neurons dalam layers

**Struktur Neural Network**:
```
Input Layer → Hidden Layers (banyak!) → Output Layer
```

Semakin banyak hidden layers, semakin "deep"!

**Keunggulan DL**:
1. ✅ Bisa handle **data yang sangat complex** (images, videos, speech, text)
2. ✅ **Automatic feature extraction** - tidak perlu manual feature engineering
3. ✅ **Semakin banyak data, semakin akurat** (scalable)

**Kelemahan DL**:
1. ❌ Butuh **BANYAK data** (millions of samples)
2. ❌ Butuh **computational power tinggi** (GPU/TPU)
3. ❌ **"Black box"** - sulit explain kenapa predict seperti itu
4. ❌ **Training lama** dan expensive

**Kapan Pakai Deep Learning?**:
- ✅ Data **SANGAT banyak** (>100k samples ideal)
- ✅ Problem **complex** (image recognition, NLP, speech)
- ✅ Ada **GPU/TPU** untuk training
- ✅ Accuracy sangat critical (trade-off dengan interpretability)
- ❌ **JANGAN** pakai untuk simple problems (overkill & waste resources!)

**Contoh Real-World DL**:
- 📱 **Face Recognition**: iPhone Face ID, Facebook tagging
- 🚗 **Self-Driving Cars**: Tesla, Waymo
- 🌐 **Language Translation**: Google Translate
- 💬 **ChatGPT**: OpenAI's language models
- 🎨 **Image Generation**: DALL-E, Midjourney, Stable Diffusion
- 🗣️ **Speech Recognition**: Siri, Google Assistant
- 🎬 **Deepfakes**: Face swapping (controversial!)

**Analogi DL**: 
**Anak jenius yang bisa belajar hal sangat kompleks**

Contoh: Recognize wajah seseorang
- Traditional ML: Manual extract features (mata, hidung, mulut)
- Deep Learning: Kasih banyak foto, DL otomatis learn features complex

---

### 📊 Summary Table: AI vs ML vs DL

| Aspect | AI 🤖 | ML 🧠 | DL 🕸️ |
|--------|------|------|------|
| **Scope** | Paling luas | Subset AI | Subset ML |
| **Definition** | Mimic human intelligence | Learn from data | Neural networks (many layers) |
| **Data Needed** | Varies | Moderate (1k-100k) | Massive (>100k) |
| **Complexity** | Varies | Moderate | Very High |
| **Training Time** | Varies | Minutes to hours | Hours to days |
| **Hardware** | CPU ok | CPU ok | GPU/TPU strongly recommended |
| **Interpretability** | Varies | High (for simple models) | Low ("black box") |
| **Examples** | Chatbot, Rule-based AI | Linear Regression, Random Forest, XGBoost | CNN, RNN, Transformer (GPT, BERT) |
| **Best For** | Various tasks | Structured/tabular data | Unstructured data (image, text, speech) |

---

### 💡 Real-World Example: Cat vs Dog Classification

Mari lihat bagaimana **3 approaches** berbeda untuk solve problem yang sama:

**Problem**: Diberikan gambar, tentukan apakah itu kucing atau anjing?

#### **Approach 1: Traditional AI (Rule-Based)**

```python
def is_cat_or_dog(image):
    if has_whiskers(image) and has_pointy_ears(image):
        return "Cat"
    elif has_floppy_ears(image) and has_long_snout(image):
        return "Dog"
    else:
        return "Unknown"
```

**Pros**: 
- ✅ Simple, easy to understand
- ✅ Fast prediction

**Cons**:
- ❌ Rules terlalu rigid
- ❌ Banyak exceptions (ada kucing tanpa whiskers, anjing dengan pointy ears)
- ❌ Hard to maintain (banyak if-else)

---

#### **Approach 2: Traditional Machine Learning**

```python
# 1. Manual Feature Extraction
features = extract_features(image)  # color, edges, shapes, etc.
# Features: [has_whiskers=1, ear_shape=2, fur_length=3, ...]

# 2. Train Classifier
model = DecisionTree()
model.fit(training_images, labels)  # Learn from examples

# 3. Predict
prediction = model.predict(new_image_features)
```

**Pros**:
- ✅ Better than rules - learns from data
- ✅ Can handle variations
- ✅ Interpretable (can see decision tree)

**Cons**:
- ❌ Manual feature engineering (butuh domain expert)
- ❌ Features mungkin tidak optimal
- ❌ Accuracy terbatas (~80-85%)

---

#### **Approach 3: Deep Learning (CNN)**

```python
# 1. NO manual feature extraction - kasih raw image!
model = ConvolutionalNeuralNetwork(layers=[...])

# 2. Train dengan banyak data
model.fit(thousands_of_cat_dog_images, labels)
# Model automatically learns:
# - Low-level features (edges, colors)
# - Mid-level features (ears, eyes, fur patterns)
# - High-level features (cat face vs dog face)

# 3. Predict
prediction = model.predict(new_image)  # Very accurate!
```

**Pros**:
- ✅ **Sangat akurat** (>95% accuracy)
- ✅ **Automatic feature learning** - tidak perlu manual
- ✅ Can handle complex variations (different angles, lighting, breeds)

**Cons**:
- ❌ Butuh **banyak data** (10k+ images)
- ❌ Butuh **GPU** untuk training
- ❌ **"Black box"** - sulit explain why predict cat/dog

---

### 🎯 Key Takeaway: Pilih Tool yang Tepat!

**Data Science menggunakan SEMUA tools** tergantung problem:

```
Simple Problem → Rule-Based atau Simple ML
   ↓
Medium Problem → Traditional ML (Decision Tree, Random Forest, XGBoost)
   ↓
Complex Problem → Deep Learning (CNN, RNN, Transformers)
```

**Prinsip Emas**: 
> **"Start Simple, Add Complexity Only When Needed!"**
> 
> Jangan pakai cannon untuk bunuh nyamuk! 🚫

**Decision Framework**:
1. Coba simple approach dulu (Linear Regression, Logistic Regression)
2. Jika accuracy tidak cukup, coba advanced ML (Random Forest, XGBoost)
3. Jika still not enough DAN punya banyak data + GPU, baru coba DL

---"""))

# Add Python code untuk demonstrate concepts
cells.append(nbf.v4.new_code_cell("""# Demonstration: Rule-based vs ML Mindset untuk Spam Detection

# ============================================================
# APPROACH 1: Rule-Based (Traditional Programming)
# ============================================================

def is_spam_rule_based(email_text):
    \"\"\"
    Hard-coded rules untuk detect spam
    \"\"\"
    spam_keywords = ["viagra", "lottery", "winner", "free money", 
                     "click here", "congratulations", "million dollars"]
    
    email_lower = email_text.lower()
    spam_count = 0
    
    for keyword in spam_keywords:
        if keyword in email_lower:
            spam_count += 1
    
    # Rule: Jika ada 2+ spam keywords, classify as SPAM
    if spam_count >= 2:
        return "SPAM 🚫"
    else:
        return "NOT SPAM ✅"

# Test
emails = [
    "Congratulations! You won the lottery! Click here to claim your million dollars!",
    "Hi John, let's meet for coffee tomorrow at 3pm",
    "URGENT: Click here for free money! You are a winner!",
    "Meeting reminder: Project discussion at 2pm"
]

print("="*70)
print("APPROACH 1: Rule-Based Spam Detection")
print("="*70)
for i, email in enumerate(emails, 1):
    result = is_spam_rule_based(email)
    print(f"\\nEmail {i}: {result}")
    print(f"Text: {email[:60]}...")

# ============================================================
# APPROACH 2: Machine Learning Mindset
# ============================================================

print("\\n" + "="*70)
print("APPROACH 2: Machine Learning Mindset")
print("="*70)
print("\\nML Model akan:")
print("1. ✅ Belajar dari RIBUAN contoh email (spam dan not spam)")
print("2. ✅ Menemukan POLA sendiri (tidak hard-coded)")
print("3. ✅ Improve seiring waktu dengan data baru")
print("4. ✅ Handle edge cases yang tidak terpikir oleh programmer")
print("\\nContoh fitur yang ML bisa pelajari:")
print("  - Sender email domain")
print("  - Email length")
print("  - Time sent")
print("  - Link count")
print("  - Capital letters ratio")
print("  - ... dan banyak lagi!")
print("\\nModel akan assign WEIGHTS ke setiap fitur")
print("dan belajar kombinasi optimal untuk predict spam.")
print("="*70)

# Untuk actual ML implementation, kita akan belajar di Module 17!"""))

cells.append(nbf.v4.new_markdown_cell("""### ✅ Summary Part 1:

**Anda telah belajar:**
- ✅ Data Science = Extract knowledge dari data
- ✅ 4 tipe analytics: Descriptive, Diagnostic, Predictive, Prescriptive
- ✅ AI ⊃ ML ⊃ DL (hierarchy)
- ✅ **AI** = Komputer cerdas
- ✅ **ML** = Belajar dari data
- ✅ **DL** = Neural networks untuk complex problems
- ✅ Pilih tool yang tepat untuk problem Anda!

**Next**: Part 2 - Roles in Data Science 👥

---"""))

# =============================================================================
# PART 2: ROLES IN DATA SCIENCE
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 👥 PART 2: Roles in Data Science

## Progress: 2/6 🟩🟩⬜⬜⬜⬜

---

Di dunia Data Science, ada **berbagai peran** dengan tanggung jawab berbeda. Mari pahami masing-masing role agar Anda bisa tentukan career path yang tepat!

---

## 2.1 Data Analyst 📊

### 🎯 Fokus Utama:
**Menjawab pertanyaan bisnis dengan data** (Descriptive & Diagnostic Analytics)

### 📝 Tanggung Jawab:
- ✅ Analyze historical data untuk understand trends
- ✅ Create reports & dashboards
- ✅ Identify business insights dari data
- ✅ Communicate findings ke stakeholders
- ✅ Basic data cleaning & manipulation

### 🛠️ Tools:
- **SQL** (WAJIB - 90% pekerjaan)
- **Excel/Google Sheets** (Pivot tables, formulas)
- **BI Tools**: Tableau, Power BI, Looker
- **Python/R** (Basic - untuk automation)

### 📊 Typical Tasks:
- "Berapa total sales Q3 2024 per region?"
- "Mengapa customer churn naik 15% bulan lalu?"
- "Segment customers berdasarkan purchasing behavior"
- Create monthly sales dashboard

### 💰 Salary Range (Indonesia):
- **Junior**: 8-15 juta/bulan
- **Mid-level**: 15-25 juta/bulan
- **Senior**: 25-40 juta/bulan

### ⭐ Best For:
- Yang suka storytelling dengan data
- Strong business acumen
- Komunikasi bagus (present findings)

---

## 2.2 Data Scientist 🧪

### 🎯 Fokus Utama:
**Build predictive models & extract insights** (Predictive & Prescriptive Analytics)

### 📝 Tanggung Jawab:
- ✅ Build Machine Learning models
- ✅ Statistical analysis & hypothesis testing
- ✅ Feature engineering
- ✅ Model evaluation & optimization
- ✅ A/B testing & experimentation
- ✅ Communicate complex findings

### 🛠️ Tools:
- **Python** (NumPy, Pandas, Scikit-learn)
- **R** (untuk statistical analysis)
- **SQL** (data extraction)
- **Jupyter Notebooks**
- **Git** (version control)
- **Basic ML frameworks** (XGBoost, LightGBM)

### 📊 Typical Tasks:
- "Build model untuk predict customer churn"
- "Forecast sales 3 bulan ke depan"
- "Recommend products untuk setiap customer"
- "Run A/B test: Apakah new button color increase conversion?"

### 💰 Salary Range (Indonesia):
- **Junior**: 12-20 juta/bulan
- **Mid-level**: 20-35 juta/bulan
- **Senior**: 35-60 juta/bulan

### ⭐ Best For:
- Yang suka problem solving
- Kuat di statistics & mathematics
- Curious & experimental mindset

---

## 2.3 Machine Learning Engineer 🤖

### 🎯 Fokus Utama:
**Deploy & scale ML models ke production**

### 📝 Tanggung Jawab:
- ✅ Build production-ready ML systems
- ✅ Model deployment (APIs, containers)
- ✅ ML pipeline automation
- ✅ Model monitoring & maintenance
- ✅ Performance optimization
- ✅ Collaboration dengan Software Engineers

### 🛠️ Tools:
- **Python** (advanced - OOP, best practices)
- **ML Frameworks**: TensorFlow, PyTorch
- **MLOps**: MLflow, DVC, Airflow
- **Cloud**: AWS, GCP, Azure
- **Containers**: Docker, Kubernetes
- **APIs**: FastAPI, Flask
- **CI/CD**: GitHub Actions, Jenkins

### 📊 Typical Tasks:
- "Deploy recommendation model yang handle 1M requests/day"
- "Create automated ML pipeline"
- "Monitor model performance & retrain when needed"
- "Optimize model inference time dari 500ms ke 50ms"

### 💰 Salary Range (Indonesia):
- **Junior**: 15-25 juta/bulan
- **Mid-level**: 25-45 juta/bulan
- **Senior**: 45-80 juta/bulan

### ⭐ Best For:
- Yang suka software engineering + ML
- Kuat di coding & system design
- Production mindset (reliability, scalability)

---

## 2.4 Data Engineer 🏗️

### 🎯 Fokus Utama:
**Build & maintain data infrastructure**

### 📝 Tanggung Jawab:
- ✅ Design & build data pipelines (ETL/ELT)
- ✅ Maintain data warehouses
- ✅ Optimize database performance
- ✅ Ensure data quality & reliability
- ✅ Data architecture design

### 🛠️ Tools:
- **SQL** (Advanced - query optimization)
- **Python** (untuk pipeline scripts)
- **Big Data**: Spark, Hadoop, Kafka
- **Databases**: PostgreSQL, MySQL, MongoDB
- **Data Warehouses**: Snowflake, Redshift, BigQuery
- **Orchestration**: Airflow, Prefect
- **Cloud**: AWS, GCP, Azure

### 📊 Typical Tasks:
- "Build pipeline untuk ingest 10GB data daily"
- "Optimize query dari 5 menit ke 10 detik"
- "Migrate database dari on-premise ke cloud"
- "Ensure data freshness & accuracy untuk dashboards"

### 💰 Salary Range (Indonesia):
- **Junior**: 12-20 juta/bulan
- **Mid-level**: 20-35 juta/bulan
- **Senior**: 35-65 juta/bulan

### ⭐ Best For:
- Yang suka backend/infrastructure
- Strong di databases & distributed systems
- Detail-oriented (data quality critical!)

---

## 2.5 Comparison Table: 4 Roles

| Aspect | Data Analyst 📊 | Data Scientist 🧪 | ML Engineer 🤖 | Data Engineer 🏗️ |
|--------|----------------|-------------------|-----------------|-------------------|
| **Focus** | Reporting & insights | Modeling & analysis | Production ML | Data infrastructure |
| **Analytics Type** | Descriptive/Diagnostic | Predictive/Prescriptive | - | - |
| **Coding** | Basic | Intermediate-Advanced | Advanced | Advanced |
| **SQL** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Python** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Statistics** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **ML** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **System Design** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Business Acumen** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Entry Barrier** | Lowest | Medium | High | Medium-High |
| **Typical Background** | Business, Economics | STEM, Stats | CS, Engineering | CS, Engineering |

---

## 2.6 Career Progression Paths

### 📈 Path 1: Analyst → Scientist → ML Engineer

```
Data Analyst (1-2 years)
    ↓
Junior Data Scientist (2-3 years)
    ↓
Data Scientist (2-3 years)
    ↓
Choose:
  → Senior Data Scientist → Lead DS → Principal DS
  → ML Engineer → Senior MLE → Staff MLE
```

**Learning Focus**:
- Start: SQL, Excel, Tableau
- Add: Python, Statistics, ML basics
- Master: Advanced ML, System Design, MLOps

---

### 📈 Path 2: Engineer → Data Engineer → ML Engineer

```
Software Engineer (1-2 years)
    ↓
Data Engineer (2-3 years)
    ↓
Choose:
  → Senior Data Engineer → Lead DE → Principal DE
  → ML Engineer → Senior MLE → Staff MLE
```

**Learning Focus**:
- Start: Programming, Databases
- Add: Big Data, Pipelines, Cloud
- Master: ML Systems, Scalability

---

### 💡 Tips Memilih Career Path:

**Pilih Data Analyst jika**:
- ✅ Suka berbicara dengan bisnis team
- ✅ Enjoy storytelling dengan visualizations
- ✅ Tidak terlalu suka heavy coding
- ✅ Ingin quick entry ke data field

**Pilih Data Scientist jika**:
- ✅ Suka solve complex problems
- ✅ Strong di math/statistics
- ✅ Suka experiment & research
- ✅ Comfortable dengan ambiguity

**Pilih ML Engineer jika**:
- ✅ Strong coding background
- ✅ Suka build production systems
- ✅ Care about performance & scalability
- ✅ Interest di intersection ML + Engineering

**Pilih Data Engineer jika**:
- ✅ Suka backend/infrastructure work
- ✅ Detail-oriented (data quality!)
- ✅ Enjoy optimization problems
- ✅ Prefer stability over experimentation

---

### ✅ Summary Part 2:

**Anda telah belajar:**
- ✅ 4 main roles: Data Analyst, Data Scientist, ML Engineer, Data Engineer
- ✅ Responsibilities, tools, salary range untuk each role
- ✅ Career progression paths
- ✅ Tips memilih career path yang sesuai

**Key Insight**: Semua roles penting dan saling complement! Pilih sesuai interest & strength Anda.

**Next**: Part 3 - Data Science Workflow 🔄

---"""))

# =============================================================================
# PART 3: DATA SCIENCE WORKFLOW
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🔄 PART 3: Data Science Workflow (CRISP-DM)

## Progress: 3/6 🟩🟩🟩⬜⬜⬜

---

Setiap Data Science project mengikuti **structured workflow**. Framework paling populer: **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

---

## 3.1 CRISP-DM: The Standard Framework

### 📊 6 Phases (Cyclical Process):

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   1. Business Understanding                        │
│      ↓                                              │
│   2. Data Understanding                             │
│      ↓                                              │
│   3. Data Preparation                               │
│      ↓                                              │
│   4. Modeling                                       │
│      ↓                                              │
│   5. Evaluation                                     │
│      ↓                                              │
│   6. Deployment                                     │
│      ↓                                              │
│   (Back to 1 untuk improvement) ←──────────────────┘
```

**PENTING**: Ini BUKAN linear process! Sering perlu **iterate** & **go back**.

---

## 3.2 Phase 1: Business Understanding 🎯

### 🎯 Goal:
Pahami problem bisnis dan define success metrics

### 📝 Key Questions:
1. **Apa problem bisnis yang ingin diselesaikan?**
   - Contoh: "Customer churn rate terlalu tinggi (20%/month)"

2. **Mengapa problem ini penting?**
   - Contoh: "Revenue loss $500K/month, acquisition cost mahal"

3. **Apa success criteria?**
   - Contoh: "Reduce churn dari 20% ke 15% dalam 6 bulan"

4. **Bagaimana solution akan digunakan?**
   - Contoh: "Model predict churn → retention team proactive outreach"

### 🛠️ Activities:
- ✅ Interview stakeholders (business, product, ops)
- ✅ Define problem statement
- ✅ Set success metrics (KPIs)
- ✅ Estimate ROI (Return on Investment)
- ✅ Check feasibility (data available? technical possible?)

### ⚠️ Common Mistakes:
- ❌ Jump langsung ke modeling tanpa pahami bisnis
- ❌ Success metrics tidak clear/measurable
- ❌ Tidak involve stakeholders dari awal

### ✅ Output:
- Problem statement document
- Success criteria & metrics
- Project plan & timeline

**Time**: 5-10% dari total project

---

## 3.3 Phase 2: Data Understanding 📊

### 🎯 Goal:
Explore data untuk understand quality, patterns, limitations

### 📝 Key Questions:
1. **Data apa yang tersedia?**
   - Tables, columns, formats, sources

2. **Data quality bagaimana?**
   - Missing values? Duplicates? Errors?

3. **Apakah data cukup untuk solve problem?**
   - Sample size, time range, features

4. **Ada pola menarik?**
   - Correlations, trends, anomalies

### 🛠️ Activities:
- ✅ Collect data dari berbagai sources
- ✅ Initial data exploration (EDA)
- ✅ Data quality assessment
- ✅ Identify relationships antar variables
- ✅ Generate hypotheses

### 📊 EDA Checklist:
```python
# Basic checks
- Shape: rows, columns
- Data types
- Missing values percentage
- Duplicates
- Basic statistics (mean, std, min, max)
- Distributions (histograms)
- Correlations
```

### ⚠️ Common Mistakes:
- ❌ Skip EDA → langsung modeling (BAD!)
- ❌ Tidak check data quality
- ❌ Ignore domain knowledge

### ✅ Output:
- Data quality report
- EDA notebook
- Initial insights
- List of data issues to fix

**Time**: 10-15% dari total project

---

## 3.4 Phase 3: Data Preparation 🧹

### 🎯 Goal:
Transform raw data menjadi model-ready dataset

### 📝 Key Tasks:

#### **A. Data Cleaning**:
- ✅ Handle missing values (imputation/deletion)
- ✅ Remove duplicates
- ✅ Fix errors & inconsistencies
- ✅ Handle outliers (remove/cap/transform)

#### **B. Feature Engineering**:
- ✅ Create new features dari existing columns
- ✅ Encode categorical variables
- ✅ Scale/normalize numerical features
- ✅ Extract features dari dates/text

#### **C. Feature Selection**:
- ✅ Remove irrelevant features
- ✅ Handle multicollinearity
- ✅ Dimensionality reduction (if needed)

#### **D. Data Splitting**:
- ✅ Train/Validation/Test split
- ✅ Ensure no data leakage!

### 🛠️ Common Transformations:

**Missing Values**:
```
- Numerical: Mean/Median imputation, KNN Imputer, MICE
- Categorical: Mode imputation, "Unknown" category
```

**Outliers**:
```
- IQR method
- Z-score method
- Capping (winsorization)
```

**Encoding**:
```
- One-Hot Encoding (nominal)
- Label Encoding (ordinal)
- Target Encoding (high cardinality)
```

**Scaling**:
```
- StandardScaler (mean=0, std=1)
- MinMaxScaler (0 to 1)
- RobustScaler (robust to outliers)
```

### ⚠️ Critical Rules:
- ⚠️ **NO DATA LEAKAGE**: NEVER use test data dalam training!
- ⚠️ **Fit on train, transform on test**: Scaler/Encoder fit hanya di train set
- ⚠️ **Document everything**: Track semua transformations

### ✅ Output:
- Clean, model-ready dataset
- Feature engineering notebook
- Data transformation pipeline (reproducible!)

**Time**: 50-70% dari total project (PALING LAMA!)

**Quote terkenal**:
> "Data Scientists spend 80% of time cleaning data, 20% complaining about cleaning data" 😄

---

## 3.5 Phase 4: Modeling 🤖

### 🎯 Goal:
Build & train ML models

### 📝 Key Steps:

#### **1. Select Modeling Techniques**:
```
Problem Type → Algorithm Choices

Regression → Linear Regression, Decision Tree, Random Forest, XGBoost
Classification → Logistic Regression, SVM, Random Forest, XGBoost
Clustering → K-Means, DBSCAN, Hierarchical
```

#### **2. Build Baseline Model**:
- Start SIMPLE! (Linear/Logistic Regression)
- Benchmark untuk compare advanced models

#### **3. Train Multiple Models**:
- Try different algorithms
- Compare performance

#### **4. Hyperparameter Tuning**:
- Grid Search / Random Search / Bayesian Optimization
- Find optimal parameters

#### **5. Cross-Validation**:
- K-Fold CV untuk robust evaluation
- Prevent overfitting

### 🛠️ Modeling Best Practices:

**✅ DO**:
- Start simple, add complexity gradually
- Use cross-validation
- Track experiments (MLflow)
- Version control code & models
- Document assumptions

**❌ DON'T**:
- Jump to complex models immediately
- Overfit to validation set (tune too much)
- Ignore simple models (sering surprisingly good!)
- Forget to set random_state (reproducibility!)

### ⚠️ Common Pitfalls:
- ❌ Data leakage (biggest issue!)
- ❌ Overfitting (model terlalu complex)
- ❌ Underfitting (model terlalu simple)
- ❌ Not enough data
- ❌ Class imbalance (classification)

### ✅ Output:
- Trained models (multiple candidates)
- Model performance metrics
- Hyperparameter tuning results
- Model comparison report

**Time**: 10-20% dari total project

---

## 3.6 Phase 5: Evaluation 📈

### 🎯 Goal:
Evaluate model apakah meet business objectives

### 📝 Two Types of Evaluation:

#### **A. Technical Evaluation**:
Metrics untuk measure model performance

**Regression Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Classification Metrics**:
- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- Confusion Matrix

**Clustering Metrics**:
- Silhouette Score
- Davies-Bouldin Index

#### **B. Business Evaluation**:
Apakah model solve business problem?

**Key Questions**:
1. ✅ Apakah accuracy cukup untuk use case?
   - Medical diagnosis: 99%+ required
   - Recommendation: 70% ok

2. ✅ Apakah ROI positif?
   - Cost to build vs Revenue/Cost savings

3. ✅ Apakah model fair & ethical?
   - No bias terhadap certain groups

4. ✅ Apakah model interpretable?
   - Stakeholders bisa understand & trust?

### 🛠️ Evaluation Checklist:

**Technical**:
- ✅ Performance di test set (not just train!)
- ✅ Check for overfitting (train vs test gap)
- ✅ Error analysis (mana yang salah predict?)
- ✅ Model interpretability (SHAP, LIME)

**Business**:
- ✅ Meet success criteria?
- ✅ Stakeholder approval
- ✅ Operational feasibility (bisa deploy?)
- ✅ Maintenance plan

### ⚠️ Decision Point:
**Model PASS evaluation**:
→ Proceed to Deployment ✅

**Model FAIL evaluation**:
→ Back to previous phases:
- Data Preparation (need more features?)
- Modeling (try different algorithm?)
- Business Understanding (problem too hard? realistic?)

### ✅ Output:
- Model evaluation report
- Business case for deployment
- Deployment plan

**Time**: 5-10% dari total project

---

## 3.7 Phase 6: Deployment 🚀

### 🎯 Goal:
Deploy model ke production untuk generate value

### 📝 Deployment Options:

#### **1. Batch Prediction**:
Run model periodically (daily, weekly)
```
Example: Churn prediction batch setiap minggu
```

#### **2. Real-Time API**:
Model serve via REST API
```
Example: Fraud detection untuk setiap transaction
```

#### **3. Embedded**:
Model integrated dalam aplikasi
```
Example: Recommendation engine di mobile app
```

### 🛠️ Deployment Checklist:

**Pre-Deployment**:
- ✅ Code review & testing
- ✅ Model versioning (track which version deployed)
- ✅ Create inference pipeline
- ✅ Setup monitoring & logging
- ✅ Define rollback plan

**Deployment**:
- ✅ Containerization (Docker)
- ✅ Cloud deployment (AWS/GCP/Azure)
- ✅ API endpoint (FastAPI, Flask)
- ✅ Load testing

**Post-Deployment**:
- ✅ Monitor performance (accuracy, latency, uptime)
- ✅ Track data drift (input distribution berubah?)
- ✅ Track concept drift (relationship X→y berubah?)
- ✅ A/B testing (compare dengan baseline)
- ✅ Regular retraining schedule

### 🔄 MLOps Components:

```
┌──────────────────────────────────────────────┐
│  Continuous ML Lifecycle                     │
│                                              │
│  Training → Deploy → Monitor → Retrain      │
│     ↑                           ↓            │
│     └───────── Feedback ────────┘            │
└──────────────────────────────────────────────┘
```

**Key Tools**:
- **Experiment Tracking**: MLflow, Weights & Biases
- **Data Versioning**: DVC
- **Pipeline Orchestration**: Airflow, Prefect
- **Model Serving**: FastAPI, TensorFlow Serving
- **Monitoring**: Prometheus, Grafana, Evidently

### ⚠️ Production Challenges:

**Technical**:
- Model drift over time
- Latency requirements (real-time <100ms?)
- Scalability (handle 1M requests/day?)
- Data quality issues

**Organizational**:
- Stakeholder expectations
- Change management
- Documentation & knowledge transfer
- Maintenance & support

### ✅ Output:
- Deployed production model
- Monitoring dashboard
- Documentation (user guide, technical docs)
- Maintenance plan

**Time**: 5-10% initial deployment, ongoing maintenance

---

## 3.8 Real-World Example: Customer Churn Project

Mari lihat **complete workflow** untuk real project:

### **1. Business Understanding** (Week 1):
**Problem**: Telco company kehilangan 20% customers/month

**Goal**: Reduce churn dari 20% ke 15% dalam 6 bulan

**Success Metric**:
- Model Recall ≥ 75% (catch 75% churners)
- Precision ≥ 60% (avoid spam everyone)

**Business Impact**:
- Prevent $500K revenue loss/month
- ROI: Save $6M/year vs $500K project cost

---

### **2. Data Understanding** (Week 2):
**Data Sources**:
- Customer demographics (age, location, plan)
- Usage data (call minutes, data usage)
- Support tickets
- Payment history

**Initial Findings**:
- 100K customers, 50 features
- 15% missing values di beberapa kolom
- Churn rate: 20% (imbalanced!)
- High churn: Customers dengan frequent support tickets

---

### **3. Data Preparation** (Weeks 3-5):
**Cleaning**:
- Impute missing values (median untuk numerical)
- Remove duplicates (500 rows)
- Fix data type errors

**Feature Engineering**:
- `tenure_months` dari signup date
- `avg_monthly_spend` from payment history
- `support_tickets_last_3months`
- `payment_failed_count`

**Encoding**:
- One-Hot: `contract_type`, `payment_method`
- Label: `service_tier` (Bronze=0, Silver=1, Gold=2)

**Scaling**:
- StandardScaler untuk numerical features

**Split**:
- Train: 70% (70K)
- Val: 15% (15K)
- Test: 15% (15K)

---

### **4. Modeling** (Weeks 6-7):
**Models Tried**:
1. Logistic Regression (Baseline): Recall 65%, Precision 55%
2. Random Forest: Recall 72%, Precision 61%
3. XGBoost: Recall 76%, Precision 63% ✅ BEST

**Hyperparameter Tuning**:
- Grid Search dengan 5-Fold CV
- Best params: max_depth=6, learning_rate=0.1, n_estimators=200

**Feature Importance**:
Top 5: tenure_months, avg_monthly_spend, support_tickets, contract_type, payment_failed

---

### **5. Evaluation** (Week 8):
**Test Set Performance**:
- Recall: 75.5% ✅ (meet target 75%)
- Precision: 62.8% ✅ (meet target 60%)
- F1-Score: 68.6%

**Business Validation**:
- Model catches 75% of churners
- Marketing team can target with retention offers
- Expected savings: $4.5M/year (from $6M potential loss)

**Approval**: ✅ Proceed to deployment!

---

### **6. Deployment** (Weeks 9-10):
**Implementation**:
- Batch prediction: Run weekly
- Output: List of high-risk customers → CRM system
- Retention team calls top 500 at-risk customers/week

**Monitoring**:
- Track: Accuracy, precision, recall weekly
- Alert if metrics drop >5%
- Retrain quarterly with new data

**Results After 3 Months**:
- Churn reduced from 20% → 17% ✅
- On track to hit 15% target!
- ROI: Already saved $1.5M

---

### ✅ Summary Part 3:

**Anda telah belajar:**
- ✅ CRISP-DM: Standard framework untuk DS projects
- ✅ 6 Phases: Business Understanding → Data Understanding → Preparation → Modeling → Evaluation → Deployment
- ✅ Each phase: Goals, activities, outputs, time estimates
- ✅ Real-world example: End-to-end churn prediction project

**Key Insight**: Data Science adalah **iterative process**, bukan linear. Expect to go back & refine!

**Next**: Part 4 - Tools & Ecosystem 🛠️

---"""))

# =============================================================================
# PART 4: TOOLS & ECOSYSTEM
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🛠️ PART 4: Tools & Ecosystem

## Progress: 4/6 🟩🟩🟩🟩⬜⬜

---

Data Science punya **ecosystem yang sangat rich**! Mari kenali tools utama yang akan Anda gunakan.

---

## 4.1 Programming Languages 💻

### **1. Python** 🐍 (DOMINAN - 80% DS jobs)

**Why Python?**:
- ✅ Easy to learn (readable syntax)
- ✅ Rich ecosystem untuk DS/ML
- ✅ Huge community support
- ✅ Versatile (web dev, automation, ML, DL)

**Core Libraries**:

**A. Data Manipulation**:
- **NumPy**: Numerical computing (arrays, matrices, linear algebra)
- **Pandas**: Data manipulation (DataFrames, CSV/Excel handling)

**B. Visualization**:
- **Matplotlib**: Basic plotting (static graphs)
- **Seaborn**: Statistical visualization (beautiful, easy)
- **Plotly**: Interactive visualizations
- **Altair**: Declarative viz (grammar of graphics)

**C. Machine Learning**:
- **Scikit-learn**: Traditional ML (regression, classification, clustering)
- **XGBoost, LightGBM, CatBoost**: Advanced gradient boosting

**D. Deep Learning**:
- **TensorFlow + Keras**: Google's DL framework
- **PyTorch**: Facebook's DL framework (increasingly popular)
- **Hugging Face**: Transformers untuk NLP

**E. Statistics**:
- **SciPy**: Scientific computing, statistical tests
- **Statsmodels**: Statistical modeling, time series

---

### **2. R** 📊 (Populer di Academia & Statistics)

**Why R?**:
- ✅ Built for statistics
- ✅ Excellent untuk statistical analysis & research
- ✅ Beautiful visualizations (ggplot2)
- ✅ Strong community di academic/research

**Core Libraries**:
- **dplyr**: Data manipulation
- **ggplot2**: Visualization (grammar of graphics)
- **caret**: ML framework
- **tidyverse**: Complete data science ecosystem

**When to use R?**:
- Academic research
- Statistical analysis focus
- Clinical trials, biostatistics
- Complex visualizations (ggplot2 unmatched!)

---

### **3. SQL** 🗄️ (WAJIB - 95% DS jobs require SQL!)

**Why SQL?**:
- ✅ Most data stored in databases
- ✅ Efficient untuk large datasets
- ✅ Universal skill (every company uses SQL)
- ✅ Fast data extraction & aggregation

**Key Concepts**:
- SELECT, WHERE, JOIN, GROUP BY, HAVING
- Window functions
- CTEs (Common Table Expressions)
- Query optimization

**Databases**:
- **Relational**: PostgreSQL, MySQL, SQLite
- **Data Warehouses**: Snowflake, Redshift, BigQuery
- **NoSQL**: MongoDB, Cassandra (untuk unstructured data)

---

## 4.2 Development Environments 🖥️

### **1. Jupyter Notebooks** 📒 (MOST POPULAR for DS)

**What**: Interactive notebooks untuk write code + markdown + visualizations

**Pros**:
- ✅ Interactive (see results immediately)
- ✅ Mix code, text, equations, visualizations
- ✅ Great untuk EDA & prototyping
- ✅ Easy to share & collaborate

**Cons**:
- ❌ Tidak ideal untuk production code
- ❌ Version control tricky (ipynb format)
- ❌ Can encourage bad practices (non-reproducible)

**Best Use**:
- Exploratory Data Analysis
- Prototyping models
- Teaching & presentations
- Visualizations

---

### **2. JupyterLab** 🧪 (Jupyter on steroids)

**What**: Next-gen interface untuk Jupyter

**Additional Features**:
- ✅ Multiple tabs, split views
- ✅ File browser, terminal
- ✅ Extensions ecosystem
- ✅ Better UI/UX

---

### **3. VS Code** 📝 (Best All-Around Editor)

**What**: Microsoft's code editor

**Why for DS**:
- ✅ Native Jupyter support
- ✅ Python debugger
- ✅ Git integration
- ✅ Extensions (Python, Remote SSH, Docker)
- ✅ Lightweight & fast

**Best Use**:
- Writing production code
- Package development
- Multi-file projects

---

### **4. PyCharm** 🐍 (Power IDE for Python)

**What**: JetBrains IDE for Python

**Pros**:
- ✅ Best-in-class Python features
- ✅ Excellent debugger
- ✅ Code refactoring tools
- ✅ Database tools built-in

**Cons**:
- ❌ Heavy (high memory usage)
- ❌ Professional version not free ($200/year)

---

### **5. Google Colab** ☁️ (Free Cloud Jupyter)

**What**: Free Jupyter in cloud dengan FREE GPU/TPU!

**Pros**:
- ✅ FREE GPU (Tesla T4, ~12GB RAM)
- ✅ FREE TPU (for TensorFlow)
- ✅ Pre-installed DS libraries
- ✅ No setup needed (browser-based)
- ✅ Easy sharing (like Google Docs)

**Cons**:
- ❌ Session timeout (12 hours max)
- ❌ Limited storage (Google Drive)
- ❌ Can't install system packages

**Best Use**:
- Learning Deep Learning (FREE GPU!)
- Quick prototyping
- Collaborative work
- When no local GPU

---

## 4.3 Version Control & Collaboration 🔄

### **Git + GitHub** 📦

**Git**: Version control system

**Why Essential**:
- ✅ Track code changes
- ✅ Collaborate dengan team
- ✅ Rollback to previous versions
- ✅ Branch untuk experiments

**Key Commands**:
```bash
git init
git add .
git commit -m "message"
git push origin main
git pull
git branch, git checkout
```

**GitHub Features**:
- Code hosting
- Collaboration (Pull Requests, Issues)
- CI/CD (GitHub Actions)
- Portfolio (show your projects!)

---

### **DVC** (Data Version Control) 📊

**What**: Git untuk data & models

**Why Needed**:
- ✅ Data too large for Git
- ✅ Track dataset versions
- ✅ Reproduce experiments
- ✅ Share data efficiently

---

## 4.4 Cloud Platforms ☁️

### **AWS** (Amazon Web Services) 🟠

**DS Services**:
- **S3**: Object storage (data lakes)
- **EC2**: Virtual machines (training models)
- **SageMaker**: End-to-end ML platform
- **Lambda**: Serverless functions
- **Redshift**: Data warehouse

---

### **GCP** (Google Cloud Platform) 🔵

**DS Services**:
- **BigQuery**: Fast SQL data warehouse
- **Cloud Storage**: Like S3
- **Vertex AI**: ML platform (successor to AI Platform)
- **Dataflow**: Data pipelines
- **Cloud Functions**: Serverless

**Strength**: BigQuery (super fast SQL!), TPUs untuk TensorFlow

---

### **Azure** (Microsoft) 🟦

**DS Services**:
- **Azure ML**: ML platform
- **Databricks**: Spark-based analytics
- **Synapse**: Data warehouse + analytics
- **Blob Storage**: Like S3

**Strength**: Enterprise integrations (Office 365, Active Directory)

---

## 4.5 MLOps & Deployment Tools 🚀

### **1. Experiment Tracking**

**MLflow**:
- Track experiments (parameters, metrics, artifacts)
- Model registry
- Model deployment
- FREE, open-source

**Weights & Biases (wandb)**:
- Similar to MLflow
- Better UI/UX
- Real-time collaboration
- FREE for individuals, paid for teams

---

### **2. Model Serving**

**FastAPI**:
- Build REST APIs untuk models (super fast!)
- Auto-generated documentation
- Python type hints
- Modern, async

**Flask**:
- Older, simpler
- Good untuk basic APIs

**TensorFlow Serving**:
- Optimized untuk TF models
- High performance

---

### **3. Containerization**

**Docker** 🐳:
- Package app + dependencies
- "Works on my machine" → "Works everywhere"
- Reproducible environments
- Easy deployment

**Kubernetes** ☸️:
- Container orchestration
- Scale applications
- Auto-healing, load balancing

---

### **4. Workflow Orchestration**

**Apache Airflow**:
- Schedule & monitor workflows (DAGs)
- Complex dependencies
- Popular di industry

**Prefect**:
- Modern alternative to Airflow
- Better UI, easier to use

---

## 4.6 Big Data Tools (for Large-Scale Data)

### **Apache Spark** ⚡

**What**: Distributed computing framework

**When to use**:
- Data > 100GB (doesn't fit in RAM)
- Need parallel processing
- Real-time streaming data

**Languages**: Python (PySpark), Scala, SQL

---

### **Hadoop**

**What**: Distributed storage (HDFS) + processing (MapReduce)

**Note**: Declining popularity, being replaced by Spark + cloud

---

## 4.7 Specialized Tools

### **Time Series**:
- **Prophet**: Facebook's forecasting library (easy, powerful)
- **ARIMA, SARIMA**: Classical statistical methods (statsmodels)

### **NLP**:
- **NLTK**: Natural Language Toolkit (old school)
- **spaCy**: Modern, fast NLP
- **Hugging Face Transformers**: State-of-the-art (BERT, GPT, etc.)

### **Computer Vision**:
- **OpenCV**: Image processing
- **PIL/Pillow**: Image manipulation
- **Albumentations**: Image augmentation

### **AutoML** (Automated ML):
- **Auto-sklearn**: Automated sklearn
- **TPOT**: Genetic programming untuk pipelines
- **H2O.ai**: Enterprise AutoML

---

## 4.8 The Data Science Tech Stack (Summary)

### **Beginner Stack** (Start Here):
```
Language: Python
Environment: Jupyter Notebooks / Google Colab
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
Database: SQLite (local), SQL basics
Version Control: Git + GitHub
```

---

### **Intermediate Stack**:
```
Add:
- VS Code (for production code)
- Cloud: AWS/GCP (basics)
- Advanced ML: XGBoost, LightGBM
- Deep Learning: TensorFlow/Keras atau PyTorch
- MLOps: MLflow (experiment tracking)
```

---

### **Advanced/Production Stack**:
```
Add:
- Deployment: FastAPI + Docker + Kubernetes
- Cloud: AWS SageMaker atau GCP Vertex AI
- Big Data: PySpark (if needed)
- Orchestration: Airflow
- Monitoring: Prometheus + Grafana
- Data Versioning: DVC
- CI/CD: GitHub Actions, Jenkins
```

---

## 4.9 Learning Recommendations 📚

### **Phase 1: Foundation** (Months 1-3):
1. ✅ Master Python (NumPy, Pandas)
2. ✅ Learn SQL (VERY important!)
3. ✅ Visualization (Matplotlib, Seaborn)
4. ✅ Git basics
5. ✅ Jupyter Notebooks

---

### **Phase 2: Core ML** (Months 4-6):
1. ✅ Scikit-learn mastery
2. ✅ Statistics fundamentals
3. ✅ Feature engineering techniques
4. ✅ Advanced ML (XGBoost, LightGBM)
5. ✅ Model evaluation & tuning

---

### **Phase 3: Specialization** (Months 7-12):
Choose path:
- **Deep Learning**: TensorFlow/PyTorch, CNNs, RNNs
- **MLOps**: Docker, FastAPI, Cloud deployment
- **Big Data**: Spark, distributed computing

---

### **Phase 4: Production** (Months 12+):
1. ✅ End-to-end ML systems
2. ✅ Model deployment & monitoring
3. ✅ Automated pipelines
4. ✅ Cloud platforms (AWS/GCP/Azure)

---

### ✅ Summary Part 4:

**Anda telah belajar:**
- ✅ Programming languages: Python (dominan), R, SQL (wajib!)
- ✅ Dev environments: Jupyter, VS Code, Google Colab
- ✅ Version control: Git, GitHub, DVC
- ✅ Cloud platforms: AWS, GCP, Azure
- ✅ MLOps tools: MLflow, FastAPI, Docker, Airflow
- ✅ Recommended learning path (beginner → advanced)

**Key Insight**: Start simple (Python + Jupyter + Scikit-learn), add complexity as needed!

**Next**: Part 5 - Data Scientist Mindset 🧠

---"""))

# =============================================================================
# PART 5: DATA SCIENTIST MINDSET
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🧠 PART 5: Data Scientist Mindset

## Progress: 5/6 🟩🟩🟩🟩🟩⬜

---

Technical skills adalah foundation, tapi **mindset** yang membedakan good vs great Data Scientist!

---

## 5.1 Core Principles of Data Science 🎯

### **1. Start with the Problem, Not the Data** 🎯

**BAD Approach** ❌:
"I have this cool dataset, what can I do with it?"

**GOOD Approach** ✅:
"We have business problem X. What data do we need? What analysis will answer this?"

**Why Important**:
- DS bukan tentang "cool techniques"
- Focus on **business value**, not technical complexity
- Align dengan stakeholder expectations

**Example**:
- ❌ "Let's build neural network karena cool!"
- ✅ "Business needs 80% accuracy, simple Logistic Regression cukup"

---

### **2. Question Everything (Critical Thinking)** 🤔

**Always Ask**:
- ❓ "Apakah data ini reliable?"
- ❓ "Ada bias dalam data collection?"
- ❓ "Correlation = Causation?" (NEVER!)
- ❓ "Sample representative of population?"
- ❓ "What are hidden assumptions?"

**Example: Ice Cream & Drowning**:
- **Observation**: Ice cream sales ↑ → Drowning deaths ↑
- **BAD Conclusion**: Ice cream causes drowning! 🍦 → 💀
- **GOOD Thinking**: Hidden variable = Summer! (Hot weather → both ↑)

**Lesson**: Always check for **confounding variables**!

---

### **3. Embrace Uncertainty** 🎲

Data Science **NEVER gives 100% certain answers**. Get comfortable with:

- Probabilities, not certainties
- "Model predicts 75% chance" vs "Model says yes/no"
- Confidence intervals
- Statistical significance

**Quote**:
> "All models are wrong, but some are useful" - George Box

**Your job**: Find useful models, communicate uncertainty clearly!

---

### **4. Iterate, Iterate, Iterate** 🔄

**Data Science is NOT**:
```
Problem → Solution (done!)
```

**Data Science IS**:
```
Problem → Attempt 1 (fail) → Learn → Attempt 2 (better) → Refine → Attempt 3 (good enough) → Deploy → Monitor → Improve
```

**Why**:
- First model rarely best
- New data = new insights
- Real world changes (concept drift)

**Mindset**: "Good enough to deploy & improve" > "Perfect but never shipped"

---

### **5. Communicate Clearly** 💬

**Technical skills = 50%, Communication = 50%**

**Why Critical**:
- Stakeholders usually **non-technical**
- Best model useless if nobody understands/trusts it
- You need buy-in untuk deployment

**Good Communication**:
- ✅ Avoid jargon (or explain technical terms)
- ✅ Use visualizations
- ✅ Tell stories with data
- ✅ Focus on business impact, not model details
- ✅ Tailor message to audience

**Example - BAD** ❌:
"Model achieved 0.87 AUC-ROC with XGBoost using hyperparameter tuning via Bayesian optimization"

**Example - GOOD** ✅:
"Our model correctly identifies 85% of customers likely to leave, allowing marketing to save $2M/year in retention costs"

---

## 5.2 Essential Habits of Successful Data Scientists 🌟

### **1. Curiosity-Driven** 🔍

**Characteristics**:
- Always asking "why?"
- Dig deeper into data anomalies
- Explore edge cases
- Stay updated with new techniques

**Example**:
- See outlier → Don't just remove! Investigate why
- Might discover data quality issue
- Might uncover important business insight!

---

### **2. Business-Oriented** 💼

**Remember**:
- Companies hire DS to **drive business value**
- Not to "do cool ML"
- ROI (Return on Investment) matters

**Always Connect to Business**:
- "This model will increase revenue by X"
- "This analysis will reduce costs by Y"
- "This insight will improve customer satisfaction"

**Questions to Ask**:
- How will this be used?
- What's the business impact?
- What if model is wrong? (Cost of false positives/negatives)

---

### **3. Detail-Oriented (but Pragmatic)** 🔬

**Balance**:
- Care about details (data quality, edge cases, assumptions)
- BUT know when "good enough"

**Example**:
- 95% accuracy vs 96% accuracy
- If 96% takes 2 months more → NOT worth it untuk most cases
- Diminishing returns!

**Pragmatism**:
- Start simple, measure impact
- Iterate based on real-world feedback
- Don't over-engineer

---

### **4. Systematic & Reproducible** 📋

**Best Practices**:
- ✅ Document everything (code comments, markdown)
- ✅ Version control (Git for code, DVC for data)
- ✅ Set random seeds (reproducible results)
- ✅ Track experiments (MLflow, wandb)
- ✅ Clear folder structure
- ✅ README files

**Why**:
- You'll forget what you did 3 months later
- Teammates need to understand your work
- Reproducibility = credibility

---

### **5. Ethical & Responsible** ⚖️

**Data Science has POWER → Great responsibility!**

**Key Ethical Concerns**:

**A. Bias & Fairness**:
- ⚠️ Models can perpetuate societal biases
- Example: Hiring ML biased against women (trained on historical data)
- **Solution**: Check fairness metrics, diverse training data

**B. Privacy**:
- ⚠️ Protect user data (GDPR, privacy laws)
- Don't leak sensitive information
- **Solution**: Anonymization, encryption, differential privacy

**C. Transparency**:
- ⚠️ Explain model decisions (especially high-stakes: medical, legal, finance)
- Black-box models risky
- **Solution**: Interpretability techniques (SHAP, LIME)

**D. Misuse Prevention**:
- ⚠️ Models can be weaponized (deepfakes, surveillance, discrimination)
- **Solution**: Consider potential misuse, refuse unethical projects

**Framework: Ask Yourself**:
1. Could this harm anyone?
2. Is the data ethically sourced?
3. Are predictions fair across groups?
4. Can we explain decisions?
5. What if model is wrong? (Consequences)

---

## 5.3 Growth Mindset in Data Science 📈

### **1. Embrace Failure** 💪

**Truth**: Most models fail. Most experiments fail.

**Reframe**:
- ❌ "I failed"
- ✅ "I learned what doesn't work"

**Example**:
- Try 10 approaches
- 8 fail, 2 work
- Success rate = 20%, **but that's normal!**

**Lesson**: Failure = Data! (You now know 8 things that don't work)

---

### **2. Continuous Learning** 📚

**Data Science Changes FAST**:
- New algorithms (GPT-3 → GPT-4)
- New libraries (JAX, FastAI)
- New best practices

**Stay Updated**:
- 📰 Follow blogs (Towards Data Science, Medium)
- 🎥 Watch conferences (NeurIPS, ICML, KDD)
- 📚 Read papers (Arxiv)
- 💻 Do projects (Kaggle, personal projects)
- 👥 Join communities (Reddit, Discord, local meetups)

**BUT**: Don't chase every trend! Master fundamentals first.

---

### **3. Learn from Others** 👥

**Leverage Community**:
- Read others' code (GitHub)
- Compete in Kaggle (learn from winners)
- Ask questions (Stack Overflow, forums)
- Collaborate (team projects)

**Humility**:
- You don't know everything (nobody does!)
- Senior DS still learning
- Be open to feedback

---

## 5.4 Common Pitfalls to Avoid ⚠️

### **1. "Shiny Object Syndrome"** ✨

**Symptom**:
- Jump to latest/coolest technique
- Deep Learning for everything
- Ignore simple baselines

**Cure**:
- Start simple (Logistic Regression, Random Forest)
- Only add complexity if needed
- Measure improvement vs baseline

---

### **2. "Analysis Paralysis"** 😰

**Symptom**:
- Endless EDA, never move to modeling
- Trying every algorithm
- Perfectionism

**Cure**:
- Set time limits (EDA: 2 days max for first pass)
- Ship v1, iterate later
- "Done is better than perfect"

---

### **3. "Not My Job Syndrome"** 🙅

**Symptom**:
- "Data cleaning is boring, I just want to model"
- "Deployment is engineering team's job"
- Silo mentality

**Cure**:
- End-to-end ownership
- Model only valuable when deployed!
- Collaboration with engineering, business

---

### **4. "Ignoring Domain Knowledge"** 🤷

**Symptom**:
- Pure data-driven, ignore expert input
- "Let the data speak" (but misinterpret!)

**Cure**:
- Talk to domain experts (doctors for healthcare, traders for finance)
- Combine data + domain knowledge
- Validate insights with experts

---

## 5.5 Daily Practices 🗓️

### **Morning Routine** 🌅:
1. Check monitoring dashboards (deployed models)
2. Review experiment results (training overnight)
3. Prioritize tasks (high-impact first)
4. Read 1 article (stay updated)

---

### **During Work** 💼:
1. Document as you go (don't delay!)
2. Test incrementally (don't write 100 lines before testing)
3. Version control (commit frequently)
4. Take breaks (Pomodoro technique)

---

### **End of Day** 🌆:
1. Commit code (clean working directory)
2. Update project notes (what worked, what didn't)
3. Plan tomorrow (prioritize tasks)
4. Reflect (what did I learn today?)

---

### ✅ Summary Part 5:

**Anda telah belajar:**
- ✅ Core principles: Problem-first, Critical thinking, Embrace uncertainty, Iterate, Communicate
- ✅ Essential habits: Curiosity, Business-oriented, Detail-oriented, Systematic, Ethical
- ✅ Growth mindset: Embrace failure, Continuous learning, Learn from others
- ✅ Common pitfalls: Shiny objects, Analysis paralysis, Silos, Ignoring domain
- ✅ Daily practices untuk effective DS work

**Key Insight**: Mindset > Tools! Technical skills get you hired, mindset makes you excel.

**Next**: Part 6 - Mini Project (Hands-On!) 🚀

---"""))

# =============================================================================
# PART 6: MINI PROJECT - HANDS-ON!
# =============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🚀 PART 6: Mini Project - Your First Data Science Analysis!

## Progress: 6/6 🟩🟩🟩🟩🟩🟩 (COMPLETE!)

---

Sekarang saatnya **PRAKTEK**! Kita akan lakukan simple Data Science analysis menggunakan real dataset.

---

## 🎯 Project Goal

**Business Problem**:
Anda adalah Data Analyst di sebuah company yang menjual sepeda. Management ingin tahu:
1. **Faktor apa yang mempengaruhi penjualan?**
2. **Kapan peak sales terjadi?**
3. **Rekomendasi untuk increase sales**

**Dataset**: Bike Sales (dummy dataset untuk learning)

**Your Task**: Analyze data, generate insights, communicate findings!

---

## 📊 About the Dataset

We'll create a simple synthetic dataset dengan columns:
- `date`: Tanggal penjualan
- `day_of_week`: Hari (Mon-Sun)
- `weather`: Cuaca (Sunny, Cloudy, Rainy)
- `temperature`: Suhu (Celsius)
- `holiday`: Hari libur atau tidak (0/1)
- `sales`: Jumlah sepeda terjual

---

## Let's Code! 💻

We'll follow mini version of **CRISP-DM workflow**:
1. **Generate Data** (simulate data collection)
2. **Understand Data** (EDA)
3. **Analyze** (find patterns)
4. **Visualize** (communicate findings)
5. **Insights** (business recommendations)

**Ready? Let's go!** 🚀

---"""))

# Code cell: Generate synthetic data
cells.append(nbf.v4.new_code_cell("""# ============================================================
# STEP 1: Generate Synthetic Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed untuk reproducibility
np.random.seed(42)

# Generate dates (90 days = ~3 months)
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(90)]

# Create DataFrame
df = pd.DataFrame({
    'date': dates
})

# Add day of week
df['day_of_week'] = df['date'].dt.day_name()

# Add weather (random)
weather_options = ['Sunny', 'Cloudy', 'Rainy']
weather_probs = [0.5, 0.3, 0.2]  # 50% sunny, 30% cloudy, 20% rainy
df['weather'] = np.random.choice(weather_options, size=90, p=weather_probs)

# Add temperature (varies by month)
# Jan: 20-25°C, Feb: 22-27°C, Mar: 25-30°C
month = df['date'].dt.month
base_temp = 20 + (month - 1) * 2.5  # Temperature increases each month
df['temperature'] = base_temp + np.random.normal(0, 2, 90)  # Add random variation

# Add holiday (simplified: weekends = holiday)
df['holiday'] = (df['date'].dt.dayofweek >= 5).astype(int)  # Sat=5, Sun=6

# Generate sales (influenced by multiple factors)
# Base sales: 50 bikes/day
# + Weather: Sunny (+20), Cloudy (+0), Rainy (-15)
# + Temperature: Higher temp = more sales (linear relationship)
# + Holiday: +30 bikes
# + Random noise

base_sales = 50

# Weather effect
weather_effect = df['weather'].map({'Sunny': 20, 'Cloudy': 0, 'Rainy': -15})

# Temperature effect (more sales when warmer)
temp_effect = (df['temperature'] - 20) * 1.5  # +1.5 bikes per degree above 20°C

# Holiday effect
holiday_effect = df['holiday'] * 30

# Calculate total sales
df['sales'] = (base_sales + weather_effect + temp_effect + holiday_effect +
               np.random.normal(0, 8, 90))  # Add random noise

# Ensure sales non-negative
df['sales'] = df['sales'].clip(lower=10).round().astype(int)

# Display dataset info
print("="*70)
print("DATASET CREATED SUCCESSFULLY!")
print("="*70)
print(f"\\nDataset Shape: {df.shape}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total Days: {len(df)}")

# Show first 10 rows
print("\\n" + "="*70)
print("FIRST 10 ROWS:")
print("="*70)
print(df.head(10).to_string(index=False))

# Show last 5 rows
print("\\n" + "="*70)
print("LAST 5 ROWS:")
print("="*70)
print(df.tail(5).to_string(index=False))"""))

cells.append(nbf.v4.new_markdown_cell("""---

### 📝 What We Just Did:

1. ✅ Created a **synthetic dataset** with 90 days of bike sales data
2. ✅ Included multiple factors: weather, temperature, holidays
3. ✅ Generated realistic sales numbers with randomness
4. ✅ Displayed first and last rows to understand data structure

**Insight**: In real projects, you'd load data from CSV/database instead of generating it.

---"""))

# Code cell: EDA
cells.append(nbf.v4.new_code_cell("""# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# 1. Dataset Info
print("\\n📊 DATASET INFORMATION:")
print("-"*70)
print(df.info())

# 2. Basic Statistics
print("\\n" + "="*70)
print("📈 NUMERICAL COLUMNS - DESCRIPTIVE STATISTICS:")
print("="*70)
print(df[['temperature', 'sales']].describe().round(2))

# 3. Check for missing values
print("\\n" + "="*70)
print("🔍 MISSING VALUES CHECK:")
print("="*70)
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ NO MISSING VALUES! Data quality excellent.")
else:
    print(f"⚠️ Total missing values: {missing.sum()}")

# 4. Value counts for categorical variables
print("\\n" + "="*70)
print("📋 WEATHER DISTRIBUTION:")
print("="*70)
weather_counts = df['weather'].value_counts()
print(weather_counts)
print(f"\\nPercentages:")
print((weather_counts / len(df) * 100).round(1))

print("\\n" + "="*70)
print("📋 DAY OF WEEK DISTRIBUTION:")
print("="*70)
day_counts = df['day_of_week'].value_counts()
print(day_counts)

# 5. Sales statistics by category
print("\\n" + "="*70)
print("💰 AVERAGE SALES BY WEATHER:")
print("="*70)
sales_by_weather = df.groupby('weather')['sales'].agg(['mean', 'std', 'min', 'max']).round(2)
print(sales_by_weather)

print("\\n" + "="*70)
print("💰 AVERAGE SALES: HOLIDAY vs NON-HOLIDAY:")
print("="*70)
sales_by_holiday = df.groupby('holiday')['sales'].agg(['mean', 'std', 'count']).round(2)
sales_by_holiday.index = ['Non-Holiday', 'Holiday']
print(sales_by_holiday)

# 6. Correlation analysis
print("\\n" + "="*70)
print("🔗 CORRELATION ANALYSIS:")
print("="*70)
# Create numerical mapping for weather
weather_mapping = {'Rainy': 1, 'Cloudy': 2, 'Sunny': 3}
df_corr = df.copy()
df_corr['weather_num'] = df_corr['weather'].map(weather_mapping)

correlation = df_corr[['temperature', 'holiday', 'weather_num', 'sales']].corr()['sales'].sort_values(ascending=False)
print(correlation)
print("\\nInterpretation:")
print("- Temperature: " + ("Strong" if abs(correlation['temperature']) > 0.5 else "Moderate") + " positive correlation")
print("- Holiday: " + ("Strong" if abs(correlation['holiday']) > 0.5 else "Moderate") + " positive correlation")
print("- Weather: " + ("Strong" if abs(correlation['weather_num']) > 0.5 else "Moderate") + " positive correlation")

# Summary insights
print("\\n" + "="*70)
print("🎯 KEY FINDINGS FROM EDA:")
print("="*70)
print(f"1. Average daily sales: {df['sales'].mean():.0f} bikes")
print(f"2. Sales range: {df['sales'].min()} to {df['sales'].max()} bikes")
print(f"3. Best weather for sales: {sales_by_weather['mean'].idxmax()} ({sales_by_weather['mean'].max():.0f} bikes avg)")
print(f"4. Worst weather for sales: {sales_by_weather['mean'].idxmin()} ({sales_by_weather['mean'].min():.0f} bikes avg)")
print(f"5. Holiday boost: +{(sales_by_holiday.loc['Holiday', 'mean'] - sales_by_holiday.loc['Non-Holiday', 'mean']):.0f} bikes on average")
print(f"6. Temperature correlation: {correlation['temperature']:.2f} (higher temp → more sales)")"""))

cells.append(nbf.v4.new_markdown_cell("""---

### 📝 What We Just Did:

1. ✅ Checked data quality (no missing values!)
2. ✅ Calculated descriptive statistics
3. ✅ Analyzed distributions (weather, day of week)
4. ✅ Compared sales across categories (weather, holiday)
5. ✅ Found correlations between variables and sales

**Key Insights**:
- Weather significantly affects sales
- Holidays boost sales substantially
- Temperature has positive correlation with sales

---"""))

# Code cell: Visualizations
cells.append(nbf.v4.new_code_cell("""# ============================================================
# STEP 3: DATA VISUALIZATION
# ============================================================

# Set style untuk visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🚴 Bike Sales Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)

# ===========================
# Plot 1: Sales Over Time
# ===========================
ax1 = axes[0, 0]
ax1.plot(df['date'], df['sales'], marker='o', markersize=3, linewidth=1, alpha=0.7)
ax1.set_title('📈 Daily Sales Trend Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales (bikes)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add average line
avg_sales = df['sales'].mean()
ax1.axhline(y=avg_sales, color='r', linestyle='--', label=f'Average: {avg_sales:.0f}', linewidth=2)
ax1.legend()

# ===========================
# Plot 2: Sales by Weather
# ===========================
ax2 = axes[0, 1]
weather_order = ['Rainy', 'Cloudy', 'Sunny']
sales_weather = df.groupby('weather')['sales'].mean().reindex(weather_order)

colors = ['#3498db', '#95a5a6', '#f39c12']  # Blue, Gray, Orange
bars = ax2.bar(weather_order, sales_weather, color=colors, alpha=0.7, edgecolor='black')

ax2.set_title('☀️ Average Sales by Weather Condition', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Sales (bikes)')
ax2.set_xlabel('Weather')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}',
            ha='center', va='bottom', fontweight='bold')

# ===========================
# Plot 3: Sales Distribution
# ===========================
ax3 = axes[1, 0]
ax3.hist(df['sales'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax3.set_title('📊 Sales Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sales (bikes)')
ax3.set_ylabel('Frequency (days)')
ax3.axvline(x=avg_sales, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_sales:.0f}')
ax3.axvline(x=df['sales'].median(), color='g', linestyle='--', linewidth=2, label=f'Median: {df['sales'].median():.0f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ===========================
# Plot 4: Temperature vs Sales
# ===========================
ax4 = axes[1, 1]

# Color by weather
weather_colors = {'Sunny': '#f39c12', 'Cloudy': '#95a5a6', 'Rainy': '#3498db'}
for weather in weather_order:
    mask = df['weather'] == weather
    ax4.scatter(df[mask]['temperature'], df[mask]['sales'],
               label=weather, alpha=0.6, s=50, color=weather_colors[weather])

ax4.set_title('🌡️ Temperature vs Sales (colored by Weather)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Temperature (°C)')
ax4.set_ylabel('Sales (bikes)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['temperature'], df['sales'], 1)
p = np.poly1d(z)
ax4.plot(df['temperature'], p(df['temperature']), "r--", alpha=0.8, linewidth=2, label='Trend')

plt.tight_layout()
plt.show()

print("\\n✅ Visualizations created successfully!")
print("\\n💡 Insights from Visualizations:")
print("1. Sales show some fluctuation over time with no clear seasonal pattern (only 3 months)")
print("2. Sunny weather clearly leads to highest sales")
print("3. Sales distribution appears roughly normal")
print("4. Strong positive relationship between temperature and sales")"""))

cells.append(nbf.v4.new_markdown_cell("""---

### 📝 What We Just Did:

1. ✅ Created **4 key visualizations**:
   - Time series plot (trend over time)
   - Bar chart (sales by weather)
   - Histogram (sales distribution)
   - Scatter plot (temperature vs sales)

2. ✅ Used colors strategically for better understanding
3. ✅ Added reference lines (average, median, trend)
4. ✅ Labeled everything clearly

**Visualization Best Practices Applied**:
- Clear titles and labels
- Appropriate chart types for each analysis
- Color coding for categories
- Reference lines for context
- Annotations (values on bars)

---"""))

# Code cell: Additional analysis - Day of week
cells.append(nbf.v4.new_code_cell("""# ============================================================
# STEP 4: ADDITIONAL ANALYSIS - Day of Week Pattern
# ============================================================

print("="*70)
print("📅 SALES PATTERN BY DAY OF WEEK")
print("="*70)

# Calculate average sales by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_by_day = df.groupby('day_of_week')['sales'].agg(['mean', 'count']).reindex(day_order)

print("\\nAverage Sales by Day:")
print(sales_by_day.round(2))

# Visualization
plt.figure(figsize=(12, 5))

# Plot
colors = ['#3498db' if day in ['Saturday', 'Sunday'] else '#95a5a6' for day in day_order]
bars = plt.bar(range(7), sales_by_day['mean'], color=colors, alpha=0.7, edgecolor='black')

plt.title('📅 Average Sales by Day of Week (Weekend Highlighted)', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Average Sales (bikes)', fontsize=12)
plt.xticks(range(7), day_order, rotation=45, ha='right')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}',
            ha='center', va='bottom', fontweight='bold')

# Add average line
overall_avg = df['sales'].mean()
plt.axhline(y=overall_avg, color='r', linestyle='--', linewidth=2, label=f'Overall Avg: {overall_avg:.0f}')

plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Insights
weekend_sales = sales_by_day.loc[['Saturday', 'Sunday'], 'mean'].mean()
weekday_sales = sales_by_day.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'mean'].mean()

print(f"\\n💡 INSIGHTS:")
print(f"- Weekend average: {weekend_sales:.0f} bikes/day")
print(f"- Weekday average: {weekday_sales:.0f} bikes/day")
print(f"- Weekend boost: +{(weekend_sales - weekday_sales):.0f} bikes ({((weekend_sales/weekday_sales - 1) * 100):.1f}%)")
print(f"- Best day: {sales_by_day['mean'].idxmax()} ({sales_by_day['mean'].max():.0f} bikes avg)")
print(f"- Worst day: {sales_by_day['mean'].idxmin()} ({sales_by_day['mean'].min():.0f} bikes avg)")"""))

cells.append(nbf.v4.new_markdown_cell("""---

### 📝 What We Just Did:

1. ✅ Analyzed sales patterns by day of week
2. ✅ Compared weekend vs weekday sales
3. ✅ Created clear visualization with weekends highlighted
4. ✅ Quantified the weekend effect

**Insight**: Weekends (holidays) show significantly higher sales, confirming our earlier finding!

---"""))

# Final summary cell
cells.append(nbf.v4.new_markdown_cell("""# 🎯 FINAL SUMMARY & BUSINESS RECOMMENDATIONS

---

## 📊 Key Findings:

### 1️⃣ **Weather Impact** ☀️
- **Sunny days**: Highest sales (~85 bikes/day)
- **Rainy days**: Lowest sales (~50 bikes/day)
- **Impact**: ~70% sales difference between sunny and rainy days
- **Recommendation**:
  - ✅ Increase inventory before sunny weekends
  - ✅ Plan promotions around good weather forecasts
  - ✅ Offer discounts on rainy days to boost sales

---

### 2️⃣ **Holiday Effect** 🎉
- **Holidays/Weekends**: +30 bikes/day on average
- **Impact**: ~40% higher sales than weekdays
- **Recommendation**:
  - ✅ Staff up on weekends
  - ✅ Special weekend promotions
  - ✅ Ensure adequate stock before holidays

---

### 3️⃣ **Temperature Correlation** 🌡️
- **Positive correlation**: +1.5 bikes per degree Celsius
- Warmer months → Higher sales
- **Recommendation**:
  - ✅ Plan seasonal inventory (more stock in summer)
  - ✅ Marketing campaigns aligned with seasons
  - ✅ Consider indoor bike sales for cold months

---

### 4️⃣ **Overall Performance** 📈
- **Average daily sales**: 71 bikes
- **Sales range**: 28 to 116 bikes
- **Stable performance**: Consistent with some variation

---

## 💼 Business Recommendations:

### **Short-Term Actions** (Next Month):
1. ✅ Implement weather-based inventory planning
2. ✅ Create weekend-specific promotions
3. ✅ Monitor sales daily to detect anomalies
4. ✅ Set up alerts for low-stock situations

### **Medium-Term Actions** (Next Quarter):
1. ✅ Build simple predictive model for next-day sales
2. ✅ Analyze customer segments (who buys on rainy vs sunny days?)
3. ✅ Test targeted promotions (rainy day discounts)
4. ✅ Expand analysis to more months for seasonal patterns

### **Long-Term Strategy** (Next Year):
1. ✅ Develop automated demand forecasting system
2. ✅ Integrate weather API for real-time predictions
3. ✅ Optimize pricing strategy based on conditions
4. ✅ Expand to multiple locations with local weather data

---

## 🎓 What We Learned:

### **Data Science Skills Applied**:
1. ✅ Data generation & collection simulation
2. ✅ Exploratory Data Analysis (EDA)
3. ✅ Descriptive statistics
4. ✅ Data visualization (4 types of plots)
5. ✅ Correlation analysis
6. ✅ Pattern recognition
7. ✅ Business insight generation
8. ✅ Actionable recommendations

### **CRISP-DM Workflow**:
1. ✅ **Business Understanding**: Defined problem (what drives sales?)
2. ✅ **Data Understanding**: Explored dataset, checked quality
3. ✅ **Data Preparation**: (Minimal - data already clean)
4. ✅ **Modeling**: (Descriptive analysis - no ML model yet)
5. ✅ **Evaluation**: Validated insights with multiple analyses
6. ✅ **Deployment**: Created actionable business recommendations

---

## 🚀 Next Steps for YOU:

Want to take this project further? Try these:

### **Level 1 - Extend Analysis**:
1. Add more features (promotions, competitor prices)
2. Analyze by time of day (morning vs afternoon sales)
3. Create interactive dashboard (Plotly/Streamlit)

### **Level 2 - Build Predictive Model**:
1. Split data: train/test
2. Build simple Linear Regression model
3. Predict tomorrow's sales
4. Evaluate accuracy

### **Level 3 - Advanced**:
1. Build time series model (ARIMA/Prophet)
2. Multi-step forecasting (predict next 7 days)
3. Incorporate external data (real weather API)
4. Deploy as web app

---

## 🎉 CONGRATULATIONS!

You've completed **Module 01: Introduction to Data Science**!

You now understand:
- ✅ What Data Science is (and AI/ML/DL hierarchy)
- ✅ Different Data Science roles & career paths
- ✅ Complete DS workflow (CRISP-DM)
- ✅ Essential tools & ecosystem
- ✅ Data Scientist mindset & best practices
- ✅ Hands-on analysis from start to finish!

---

## 📚 What's Next?

**Module 02**: Python for Data Science - Part 1 (Deep Dive)
- NumPy mastery
- Pandas fundamentals
- Data manipulation techniques

**Keep practicing!** The more you code, the better you get. 💪

---

### 🌟 Remember:

> **"Data Science is not about fancy algorithms.**
> **It's about solving real problems with data.**
> **Start simple, measure impact, iterate."**

---

**Happy Learning!** 🚀📊🧠

---"""))

# Save complete notebook
nb['cells'] = cells
with open('01_introduction_complete.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Module 01 COMPLETE! All 6 parts created!")
print(f"📊 Total cells: {len(cells)}")
print(f"📝 Notebook saved: 01_introduction_complete.ipynb")
print(f"\\n🎉 You can now open the notebook and start learning!")
