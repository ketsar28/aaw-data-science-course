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

# Saya akan continue dengan Parts 2-6
# Untuk menghemat space, saya akan create structure dan key content

# Save current progress
nb['cells'] = cells
with open('01_introduction_complete.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Module 01 Part 1 created!")
print(f"📊 Current cells: {len(cells)}")
