#!/usr/bin/env python3
"""
Create comprehensive 01_introduction_complete.ipynb
Menggabungkan teori Data Science dengan praktik sederhana
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# HEADER
cells.append(nbf.v4.new_markdown_cell("""# 📊 Module 01: Introduction to Data Science & Ecosystem

---

## 🎯 Selamat Datang di Dunia Data Science!

Selamat! Anda telah menyelesaikan Python basics di Module 00. Sekarang saatnya memulai perjalanan **Data Science** yang sesungguhnya!

### 📖 Apa yang Akan Anda Pelajari:

1. **What is Data Science?** - Memahami apa itu Data Science, AI, ML, DL
2. **Roles in Data Science** - Berbagai peran dan responsibilities
3. **Data Science Workflow** - Complete pipeline dari data hingga insight
4. **Tools & Ecosystem** - Tools dan libraries yang digunakan
5. **Data Scientist Mindset** - Cara berpikir yang tepat
6. **Mini Project** - Hands-on pertama Anda!

### ⏱️ Estimasi Waktu: 3-4 jam

### 💡 Learning Approach:

Course ini menggunakan **Dual-Pillar Approach**:
- 🧠 **Teori**: Memahami MENGAPA dan KAPAN
- 💻 **Praktik**: Implementasi BAGAIMANA

---

**Let's begin your Data Science journey!** 🚀

---"""))

# ============================================================================
# PART 1: WHAT IS DATA SCIENCE?
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📚 PART 1: What is Data Science?

## Progress: 1/6 🟩⬜⬜⬜⬜⬜

---

## 1.1 Definisi Data Science

**Data Science** adalah ilmu yang menggabungkan berbagai bidang untuk mengekstrak pengetahuan dan insight dari data.

### 🎯 Tujuan Data Science:
1. **Descriptive**: Apa yang terjadi? (What happened?)
2. **Diagnostic**: Mengapa terjadi? (Why did it happen?)
3. **Predictive**: Apa yang akan terjadi? (What will happen?)
4. **Prescriptive**: Apa yang harus dilakukan? (What should we do?)

### 🧩 Data Science = Intersection of:

```
        Programming
             ▲
             |
             |
Statistics ◄─┼─► Domain
             |    Knowledge
             |
             ▼
         Math
```

**Data Science combines**:
- 💻 **Programming/Computer Science**: Tools untuk process data
- 📊 **Statistics & Mathematics**: Methods untuk analyze data
- 🎓 **Domain Knowledge**: Understanding of the business/field
- 🧠 **Communication**: Ability to tell story dengan data

### 📝 Simple Analogy:

**Data Science seperti detektif modern:**
- 🔍 **Data** = Bukti di TKP (crime scene)
- 🧠 **Analysis** = Investigasi dan mencari pola
- 📊 **Insights** = Kesimpulan siapa pelakunya
- 💡 **Action** = Rekomendasi untuk mencegah kejahatan berikutnya

---"""))

cells.append(nbf.v4.new_markdown_cell("""## 1.2 AI vs ML vs DL - Apa Bedanya?

Istilah-istilah ini sering membingungkan. Mari kita jelaskan dengan jelas!

### 🎯 Hierarchy (Dari Luas ke Spesifik):

```
┌─────────────────────────────────────────┐
│  Artificial Intelligence (AI)           │
│  ┌───────────────────────────────────┐  │
│  │  Machine Learning (ML)            │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Deep Learning (DL)         │  │  │
│  │  │                             │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 1. 🤖 Artificial Intelligence (AI)

**Definisi**: Komputer yang bisa melakukan tugas yang biasanya butuh kecerdasan manusia.

**Contoh**:
- Siri, Alexa (voice assistants)
- Chess AI, Game AI
- Chatbots
- Recommendation systems

**Analogi**: AI = Anak yang bisa berpikir dan belajar

---

### 2. 🧠 Machine Learning (ML)

**Definisi**: Subset dari AI. Komputer **belajar dari data** tanpa di-program secara eksplisit.

**Cara Kerja**:
- Input: Data
- Process: Algorithm belajar pola
- Output: Model yang bisa predict

**Contoh**:
- Email spam filter (belajar dari emails yang Anda mark sebagai spam)
- Netflix recommendations (belajar dari film yang Anda tonton)
- Credit scoring (belajar dari data historical)

**Analogi**: ML = Anak yang belajar dari pengalaman. 
- Tidak perlu diajarkan aturan detail
- Belajar dari contoh (data)
- Semakin banyak pengalaman (data), semakin pintar

**Types of ML**:
1. **Supervised Learning**: Belajar dari data yang sudah dilabel
   - Regression (predict angka): harga rumah, suhu
   - Classification (predict kategori): spam/not spam, cat/dog

2. **Unsupervised Learning**: Belajar dari data tanpa label
   - Clustering: Grouping customers
   - Dimensionality Reduction: Simplify data

3. **Reinforcement Learning**: Belajar dari trial & error dengan reward
   - Game AI
   - Self-driving cars

---

### 3. 🕸️ Deep Learning (DL)

**Definisi**: Subset dari ML. Menggunakan **Neural Networks** dengan banyak layers.

**Cara Kerja**:
- Terinspirasi dari otak manusia
- Banyak "neurons" yang saling terhubung
- Bisa belajar representasi yang sangat complex

**Contoh**:
- Face recognition (iPhone Face ID)
- Self-driving cars
- Language translation (Google Translate)
- Image generation (DALL-E, Midjourney)
- ChatGPT

**Analogi**: DL = Anak jenius yang bisa belajar hal sangat kompleks
- Bisa recognize wajah dari berbagai angle
- Bisa understand bahasa natural
- Butuh BANYAK data dan computational power

**Kapan Pakai DL?**:
- ✅ Data sangat banyak (millions of samples)
- ✅ Problem sangat complex (image, speech, text)
- ✅ Ada GPU/TPU untuk training
- ❌ Jangan pakai untuk simple problems (overkill!)

---

### 📊 Summary Table: AI vs ML vs DL

| Aspect | AI | ML | DL |
|--------|----|----|-----|
| **Scope** | Paling luas | Subset AI | Subset ML |
| **Goal** | Mimic human intelligence | Learn from data | Learn complex patterns |
| **Data Needed** | Varies | Moderate | Very large |
| **Complexity** | Varies | Moderate | High |
| **Examples** | Chatbot, Rule-based | Linear Regression, Random Forest | CNN, RNN, Transformer |
| **Hardware** | CPU ok | CPU ok | GPU/TPU needed |

---

### 💡 Real-World Example:

**Problem**: Identify apakah gambar adalah kucing atau anjing

**AI Approach (Rule-based)**:
```
if has_whiskers and pointy_ears:
    return "cat"
elif floppy_ears:
    return "dog"
```
❌ Problem: Rules terlalu rigid, banyak exceptions

**ML Approach (Traditional)**:
```
- Extract features manually (color histogram, edges, etc.)
- Train classifier (e.g., Decision Tree)
- Predict based on features
```
✅ Better, but feature engineering manual

**DL Approach**:
```
- Feed raw images to Neural Network
- Network learns features automatically
- Much higher accuracy
```
✅ Best, but needs lots of data

---"""))

# Add code example untuk demonstrate simple concept
cells.append(nbf.v4.new_code_cell("""# Simple demonstration: Rule-based vs ML mindset

# Rule-based (Traditional Programming)
def is_spam_rule_based(email):
    \"\"\"
    Hard-coded rules untuk detect spam
    \"\"\"
    spam_words = ["viagra", "lottery", "winner", "free money"]
    
    email_lower = email.lower()
    for word in spam_words:
        if word in email_lower:
            return "SPAM"
    return "NOT SPAM"

# Test
email1 = "Congratulations! You won the lottery!"
email2 = "Hi, let's meet for coffee tomorrow"

print("Rule-based Approach:")
print(f"Email 1: {is_spam_rule_based(email1)}")
print(f"Email 2: {is_spam_rule_based(email2)}")

print("\\n" + "="*50)
print("Machine Learning Approach:")
print("ML model akan BELAJAR dari ribuan contoh email")
print("dan menemukan pola sendiri (tidak hard-coded)")
print("="*50)"""))

cells.append(nbf.v4.new_markdown_cell("""### 🎓 Key Takeaway:

**Data Science** menggunakan semua tools ini (AI, ML, DL) tergantung problem:
- Simple problem → Simple ML (Linear Regression, Decision Tree)
- Complex problem → Advanced ML (XGBoost, Random Forest)
- Very complex (image, text) → Deep Learning (CNN, RNN, Transformer)

**Prinsip**: **Start simple, add complexity only when needed!**

---"""))

# Continue dengan Parts 2-6...
# Saya akan continue building notebook dengan topik-topik berikutnya

# Save progress
nb['cells'] = cells
with open('01_introduction_complete.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Module 01 notebook started!")
print(f"📊 Current cells: {len(cells)}")
print(f"✅ Part 1 complete: What is Data Science")
