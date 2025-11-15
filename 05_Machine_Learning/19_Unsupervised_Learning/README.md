# 🔍 Module 19 - Unsupervised Learning Complete

**© Muhammad Ketsar Ali Abi Wahid**

---

## 📌 Overview

Module ini mengajarkan **Unsupervised Learning** - teknik machine learning untuk menemukan pola dalam data **tanpa labels**. Anda akan mempelajari **Clustering**, **Dimensionality Reduction**, dan **Anomaly Detection**!

---

## 🎯 Learning Objectives

Setelah menyelesaikan module ini, Anda akan mampu:

✅ Memahami perbedaan Supervised vs Unsupervised Learning

✅ Mengimplementasikan **K-Means Clustering**

✅ Menggunakan **Hierarchical Clustering** & Dendrograms

✅ Menerapkan **DBSCAN** untuk density-based clustering

✅ Melakukan **PCA** (Principal Component Analysis) untuk dimensionality reduction

✅ Menggunakan **t-SNE** untuk visualization

✅ Mendeteksi **Anomalies** dengan Isolation Forest

✅ Mengevaluasi clustering dengan metrics yang tepat

✅ Memilih jumlah cluster optimal (Elbow Method, Silhouette)

---

## 🤔 Supervised vs Unsupervised Learning

### **Supervised Learning (Modules 16, 17, 23):**
```
Data: X (features) + y (labels) ✅
Goal: Learn X → y mapping
Task: Predict labels for new data
Examples: Classification, Regression
```

### **Unsupervised Learning (Module 19):**
```
Data: X (features) only ❌ No labels!
Goal: Find hidden patterns/structure
Task: Group similar data, reduce dimensions
Examples: Clustering, PCA, Anomaly Detection
```

### **Analogi Sederhana:**

**Supervised = Belajar dengan Guru**
- Teacher gives you questions AND answers
- You learn patterns from examples
- Test: apply what you learned

**Unsupervised = Explorasi Sendiri**
- No teacher, no answers
- You find patterns yourself
- Discover hidden structures

---

## 🎯 CLUSTERING

### **What is Clustering?**

**Definisi:** Mengelompokkan data points into groups (clusters) dimana:
- Points dalam cluster yang sama = **SIMILAR** 🟢
- Points dalam cluster berbeda = **DIFFERENT** 🔴

**Analogi:**
```
Imagine organizing your closet:
- Group 1: T-shirts 👕
- Group 2: Pants 👖
- Group 3: Shoes 👟
- Group 4: Accessories 🎒

You don't have labels, just naturally group similar items!
```

### **Use Cases:**
- Customer Segmentation (marketing groups)
- Document Categorization
- Image Segmentation
- Genomics (group similar genes)
- Anomaly Detection (outliers = separate cluster)

---

## 🎯 K-MEANS CLUSTERING

### **Algorithm:**

```
1. Choose K (number of clusters)
2. Randomly initialize K centroids
3. Assign each point to nearest centroid
4. Update centroids = mean of assigned points
5. Repeat 3-4 until convergence
```

### **Visual Example:**

```
Initial:                After Iteration 1:      Converged:
   ●  ●                     ●  ●                  ● ●
  ● ⊕ ●                   ●  ⊕  ●                ●⊕●
   ●  ●                     ●  ●                  ● ●

  ● ●                      ● ●                     ●●
 ● ⊕ ●                    ●  ⊕  ●                 ●⊕●
  ● ●                      ● ●                     ●●

⊕ = Centroid              Points move to         Stable clusters!
● = Data point            nearest centroid
```

### **Advantages:**
✅ Simple and fast
✅ Scales well to large datasets
✅ Works well with spherical clusters
✅ Easy to interpret

### **Disadvantages:**
❌ Must specify K beforehand
❌ Sensitive to initialization
❌ Assumes spherical clusters
❌ Affected by outliers

### **Choosing Optimal K:**

**1. Elbow Method**
```python
# Plot Inertia (sum of squared distances) vs K
# Look for "elbow" in curve
Inertia
   |  ╲
   |   ╲___________
   |________________
     1  2  3  4  5  K
         ↑ Elbow at K=3!
```

**2. Silhouette Score**
```python
# Measures how similar point is to own cluster vs other clusters
# Range: -1 (bad) to +1 (perfect)
# Choose K with highest average silhouette
```

**3. Domain Knowledge**
```python
# Sometimes K is known from business context
# E.g., Customer segments: Bronze, Silver, Gold = K=3
```

---

## 🌳 HIERARCHICAL CLUSTERING

### **Concept:**

Creates a **tree of clusters** (dendrogram) showing hierarchical relationships.

```
                    ALL DATA
                   /        \
              Cluster A    Cluster B
              /     \        /     \
            C1     C2      C3     C4
```

### **Two Approaches:**

**1. Agglomerative (Bottom-Up):**
```
Step 1: Each point is own cluster
Step 2: Merge closest clusters
Step 3: Repeat until 1 cluster remains
```

**2. Divisive (Top-Down):**
```
Step 1: All points in 1 cluster
Step 2: Split into smaller clusters
Step 3: Repeat until each point is own cluster
```

### **Linkage Methods:**

**Single Linkage:** Minimum distance between clusters
**Complete Linkage:** Maximum distance between clusters
**Average Linkage:** Average distance between all pairs
**Ward Linkage:** Minimize variance (most commonly used!)

### **Dendrogram:**
```
Height
   |           ─────────────
   |      ──┬──            |
   |    ──┬─┘    ──┬──     |
   |  ──┬─┘     ──┬─┘      |
   |___●___●___●___●___●___●
      1   2   3   4   5   6

Cut at different heights → different K!
```

### **Advantages:**
✅ No need to specify K upfront
✅ Produces dendrogram (interpretable)
✅ Works with any distance metric
✅ Deterministic (same result every run)

### **Disadvantages:**
❌ Slow (O(n³) time complexity)
❌ Not suitable for large datasets
❌ Sensitive to noise/outliers
❌ Once merged, can't undo

---

## 🔵 DBSCAN (Density-Based Clustering)

### **Concept:**

Clusters based on **density** - groups areas with high point concentration.

**Key Parameters:**
- **eps (ε):** Maximum distance between points to be neighbors
- **min_samples:** Minimum points to form dense region

### **Point Types:**

1. **Core Point:** Has ≥ min_samples neighbors within eps
2. **Border Point:** Within eps of core point, but not core itself
3. **Noise Point:** Not core, not within eps of core (outlier!)

```
        Core Points: ●
     Border Points: ○
      Noise Points: ×

    ●━━●━━●
    ┃     ┃
    ●     ●━━○

    ×           ×

         ●━━●
         ┃  ┃
         ●━━●
```

### **Advantages:**
✅ **No need to specify K!**
✅ Can find arbitrarily shaped clusters
✅ Robust to outliers (marks them as noise)
✅ Works well with spatial data

### **Disadvantages:**
❌ Sensitive to eps and min_samples
❌ Struggles with varying densities
❌ Not suitable for high-dimensional data
❌ Difficult to interpret parameters

### **When to Use:**
- Geographic/spatial clustering
- Outlier detection
- Clusters with irregular shapes
- Don't know K beforehand

---

## 📉 PCA (Principal Component Analysis)

### **What is PCA?**

**Dimensionality Reduction:** Transform high-dimensional data to lower dimensions while preserving most information.

```
Before:                    After:
10 features                2 features
(complex, hard to plot)    (simple, easy to plot)

Still captures 95% of variance!
```

### **How PCA Works:**

```
Step 1: Center data (subtract mean)
Step 2: Compute covariance matrix
Step 3: Find eigenvectors (principal components)
Step 4: Sort by eigenvalues (importance)
Step 5: Project data onto top K components
```

### **Visual Intuition:**

```
Original 2D Data:          PCA Finds New Axes:
      |                          ╱ PC1
    ● | ●●                     ●╱●●
   ●●●|●●●                   ●●╱●●●
  ●●●●|●●●●                 ●●╱●●●●
──────┼────── →           ──╱─────── PC2
  ●●●●|●●●●               ●╱●●●●●
   ●●●|●●●                 ╱ ●●●
    ● | ●●                ╱  ●●
      |

PC1 = Direction of maximum variance
PC2 = Orthogonal to PC1
```

### **Explained Variance:**
```
Component  Variance  Cumulative
PC1        45%       45%
PC2        30%       75%
PC3        15%       90%
PC4        7%        97%
PC5        3%        100%

→ Keep PC1-PC3 to retain 90% information!
```

### **Advantages:**
✅ Reduces dimensionality (faster training!)
✅ Removes multicollinearity
✅ Noise reduction
✅ Data visualization (2D/3D)
✅ Unsupervised feature extraction

### **Disadvantages:**
❌ Components hard to interpret
❌ Assumes linear relationships
❌ Sensitive to scaling (must standardize!)
❌ May lose some information

### **Use Cases:**
- Visualize high-dimensional data
- Speed up machine learning
- Reduce storage/computation
- Remove correlated features
- Image compression

---

## 🎨 t-SNE (t-Distributed Stochastic Neighbor Embedding)

### **What is t-SNE?**

**Non-linear dimensionality reduction** optimized for **visualization** (usually 2D or 3D).

### **PCA vs t-SNE:**

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| **Speed** | Fast ⚡ | Slow 🐌 |
| **Method** | Linear | Non-linear |
| **Purpose** | Feature reduction | **Visualization only!** |
| **Preserves** | Variance | **Local structure** |
| **Deterministic** | Yes | No (random init) |

### **When to Use:**
✅ Visualizing high-dimensional data (100+ features)
✅ Exploring data structure
✅ Identifying clusters visually
✅ Publication-quality plots

❌ **DON'T use for:**
- Machine learning features (use PCA!)
- Inference on new data
- Quantitative analysis

### **Key Parameters:**

**perplexity:** Balance between local and global structure (5-50)
**learning_rate:** Step size for optimization (10-1000)
**n_iter:** Number of iterations (250-5000)

---

## 🚨 ANOMALY DETECTION

### **What are Anomalies?**

**Anomalies (Outliers):** Data points that significantly differ from normal patterns.

**Examples:**
- Fraudulent credit card transactions 💳
- Network intrusions 🔒
- Defective products 🏭
- Unusual patient vitals 🏥

### **Isolation Forest:**

**Concept:** Anomalies are easier to isolate (require fewer splits).

```
Normal Point:              Anomaly:
Many splits to isolate     Few splits!
      |                         |
   ───┼───                  ────┼─── ●
  ●●● | ●●●                     |
   ●●●|●●●
      |
```

**How it works:**
1. Randomly select feature and split value
2. Recursively partition data
3. Anomalies have **shorter path** to isolation
4. Score = average path length across trees

### **Advantages:**
✅ Fast and scalable
✅ Works in high dimensions
✅ Few hyperparameters
✅ Unsupervised (no labels needed!)

### **Use Cases:**
- Fraud detection
- Quality control
- System monitoring
- Medical diagnosis

---

## 📊 Clustering Evaluation Metrics

### **1. Silhouette Score**
```python
Score range: -1 to +1
+1: Perfect clustering
 0: Overlapping clusters
-1: Wrong clustering

Formula: (b - a) / max(a, b)
a = avg distance within cluster
b = avg distance to nearest cluster
```

### **2. Davies-Bouldin Index**
```python
Lower is better
Measures cluster separation vs compactness
```

### **3. Calinski-Harabasz Index**
```python
Higher is better
Ratio of between-cluster to within-cluster variance
```

### **⚠️ Warning:**
These metrics are **internal** (no ground truth needed) but:
- May not align with business goals
- Should be combined with domain knowledge
- Visual inspection still important!

---

## 🎯 Module 19 Contents

### **19.1 K-Means Clustering**
- Implementation & visualization
- Elbow method for optimal K
- Silhouette analysis
- Customer segmentation example

### **19.2 Hierarchical Clustering**
- Agglomerative clustering
- Dendrogram visualization
- Linkage methods comparison
- Product categorization example

### **19.3 DBSCAN**
- Parameter tuning (eps, min_samples)
- Noise detection
- Geographic clustering example

### **19.4 PCA**
- Dimensionality reduction
- Explained variance
- Feature visualization
- Data compression example

### **19.5 t-SNE**
- High-dimensional visualization
- Parameter tuning
- Cluster visualization
- MNIST digit visualization

### **19.6 Anomaly Detection**
- Isolation Forest
- Anomaly scoring
- Fraud detection example

---

## 🚀 Quick Start

```bash
# Navigate to module
cd 05_Machine_Learning/19_Unsupervised_Learning

# Install dependencies
pip install -r requirements.txt

# Run clustering example
python 19_clustering_complete.py

# Run PCA example
python 19_pca_tsne_complete.py

# Run anomaly detection
python 19_anomaly_detection.py
```

---

## 💡 Real-World Applications

### **1. Customer Segmentation** 🛒
```
Cluster customers by:
- Purchase behavior
- Demographics
- Engagement level
→ Personalized marketing!
```

### **2. Image Segmentation** 🖼️
```
Group similar pixels:
- Medical imaging (tumor detection)
- Satellite imagery (land use)
- Object recognition
```

### **3. Recommendation Systems** 📺
```
Cluster similar items/users:
- Netflix: group similar movies
- Spotify: group similar songs
- Amazon: product recommendations
```

### **4. Fraud Detection** 🔒
```
Anomaly detection:
- Credit card fraud
- Insurance claims
- Network intrusions
```

### **5. Gene Expression Analysis** 🧬
```
Cluster genes with similar expression:
- Disease classification
- Drug discovery
- Personalized medicine
```

---

## 🎓 Best Practices

### **1. Always Standardize Data**
```python
# Clustering is distance-based!
# Features with different scales will dominate
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### **2. Visualize First**
```python
# Use PCA or t-SNE to see if clusters exist
# Don't force clustering on random data!
```

### **3. Try Multiple Methods**
```python
# Different algorithms for different data:
# - Spherical clusters → K-Means
# - Hierarchical structure → Hierarchical
# - Arbitrary shapes → DBSCAN
```

### **4. Validate Results**
```python
# Use multiple metrics
# Check business relevance
# Iterate and refine
```

---

## 📚 Comparison Table

| Method | Speed | Scalability | K Required | Cluster Shape | Outliers |
|--------|-------|-------------|------------|---------------|----------|
| **K-Means** | ⚡⚡⚡ | Excellent | Yes | Spherical | Sensitive |
| **Hierarchical** | 🐌 | Poor | No | Any | Sensitive |
| **DBSCAN** | ⚡⚡ | Good | No | Any | **Robust** |
| **PCA** | ⚡⚡⚡ | Excellent | N/A | Linear | Sensitive |
| **t-SNE** | 🐌🐌 | Poor | N/A | Non-linear | Robust |

---

**© Muhammad Ketsar Ali Abi Wahid**

**Data Science Zero to Hero: Complete MLOps & Production ML Engineering**

**Module 19 - Unsupervised Learning Complete**

---

> "Unsupervised learning is like exploring a new city without a map - you discover hidden gems by wandering and observing patterns!" 🗺️✨
