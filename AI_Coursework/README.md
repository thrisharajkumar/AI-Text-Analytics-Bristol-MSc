# AI Coursework – MSc Data Science

This folder contains all files related to the **Artificial Intelligence** coursework from the MSc Data Science program at the University of Bristol. It covers **semantic learning**, **unsupervised clustering**, and **supervised neural network models**, completed across two major tasks.

---

## 📁 Contents

- `ai_semantic_clustering_supervised_models_msc.pdf` — Full report with analysis and results
- `notebooks/`
  - `task1_unsupervised_semantic_clustering.ipynb`
  - `task2_supervised_model_robustness.ipynb`
- `data/` — Includes co-occurrence matrices, vocabulary sets
- `images/` — Visualizations like PCA plots, clustering graphs

---

## 🧠 Task 1: Semantic Clustering and Vocabulary Engineering

### Objective:
To explore and analyze the **semantic structure** of textual data using **unsupervised learning techniques** and dimensionality reduction.

### Key Steps:
- Built **co-occurrence matrices** from two novels
- Extracted high-frequency nouns and verbs using `spaCy`
- Created frequency-based and LDA-based vocabularies
- Applied **Principal Component Analysis (PCA)** to reduce dimensions
- Used **KMeans clustering** and **Hierarchical Clustering** to identify groups
- Constructed **semantic graphs** and applied **Dijkstra’s algorithm** for path analysis
- Compared semantic distances between vocabularies using:
  - **Graph-based distance**
  - **Distributional distance**
  - **LDA-based topic divergence**

### Tools:
`spaCy`, `Gensim`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `NLTK`, `NetworkX`

---

## 🤖 Task 2: Supervised Learning & Model Robustness

### Objective:
To implement and evaluate a set of **supervised machine learning models** and assess their **robustness** under different data conditions.

### Models Implemented:
- **Logistic Regression**
- **Single-Layer Perceptron**
- **Multi-Layer Perceptron (MLP)**:
  - Small MLP
  - Deep MLP
  - Wide MLP

### Evaluation Scenarios:
Each model was tested across 12 different scenarios combining:
- **Data curvature**
- **Noise levels**
- **Class imbalance**

### Metrics:
- Accuracy
- Confusion Matrix
- Learning Curves
- Comparative performance plots

### Key Insights:
- Deep MLP provided the highest robustness across distorted inputs.
- Logistic regression performed poorly under curved and imbalanced data.
- Model complexity helped mitigate noise and non-linearity but increased variance.

### Tools:
`Scikit-learn`, `Keras`, `TensorFlow`, `Pandas`, `Matplotlib`

---

## 📄 Report
You can find detailed methodology, equations, diagrams, and results in the report:

**`ai_semantic_clustering_supervised_models_msc.pdf`**

---

## ✅ Skills Demonstrated
- Semantic vector construction & analysis
- PCA, KMeans, clustering evaluations
- Graph algorithms for NLP (Dijkstra’s)
- Deep learning model building (MLP)
- Performance tuning & robustness testing
- Comparative ML evaluation

---

## 📚 References
(See main [README.md](../README.md#references) for full bibliography)

---

This coursework reflects rigorous academic and practical exploration of AI techniques, and is part of the EMATM0067 module at the University of Bristol.
