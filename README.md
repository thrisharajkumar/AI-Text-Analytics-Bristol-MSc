# MSc Data Science Coursework – AI & Text Analytics Projects

This repository contains my academic coursework for the MSc Data Science program at the University of Bristol. It highlights applied AI and NLP techniques implemented as part of two major modules:

- **Introduction to Artificial Intelligence**
- **Text Analytics**

Each folder includes code, analysis, and a comprehensive PDF report summarizing the findings and methodologies.

---

## Folder Structure

```
.
├── AI_Coursework/
│   ├── ai_semantic_clustering_supervised_models_msc.pdf
│   ├── notebooks/
│
├── Text_Analytics/
│   ├── text_analytics_sentiment_ner_topic_modeling_msc.pdf
│   ├── notebooks/
│
└── README.md
```

---

## 1. AI_Coursework – Semantic Learning & Supervised Models

### Summary:
This project investigates both **unsupervised** and **supervised learning** models on curated NLP datasets.

### Highlights:
- **Semantic Clustering:** Constructed co-occurrence matrices for two novels, performed PCA, KMeans, hierarchical clustering, and Dijkstra shortest-path analysis.
- **Vocabulary Engineering:** Used POS tagging and LDA to curate topic-based and frequency-based vocabularies.
- **Supervised Models:** Implemented and evaluated:
  - Logistic Regression
  - Single Layer Perceptron
  - Multi-Layer Perceptron (Small, Deep, Wide)
- **Robustness Analysis:** Tested models under varied curvature, noise, and class imbalance conditions.

### Technologies:
`Python`, `SpaCy`, `NLTK`, `Scikit-learn`, `Matplotlib`, `Pandas`, `Gensim`

📄 Full Report: `AI_Coursework/ai_semantic_clustering_supervised_models_msc.pdf`

---

## 2. Text_Analytics – Sentiment Analysis, Topic Modeling & NER

### Summary:
This project explores classical and deep NLP techniques on real-world datasets, including climate tweets and textual corpora.

### Highlights:
- **Sentiment Classification:**
  - Naive Bayes (TF-IDF, n-grams)
  - Fine-tuned TinyBERT model
- **Topic Modeling:** LDA and HDP applied to uncover latent topics in climate text data.
- **NER Comparison:**
  - CRF, BiLSTM + CRF
  - TinyBERT (transformer-based)

### Technologies:
`Python`, `Scikit-learn`, `BERT`, `PyTorch`, `CRF`, `Gensim`, `HuggingFace Transformers`, `spaCy`

📄 Full Report: `Text_Analytics/text_analytics_sentiment_ner_topic_modeling_msc.pdf`

---

## Core Skills Demonstrated

- Natural Language Processing (NLP)
- Supervised & Unsupervised Learning
- Dimensionality Reduction & Clustering
- Neural Networks & Transformer Models
- Named Entity Recognition (NER)
- Topic Modeling (LDA, HDP)
- Sentiment Analysis
- Evaluation Techniques (F1, Accuracy, Confusion Matrix)
- Data Preprocessing & Feature Engineering

---

## Academic Context
These projects were completed as part of:
- **EMATM0067: Introduction to Artificial Intelligence**
- **Text Analytics Module**
University of Bristol – MSc Data Science (2025)

---

## License
This repository is for academic and portfolio purposes only. © 2025. All rights reserved.

## References

The techniques and models implemented in this coursework are informed and inspired by the following sources:

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O’Reilly Media.
2. Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A Primer in BERTology. *TACL, 8*, 842–866.
3. Goyal, P., Pandey, S., & Jain, K. (2018). *Deep Learning for Natural Language Processing*. Apress.
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
5. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft). Stanford University.
6. Balog, K. (2018). *Entity-Oriented Search*. Springer.
7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
8. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O’Reilly Media.
9. Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press.
10. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
11. Silge, J., & Robinson, D. (2017). *Text Mining with R: A Tidy Approach*. O’Reilly Media.
12. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
13. Blei, D. M. (2012). Probabilistic Topic Models. *Communications of the ACM, 55*(4), 77–84.
14. Goldberg, Y. (2017). *Neural Network Methods for Natural Language Processing*. Morgan & Claypool.
15. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.
16. Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04816*.
