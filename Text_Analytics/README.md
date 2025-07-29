# Text Analytics Coursework – MSc Data Science

This folder contains all work related to the **Text Analytics** module of the MSc Data Science program at the University of Bristol. It focuses on practical NLP tasks including **text classification**, **topic modelling**, and **named entity recognition (NER)**.

---

## 📁 Folder Contents

- `text_analytics_sentiment_ner_topic_modeling_msc.pdf` — Full project report
- `notebooks/`
  - `task1_sentiment_classification/` — Notebook and outputs for sentiment analysis
  - `task2_topic_modeling.ipynb` — Topic modelling using LDA and HDP
  - `task3_named_entity_recognition.ipynb` — CRF, BiLSTM+CRF, TinyBERT
- `data/` — Climate tweets, tokenized corpora
- `models/` — Pretrained and fine-tuned models (e.g., TinyBERT)
- `outputs/` — Evaluation results and visualizations

---

## 📝 Task 1: Sentiment Classification on Climate Tweets

### Objective:
To classify the sentiment of climate-related tweets using classical and transformer-based models.

### Methods:
- **Preprocessing**:
  - Tokenization, stopword removal, lemmatization
  - N-gram and TF-IDF feature extraction
- **Models Implemented**:
  - Naive Bayes (MultinomialNB)
  - TinyBERT (fine-tuned using HuggingFace)
- **Evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix and ROC curves

### Tools:
`Scikit-learn`, `NLTK`, `TinyBERT`, `Transformers`, `HuggingFace`, `Pandas`, `Seaborn`

---

## 🔍 Task 2: Topic Modeling

### Objective:
To extract and compare thematic structure from textual climate data using unsupervised topic models.

### Techniques:
- Latent Dirichlet Allocation (LDA)
- Hierarchical Dirichlet Process (HDP)
- Visualization with pyLDAvis
- Perplexity and coherence score evaluation

### Tools:
`Gensim`, `pyLDAvis`, `SpaCy`, `matplotlib`

---

## 🔎 Task 3: Named Entity Recognition (NER)

### Objective:
To identify and label named entities from unstructured text using various modeling techniques.

### Models:
- Conditional Random Fields (CRF)
- BiLSTM + CRF
- Fine-tuned **TinyBERT** transformer

### Evaluation:
- F1-score per entity type (e.g., PER, LOC, ORG)
- Entity-level precision and recall
- Span-based evaluation metrics

### Tools:
`sklearn_crfsuite`, `Keras`, `PyTorch`, `HuggingFace Transformers`, `SeqEval`, `SpaCy`

---

## 📄 Report
Detailed explanation, implementation, results, and comparative analysis are available in:

**`text_analytics_sentiment_ner_topic_modeling_msc.pdf`**

---

## ✅ Skills Demonstrated
- End-to-end NLP workflow (preprocessing to modeling)
- Classical ML and deep learning for text classification
- Topic extraction and coherence analysis
- Fine-tuning transformer models for NER
- Evaluation with appropriate NLP metrics

---

## 📚 References
(See main [README.md](../README.md#references) for the full list of references.)

---

This project was completed as part of the **Text Analytics** module in the MSc Data Science program at the University of Bristol.
