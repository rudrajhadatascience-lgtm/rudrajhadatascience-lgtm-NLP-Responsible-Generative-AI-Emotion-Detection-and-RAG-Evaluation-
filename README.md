# Emotion Classification & RAG Context Adherence Analysis

A Python-based machine-learning portfolio project exploring two NLP-related tasks: classifying emotions from text and analyzing context adherence and hallucination-related indicators in large language model (LLM) responses.

## Overview

This repository contains two independent experiments:

### 1. Emotion Classification from Text

The goal is to classify text into one of six emotion categories:

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

**Workflow**
1. Load and inspect the emotion dataset.
2. Clean and preprocess text.
3. Convert text into numerical features using TF-IDF.
4. Encode emotion labels.
5. Split the data into training and testing sets.
6. Train Logistic Regression and Linear SVM classifiers.
7. Evaluate the models using accuracy, classification reports, and confusion matrices.

**Recorded results**

| Model | Test accuracy |
|---|---:|
| Logistic Regression | 86.28% |
| Linear SVM | 89.19% |

These results come from the notebook's recorded 80/20 train-test split. They have not been independently reproduced in this README.

### 2. RAG Context Adherence & Hallucination Analysis

This experiment explores a dataset of 30,000 LLM interaction records.

The dataset includes user prompts, provided context, generated responses, model names, similarity scores, context-response overlap, answer relevance, context faithfulness, and hallucination labels.

**Analysis performed**
- Dataset inspection and summary statistics
- LLM model distribution
- Vector database similarity score distribution
- Prompt and response token analysis
- Feature correlation analysis
- Random Forest classification experiment

**Important evaluation note**

The current Random Forest experiment predicts whether the vector similarity score is above its median while also using that same score as an input feature. This introduces target leakage and explains why the recorded accuracy is 100%.

Therefore, this result should not be interpreted as evidence that the model can detect hallucinations. A proper experiment should use `is_hallucination` as the target, exclude leakage-prone features, and evaluate the model on held-out data.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Logistic Regression
- Linear Support Vector Classifier (Linear SVM)
- Random Forest Classifier
- Matplotlib and Seaborn
- Jupyter Notebook

## Repository Structure

```text
emotion-rag-analysis/
├── Untitled7.ipynb
├── emotions.txt
├── RAG_Context_Adherence_And_Hallucination_Benchmark.csv
├── requirements.txt
└── README.md
```

Rename `Untitled7.ipynb` to a descriptive name such as `emotion_and_rag_analysis.ipynb` before publishing.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd emotion-rag-analysis
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run the cells in order.

## Dataset Notes

- `emotions.txt` contains text samples and emotion labels separated by a semicolon.
- `RAG_Context_Adherence_And_Hallucination_Benchmark.csv` contains structured LLM interaction records.

Review the original dataset sources and their licenses before redistribution. Do not assume that the benchmark represents real production LLM traffic unless its provenance confirms this.

## Current Limitations

- Emotion classes are imbalanced, especially the Surprise category.
- Removing stopwords can remove important negations and affect sentiment or emotion meaning.
- The recorded emotion model results should be reproduced with a documented evaluation pipeline.
- The current RAG classification experiment contains target leakage.
- Exploratory correlations do not establish causation.
- This repository does not currently implement a complete retrieval-augmented generation pipeline.

## Future Improvements

- Improve emotion preprocessing and preserve negation.
- Compare macro F1-score and per-class performance.
- Add baseline models and reproducible evaluation.
- Build a leakage-free hallucination classification experiment.
- Analyze false positives and false negatives.
- Add clearer visualizations and experiment documentation.
- Explore retrieval, reranking, and grounded response evaluation in a complete RAG pipeline.

## Learning Outcomes

This project provides practical experience with text preprocessing, TF-IDF feature extraction, supervised classification, exploratory data analysis, model evaluation, and identifying data leakage.

## Disclaimer

The results describe experiments recorded in the notebook. They are not a guarantee of real-world model performance.

## Author

**RUDRA JHA**

