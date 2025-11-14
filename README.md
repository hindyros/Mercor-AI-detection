# Mercor AI Text Detection - Final Project

**Course:** 15.095: Machine Learning Under a Modern Optimization Lens (Fall 2025)  
**Competition:** [Mercor AI Text Detection](https://www.kaggle.com/competitions/mercor-ai-detection/overview)  

---

## Team Information

- **Team Member 1:** Hindy Rossignol - hindyros@mit.edu
- **Team Member 2:** Elie Juvenspan - ejuven@mit.edu

---

## Problem Summary

With the increasing accessibility of large language models (LLMs), verifying the authenticity of written work has become a critical technical and social challenge. This project addresses the problem of **AI-generated text detection** in writing assessments, where we must distinguish between genuine human writing and inauthentic or AI-assisted text.

The problem is framed as a binary classification task:
- **Class 0:** Authentic human writing (not cheating)
- **Class 1:** Inauthentic writing (AI-generated, copy-pasted, or otherwise inauthentic)

This is a supervised learning problem where we predict the probability that a writing sample represents cheating behavior. The evaluation metric is **ROC-AUC**, which measures how well our model separates legitimate and inauthentic writing.

### Why This Problem Matters

1. **Academic Integrity:** Educational institutions need tools to verify student work authenticity
2. **Technical Challenge:** Modern LLMs produce increasingly human-like text, making detection difficult
3. **Optimization Perspective:** We can frame feature selection, model selection, and hyperparameter tuning as optimization problems
4. **Real-World Impact:** Successful detection systems can help maintain trust in written assessments

---

## Dataset

### Source
The dataset comes from the [Mercor AI Text Detection Kaggle Competition](https://www.kaggle.com/competitions/mercor-ai-detection), hosted by Mercor Inc.

### Dataset Structure
All datasets are organized in the `data/` folder:
- **Raw Data (`data/raw/`):**
  - `train.csv`: 269 labeled samples
    - `id`: Unique identifier
    - `topic`: The writing topic chosen by the writer
    - `answer`: The written response (typically 50-300 words)
    - `is_cheating`: Binary label (0 = authentic, 1 = inauthentic)
  - `test.csv`: 264 unlabeled samples (same structure as training, without `is_cheating`)
  - `sample_submission.csv`: Format template for predictions

- **Processed Data (`data/processed/`):**
  - Processed datasets, feature-engineered data, and embeddings
  - `train_for_finetune.csv`, `val_for_finetune.csv`: Train/val splits
  - `processed_data/`: Feature-engineered datasets with embeddings

- **Submissions (`data/submissions/`):**
  - All submission files for the competition

### Dataset Characteristics
- Writing samples vary in topic, tone, and length
- Encompasses multiple domains: fiction, non-fiction, marketing/social, film/screenwriting
- Responses are anonymized and from standardized writing tasks
- Class distribution needs to be analyzed (potential class imbalance)

### Data Preprocessing
We extract comprehensive text-based features including:
- Basic statistics (length, word count, sentence count)
- Punctuation and capitalization features
- Word complexity metrics (long words, vocabulary diversity)
- AI indicators (first-person pronouns, etc.)
- Topic encoding

---

## Methods and Relation to Course

### 1. **Feature Engineering as Optimization**
- **Feature Selection Problem:** We extract 20+ heuristic features from text
- **Optimization Lens:** Treat feature selection as a subset selection problem to maximize model performance while controlling complexity
- **Implementation:** Use techniques like LASSO regularization, feature importance analysis, and forward/backward selection

### 2. **Model Selection and Hyperparameter Optimization**
- **Models to Explore:**
  - **Random Forest:** Ensemble method with interpretable feature importance
  - **XGBoost:** Gradient boosting with built-in regularization
  - **OCT-H (Optimal Classification Trees with Hyperplanes):** Directly relates to optimization-based tree learning
  - **Deep Neural Networks:** Multi-layer architectures with regularization
  - **Logistic Regression:** Baseline with interpretable coefficients

- **Optimization Perspective:**
  - Hyperparameter tuning as a black-box optimization problem
  - Cross-validation for model selection
  - Regularization (L1/L2) to control model complexity
  - Ensemble methods as optimization over multiple models

### 3. **Sparse and Interpretable Models**
- **Goal:** Make models more sparse and regularized to understand why text is detected as AI
- **Methods:**
  - LASSO for robustness and a bit more sparsity, etc. 
  - Sparse neural networks
  - Interpretability techniques (SHAP, feature importance)
  - Rule extraction from tree-based models

### 4. **Multimodal Learning**
- **Approach:** Combine tabular features with text embeddings
- **Optimization Challenge:** Learn optimal combination weights between feature types
- **Implementation:**
  - Extract embeddings from pre-trained language models (e.g., BERT, RoBERTa)
  - Combine with heuristic features
  - Learn fusion weights through optimization

### 5. **Hard Example Mining and Subset Selection**
- **Challenge:** Train on hardest subset (subset selection problem)
- **Optimization Framework:** Select training samples that maximize learning signal
- **Methods:**
  - Active learning approaches
  - Curriculum learning
  - Hard negative mining

### 6. **Prescriptive Analytics**
- **Goal:** Not just predict, but provide actionable insights
- **Approach:** Understand what makes text detectable and provide recommendations
- **Optimization:** Formulate as a recommendation/action optimization problem

---

## Challenges and Ideas to Overcome Them

### Challenge 1: **Small Dataset Size**
- **Problem:** Only 269 training samples, making overfitting a significant risk
- **Solutions:**
  - Use cross-validation extensively
  - Apply strong regularization (L1/L2, dropout for neural networks)
  - Data augmentation techniques (paraphrasing, synonym replacement)
  - Transfer learning from pre-trained language models
  - Ensemble methods to reduce variance

### Challenge 2: **Class Imbalance**
- **Problem:** Potentially uneven distribution of authentic vs. inauthentic samples
- **Solutions:**
  - Stratified cross-validation
  - Class weighting in loss functions
  - SMOTE or other oversampling techniques
  - Focus on ROC-AUC (less sensitive to imbalance than accuracy)

### Challenge 3: **Unseen Topics in Test Set**
- **Problem:** Test set may contain topics not seen in training
- **Solutions:**
  - Robust topic encoding (handle unseen topics as -1 or most common)
  - Focus on topic-agnostic features
  - Domain adaptation techniques

### Challenge 4: **Feature Engineering Complexity**
- **Problem:** Balancing feature richness with overfitting risk
- **Solutions:**
  - Automated feature selection using optimization
  - Regularization to encourage sparsity
  - Feature importance analysis to focus on most predictive features

### Challenge 5: **Model Interpretability**
- **Problem:** Understanding why a text is flagged as AI-generated
- **Solutions:**
  - Use interpretable models (trees, linear models)
  - SHAP values for feature attribution
  - Rule extraction from tree-based models
  - Sparse models that highlight key features

### Challenge 6: **Generalization to New Domains**
- **Problem:** Model may not generalize to different writing styles or topics
- **Solutions:**
  - Domain-invariant feature extraction
  - Multi-domain training
  - Robust optimization objectives

---

## Project Structure

```
mercor-ai-detection/
├── README.md                    # This file
├── data_exploration.ipynb       # Exploratory data analysis
├── heuristic_model.ipynb        # Baseline models (RF, XGBoost, Logistic Regression)
├── train.csv                    # Training data
├── test.csv                     # Test data
├── sample_submission.csv        # Submission format
└── submission.csv               # Generated predictions
```

---

## Evaluation Metrics

- **Primary Metric:** ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
- **Secondary Metrics:** Accuracy, Precision, Recall, F1-Score
- **Leaderboard:** 
  - Public (30% of test data)
  - Private (70% of test data) - used for final ranking

---

## Competition Prizes

- First Place: $1,000
- Second Place: $500
- Third Place: $300
- Fourth Place: $150
- Fifth Place: $50

*Note: Must create account on Mercor platform and fill out submission form to be eligible for prizes.*

---

## Citation

Aditya Bhandari '25 and Peter Zhang. Mercor AI Text Detection. https://kaggle.com/competitions/mercor-ai-detection, 2025. Kaggle.

---

## Timeline

- **November 7, 2025:** Project Proposal Due
- **December 8, 2025:** Project Presentation Due
- **December 13, 2025:** Final Report Due (max 8 pages, not including appendices)

---

## Next Steps

1. Complete baseline model training and evaluation
2. Implement OCT-H and compare with XGBoost
3. Develop deep neural network architecture
4. Implement multimodal approach (tabular + embeddings)
5. Feature selection and sparsity optimization
6. Interpretability analysis
7. Final model ensemble and submission

---

## References

- [Mercor AI Text Detection Competition](https://www.kaggle.com/competitions/mercor-ai-detection)
- Course materials: 15.095 Machine Learning Under a Modern Optimization Lens

