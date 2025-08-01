# Fine-Tuning Transformer Models for Sentiment Analysis on the IMDb Dataset

This repository documents a practical journey into fine-tuning modern NLP models for a sentiment classification task. The project starts with a simple baseline model and progressively introduces optimizations to achieve a higher-performing, state-of-the-art result.

## Project Overview

The goal of this project is to train a model that can accurately classify movie reviews from the IMDb dataset as either "Positive" or "Negative". This is a classic binary classification problem that serves as an excellent benchmark for understanding the process of fine-tuning Large Language Models.

We explore two main experiments:
1.  A **baseline model** using the fast and efficient `DistilBERT` to establish an initial performance score.
2.  An **optimized model** using the more powerful `RoBERTa` architecture, a larger dataset, and more advanced training techniques to improve upon the baseline.

## Notebooks in this Repository

*   **[`01_sentiment_analysis_baseline_distilbert.ipynb`](./01_sentiment_analysis_baseline_distilbert.ipynb)**
    *   This notebook covers the complete, end-to-end process of fine-tuning the `distilbert-base-uncased` model on a subset of the IMDb dataset. It serves as our initial benchmark.

*   **[`02_sentiment_analysis_optimized_roberta.ipynb`](./02_sentiment_analysis_optimized_roberta.ipynb)**
    *   This notebook implements several improvements to achieve a better result. The changes include:
        1.  **Using a Better Model:** Upgrading from DistilBERT to `roberta-base`.
        2.  **Using More Data:** Training on the full 25,000-review training set.
        3.  **Advanced Training:** Training for more epochs and tuning the learning rate for more stable convergence.

## Key Concepts Demonstrated

This project demonstrates a wide range of essential AI/ML engineering skills:
-   **Fine-Tuning Pre-trained Models:** Leveraging the power of models from the Hugging Face Hub.
-   **Data Processing & Tokenization:** Preparing text data for Transformer models.
-   **The `Trainer` API:** Using the high-level Hugging Face `Trainer` for efficient training and evaluation.
-   **Hyperparameter Tuning:** Experimenting with learning rates, epochs, and schedulers.
-   **Performance Evaluation:** Measuring model performance using standard metrics like Accuracy and Loss.
-   **Identifying & Mitigating Overfitting:** Analyzing training logs to see the effects of overfitting and using techniques like `load_best_model_at_end` to handle it.

## How to Use

1.  Clone this repository.
2.  Open either of the `.ipynb` notebooks in Google Colab.
3.  To run the code, ensure you select a GPU runtime (`Runtime` -> `Change runtime type` -> `GPU`).

## Results Summary

The goal of the project was to improve upon the baseline. The results show a clear and significant improvement in performance with the optimized approach.

| Metric | Baseline Model (DistilBERT) | Optimized Model (RoBERTa) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 92.35% | **94.63%** | **+2.28%** |
| **Validation Loss** | 0.4828 | **0.1494** | **-69%** |

## Future Improvements

The model could be further improved by exploring techniques such as:
-   **Early Stopping:** To make the training process more efficient by automatically stopping when the validation loss stops improving.
-   **Data Augmentation:** To create more training data and improve model robustness.
-   **Trying larger models:** Experimenting with `roberta-large` for potentially higher accuracy, at the cost of more compute.
