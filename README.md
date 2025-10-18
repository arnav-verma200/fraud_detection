---

# Fraud Detection with RandomForest

## Overview

This project demonstrates a **machine learning model** to detect fraudulent financial transactions in a large-scale dataset (~2 million transactions). The goal is to efficiently identify fraud while minimizing false alarms for legitimate users.

## Dataset

The dataset contains transaction records with the following columns:

| Column                              | Description                                    |
| ----------------------------------- | ---------------------------------------------- |
| `type`                              | Type of transaction (CASH_OUT, TRANSFER, etc.) |
| `amount`                            | Transaction amount                             |
| `nameOrig` / `nameDest`             | Origin and destination accounts                |
| `oldbalanceOrg` / `newbalanceOrig`  | Origin balances before/after transaction       |
| `oldbalanceDest` / `newbalanceDest` | Destination balances before/after transaction  |
| `isFraud`                           | Fraud label (0 = normal, 1 = fraud)            |
| `isFlaggedFraud`                    | Internal flag (not used in model)              |

Additional features derived:

* `balance_diff_org` = oldbalanceOrg − newbalanceOrig
* `balance_diff_dest` = newbalanceDest − oldbalanceDest

**Dataset Link:** [Download Dataset]([https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download))

## Features & Preprocessing

* **Categorical:** `type` → OneHotEncoded
* **Numerical:** `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `balance_diff_org`, `balance_diff_dest`
* **Scaling:** StandardScaler for numeric features
* **Imbalance Handling:** SMOTE oversampling

## Model

* **Algorithm:** RandomForestClassifier (scikit-learn)
* **Parameters:**

  * `n_estimators=50`
  * `class_weight="balanced"`
  * `random_state=42`

The model is wrapped in an **imbalanced-learn pipeline** including preprocessing, SMOTE, and classification.

## Performance Metrics

* **ROC AUC:** 0.9934
* **Average Precision:** 0.9535
* **Confusion Matrix:**

  * True Negatives: 1,906,242
  * True Positives: 1,985
  * False Positives: 80
  * False Negatives: 479

## Visualizations

* ROC Curve
* Precision-Recall Curve
* Confusion Matrix Heatmap
  All plots are saved as `test_plots.png` when running the script.

## Usage

1. Clone the repository:

```bash
git clone <repo_url>
cd fraud_detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from [here](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in the project folder as `AIML-Dataset.csv`.
4. Run the model script:

```bash
python test.py
```

5. Outputs include classification metrics and saved plots.

## Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn
* imbalanced-learn
* matplotlib
* seaborn

## License

This project is licensed under the MIT License.

---
