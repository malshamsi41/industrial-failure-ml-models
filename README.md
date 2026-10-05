# Industrial Failure ML — Part 2: Machine-Learning Models

**EAM04DS Machine Learning | MSc Data Science & Artificial Intelligence**  
**Student:** Mohammed Ali Alshamsi

Part 2 of the three-repository AI4I 2020 industrial equipment failure project.

## Research question
How can traditional machine-learning methods detect rare industrial equipment failures while avoiding misleading conclusions from overall accuracy?

## Models
- Majority-class dummy baseline
- Logistic Regression
- Decision Tree
- Random Forest

The models use the leakage-free operational and engineered features defined in Part 1. Stratification preserves the rare failure proportion. Class weighting is used in the three ML models because only 3.39% of observations are failures.

## Why accuracy is not enough
A majority classifier obtains high accuracy by predicting no failures. Therefore the project emphasizes precision, recall, F1, PR-AUC, ROC-AUC and confusion-matrix counts.

## Current validation results
| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Dummy majority | 0.000 | 0.000 | 0.000 | 0.034 |
| Logistic Regression | 0.154 | 0.833 | 0.260 | 0.468 |
| Decision Tree | 0.395 | 0.907 | 0.551 | 0.802 |
| Random Forest | 0.930 | 0.741 | 0.825 | 0.863 |

The results illustrate an operational trade-off: the Decision Tree has higher recall but many more false alarms; Random Forest provides much higher precision with lower recall.

## Imbalance ablation
The included `imbalance_experiment.csv` compares unweighted and class-weighted Random Forests. Class weighting did **not** improve this validation split, which is retained as a critical result rather than hidden. This shows that imbalance treatments must be empirically evaluated rather than assumed to help.

## Reproduce
```bash
pip install -r requirements.txt
python src/train_models.py
pytest -q
```

## Project sequence
1. Part 1 — Data Preparation
2. **Part 2 — ML Models (this repository)**
3. Part 3 — Evaluation and Interpretation

## Limitation
These results are from a synthetic benchmark dataset and do not establish real-equipment deployment performance.
