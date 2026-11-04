# Heart Attack Risk Prediction

Predicts higher/lower chance of a heart attack from patient data using **Logistic Regression** and **Decision Tree** (classical ML only).

## Files
- `Heart_Attack_Analysis.ipynb` – full notebook with outputs (EDA, preprocessing, feature importance, tuning, evaluation)
- `Heart_Attack_Analysis_Data.csv` – dataset (303 patients, 10 features + `Target`)
- `requirements.txt` – dependencies

## Workflow
1. EDA: sanity check, distributions, target-wise plots, correlation heatmap
2. Preprocessing: missing/duplicate check, IQR outlier capping, one-hot encoding (`CP_Type`, `ECG`), standardization inside a Pipeline (no leakage)
3. Feature importance: correlation, mutual information, Random Forest
4. Models: stratified 80/20 split (plus 90/10, 70/30, 60/40), GridSearchCV with 5-fold CV
5. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves

## Result (80/20 test set)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.754 | 0.783 | 0.835 |
| Decision Tree | 0.705 | 0.780 | 0.795 |

Logistic Regression performs better and is more stable across splits.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Heart_Attack_Analysis.ipynb
```
