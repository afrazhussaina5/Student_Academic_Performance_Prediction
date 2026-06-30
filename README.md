# Student Academic Performance Predictor

A machine learning project that predicts student exam scores based on behavioural and demographic features using a dataset of 4,000 student records.

---

## Project Structure

```
SAP-4000.csv                   ← Raw dataset (4000 records, 7 features)
Student_Performance.ipynb      ← Main Jupyter Notebook (EDA + ML pipeline)
student_performance_model.pkl  ← Saved best model (Random Forest, tuned)
Model_Comparison.csv           ← R² / MAE / RMSE for all 4 models
Predictions.csv                ← Actual vs Predicted values on test set
requirements.txt               ← Python dependencies
README.md                      ← Project documentation
report.pdf                     ← Full project report
presentation.pptx              ← Internship presentation slides

---

## Dataset

**File:** `SAP-4000.csv`  
**Records:** 4,000 students (after removing duplicates)  
**Target:** `Exam_Score` (continuous, range ~17–100)

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| HoursStudied/Week | Numerical | Weekly study hours (0–16) |
| Tutoring | Categorical | Yes / No |
| Region | Categorical | Urban / Rural |
| Attendance(%) | Numerical | Class attendance percentage (50–100) |
| Parent Education | Categorical | Primary / Secondary / Tertiary |
| Exam_Score | Numerical | **Target variable** |

**Data Quality Notes:**
- `Parent Education` had 422 missing values — handled during preprocessing
- No duplicate records after cleaning (shape confirmed: 4000 × 7)
- No missing values in remaining columns

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution plots for all numeric and categorical features
- Scatter plots: `HoursStudied/Week` vs `Exam_Score` (strong positive correlation: **r = 0.82**)
- Scatter plots: `Attendance(%)` vs `Exam_Score` (weak correlation: **r = 0.24**)
- Correlation matrix computed for numerical features

### 2. Preprocessing Pipeline (scikit-learn)
- **Numerical features:** StandardScaler
- **Categorical features:** OneHotEncoder
- Pipeline wraps preprocessor + model for clean train/test separation

### 3. Models Trained

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.9061 | 0.9156 |
| Decision Tree | 0.9997 | 0.8069 |
| Random Forest | 0.9846 | 0.8984 |
| **Gradient Boosting** | **0.9202** | **0.9114** |

### 4. Hyperparameter Tuning
- `GridSearchCV` on Random Forest (cv=5, scoring=R²)
- Search space: `n_estimators` ∈ {100, 200}, `max_depth` ∈ {5, 10, None}
- **Best parameters:** `max_depth=10`, `n_estimators=200`
- **Best CV score:** 0.8947

### 5. Final Model Evaluation (Best Model)
- **MAE:** 3.99
- **RMSE:** 5.03
- **R² Score:** 0.9156

### 6. Cross-Validation (5-Fold)

| Model | CV Average R² |
|---|---|
| Linear Regression | 0.9073 |
| Decision Tree | 0.8012 |
| Random Forest | 0.8915 |
| Gradient Boosting | 0.9056 |

---

## Key Findings

- **HoursStudied/Week** is by far the strongest predictor of exam score (SHAP analysis confirms this as the dominant feature)
- **Attendance(%)** has a secondary positive effect
- **Tutoring** and **Parent Education** have small but measurable influence
- **Gender** and **Region** have minimal impact on exam scores
- Linear Regression achieves the best generalisation (Test R² = 0.9156), suggesting the relationship between features and exam score is largely linear
- The prediction error distribution is approximately normal and centred at zero, indicating unbiased predictions

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```bash
jupyter notebook Student_Performance.ipynb
```

### 3. Load the saved model
```python
import joblib
import pandas as pd

model = joblib.load("student_performance_model.pkl")

# Example prediction
sample = pd.DataFrame([{
    "Gender": "Female",
    "HoursStudied/Week": 10.0,
    "Tutoring": "Yes",
    "Region": "Urban",
    "Attendance(%)": 85,
    "Parent Education": "Secondary"
}])

prediction = model.predict(sample)
print(f"Predicted Exam Score: {prediction[0]:.2f}")
```

---

## Requirements

See `requirements.txt` for the full list. Core dependencies:

- Python 3.10+
- scikit-learn 1.4.2
- pandas 2.2.2
- numpy 1.26.4
- matplotlib 3.8.4
- seaborn 0.13.2
- shap 0.45.1
- joblib 1.4.0

---

## Results Summary

The **Linear Regression** model (within a preprocessing pipeline) achieved the best test performance with an R² of **0.9156**, MAE of **3.99**, and RMSE of **5.03**. The learning curve confirms the model is well-generalised with no significant overfitting. SHAP analysis reveals that weekly study hours account for the majority of variance in predicted exam scores.

---

## Author
B.Mohammed Afraz Hussain Basha
**Project Type:** Machine Learning — Regression  
**Dataset Size:** 4,000 records  
**Tools:** Python, scikit-learn, SHAP, Jupyter Notebook, Google Colab

