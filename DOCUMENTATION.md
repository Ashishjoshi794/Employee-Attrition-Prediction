# Employee Attrition Prediction — Project Documentation

## 1. Project Overview

### What the Project Does

This project builds a machine learning model to predict whether an employee is likely to leave a company or stay.

The model predicts:

* **Yes** → Employee is likely to leave
* **No** → Employee is likely to stay

The prediction is based on employee-related information such as age, job role, monthly income, overtime status, job satisfaction, work-life balance, and years at the company.

### Why This Problem Is Important

Employee attrition is an important business problem because replacing employees can require significant time and resources for:

* Recruitment
* Training
* Onboarding
* Lost productivity

Predicting employees who may be at risk of leaving can help organizations take preventive actions and improve employee retention.

### Main Objective

The main objective is to develop and deploy a binary classification model that predicts employee attrition using 16 employee-related features from the IBM HR Employee Attrition dataset.

---

## 2. Dataset

The project uses a selected-columns version of the IBM HR Employee Attrition dataset.

### Dataset Information

| Property        | Details                   |
| --------------- | ------------------------- |
| Dataset         | IBM HR Employee Attrition |
| Samples         | 1,470 employees           |
| Total Columns   | 17                        |
| Input Features  | 16                        |
| Target Variable | Attrition                 |
| Problem Type    | Binary Classification     |

### Target Variable

The target variable is:

```text
Attrition
```

The target contains two classes:

* `Yes` → Employee left the company
* `No` → Employee stayed in the company

### Class Distribution

| Class | Number of Employees | Percentage |
| ----- | ------------------: | ---------: |
| No    |               1,233 |      83.9% |
| Yes   |                 237 |      16.1% |

The dataset is imbalanced because the number of employees who stayed is much higher than the number who left.

To give more importance to the minority class, `class_weight="balanced"` was used in the final Logistic Regression model.

### Data Quality

* Missing values: **0**
* Duplicate rows: **0**
* Dataset size: **1,470 rows × 17 columns**

### Input Features

The project uses 16 input features.

#### Numerical Features

1. Age
2. DistanceFromHome
3. Education
4. EnvironmentSatisfaction
5. JobLevel
6. JobSatisfaction
7. MonthlyIncome
8. StockOptionLevel
9. TotalWorkingYears
10. WorkLifeBalance
11. YearsAtCompany

#### Categorical Features

1. BusinessTravel
2. Department
3. JobRole
4. MaritalStatus
5. OverTime

---

## 3. Data Preprocessing

The dataset contains both numerical and categorical features, so different preprocessing techniques were applied.

### Numerical Feature Scaling

The numerical features were standardized using:

```text
StandardScaler
```

This transforms numerical features to a similar scale with approximately:

* Mean = 0
* Standard deviation = 1

### Categorical Feature Encoding

Categorical features were converted into numerical form using:

```text
OneHotEncoder(handle_unknown="ignore")
```

This allows categorical values to be represented numerically so that they can be used by the machine learning model.

### ColumnTransformer

A `ColumnTransformer` was used to apply the appropriate preprocessing technique to each feature type.

The preprocessing pipeline was:

```text
Numerical Features
        ↓
StandardScaler
        ↓
Categorical Features
        ↓
OneHotEncoder
        ↓
ColumnTransformer
```

### Target Encoding

The target variable was converted from text into numerical values:

```text
Yes → 1
No  → 0
```

### Image Processing

Image-related preprocessing was **not used** in this project.

This is a tabular classification project, so the following are not applicable:

* Image resizing
* Image normalization
* Image augmentation
* Image masks
* Segmentation

---

## 4. Machine Learning Models

Several classification algorithms were tested during the project.

### Baseline Models

The following six models were initially trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. XGBoost

After the initial comparison, the top three models were selected for hyperparameter tuning:

* Logistic Regression
* Random Forest
* XGBoost

---

## 5. Model Architecture

The final saved model is a **Scikit-learn Pipeline**.

It contains two main steps:

```text
Input Employee Data
        ↓
ColumnTransformer
        ↓
 ┌───────────────────────┐
 │ Numerical Features    │
 │ → StandardScaler      │
 └───────────────────────┘
        +
 ┌───────────────────────┐
 │ Categorical Features  │
 │ → OneHotEncoder       │
 └───────────────────────┘
        ↓
Logistic Regression
        ↓
Prediction
```

### Pipeline Components

#### 1. Preprocessor

The `ColumnTransformer` contains:

* `StandardScaler` for numerical features
* `OneHotEncoder(handle_unknown="ignore")` for categorical features

#### 2. Final Model

The final classifier is:

```text
Logistic Regression
```

### Model Input

The model receives a pandas DataFrame containing the 16 employee features.

### Model Output

The model provides:

* Class prediction
* Probability for each class using `predict_proba()`

The output classes are:

```text
0 → No → Employee likely to stay
1 → Yes → Employee likely to leave
```

---

## 6. Training Process

The overall training process was:

```text
Load Dataset
      ↓
Data Cleaning
      ↓
Target Encoding
      ↓
Train-Test Split
      ↓
Build Preprocessing Pipeline
      ↓
Train Baseline Models
      ↓
Evaluate Models
      ↓
Select Top Models
      ↓
Hyperparameter Tuning
      ↓
Evaluate Tuned Models
      ↓
Select Final Model
      ↓
Save Model
      ↓
Deploy with Streamlit
```

### Train-Test Split

The dataset was divided into:

* **80% Training Data:** 1,176 rows
* **20% Test Data:** 294 rows

A stratified split was used to maintain a similar class distribution in both training and test datasets.

```text
train_test_split(..., stratify=y)
```

---

## 7. Hyperparameter Tuning

Hyperparameter tuning was performed using:

```text
GridSearchCV
```

### Cross-Validation

The tuning process used:

```text
StratifiedKFold
```

with:

* 5 folds
* `shuffle=True`

### Scoring Metric

The main scoring metric used during hyperparameter tuning was:

```text
F1 Score
```

F1 score was selected because the dataset is imbalanced and both precision and recall are important.

### Final Logistic Regression Parameters

The best parameters for the final Logistic Regression model were:

| Parameter    | Value    |
| ------------ | -------- |
| C            | 0.1      |
| class_weight | balanced |
| solver       | lbfgs    |
| max_iter     | 1000     |

---

## 8. Baseline Model Evaluation

Before hyperparameter tuning, six baseline models were evaluated on the test dataset.

| Model               | Accuracy | F1 Score | Precision | Recall | ROC-AUC |
| ------------------- | -------: | -------: | --------: | -----: | ------: |
| Logistic Regression |   0.8639 |   0.3750 |    0.7059 | 0.2553 |  0.6175 |
| Decision Tree       |   0.7789 |   0.3564 |    0.3333 | 0.3830 |  0.6186 |
| Random Forest       |   0.8605 |   0.3279 |    0.7143 | 0.2128 |  0.5983 |
| AdaBoost            |   0.8299 |   0.2188 |    0.4118 | 0.1489 |  0.5542 |
| Gradient Boosting   |   0.8537 |   0.3582 |    0.6000 | 0.2553 |  0.6115 |
| XGBoost             |   0.8435 |   0.3784 |    0.5185 | 0.2979 |  0.6226 |

---

## 9. Tuned Model Evaluation

After hyperparameter tuning, three models were compared in detail.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7551 |    0.3529 | 0.6383 |   0.4545 |  0.7658 |
| Random Forest       |   0.8503 |    0.5556 | 0.3191 |   0.4054 |  0.7845 |
| XGBoost             |   0.8639 |    0.7059 | 0.2553 |   0.3750 |  0.7691 |

---

## 10. Final Model Selection

The final model selected for deployment was:

```text
Logistic Regression
```

### Why Logistic Regression?

XGBoost and Random Forest achieved higher overall accuracy.

However, employee attrition is an imbalanced classification problem, and identifying employees who may leave is an important objective.

The tuned Logistic Regression model achieved:

* **Recall = 63.83%** for the attrition class
* **F1 Score = 45.45%**

Compared with:

* Random Forest Recall = 31.91%
* XGBoost Recall = 25.53%

Therefore, Logistic Regression was selected because it identifies a larger proportion of employees who actually leave the company.

The final model was selected in the training notebook and saved as the deployed `.pkl` file.

---

## 11. Final Model Classification Report

The final Logistic Regression model was evaluated on the 294-row test dataset.

| Class   | Precision | Recall | F1 Score | Support |
| ------- | --------: | -----: | -------: | ------: |
| 0 — No  |      0.92 |   0.78 |     0.84 |     247 |
| 1 — Yes |      0.35 |   0.64 |     0.45 |      47 |

### Overall Performance

| Metric             | Score |
| ------------------ | ----: |
| Accuracy           |  0.76 |
| Macro Precision    |  0.64 |
| Macro Recall       |  0.71 |
| Macro F1 Score     |  0.65 |
| Weighted Precision |  0.83 |
| Weighted Recall    |  0.76 |
| Weighted F1 Score  |  0.78 |

The most important metric for the attrition class is **Recall**, because a higher recall means the model identifies more employees who actually leave.

---

## 12. Model Saving

The final trained model was saved using Joblib.

Model location:

```text
model/employee_attrition_model.pkl
```

The saved file contains the complete trained Scikit-learn Pipeline, including:

* Preprocessing
* Feature transformation
* Logistic Regression model

This allows the same preprocessing and model to be used during prediction without manually repeating the preprocessing steps.

---

## 13. Model Deployment

The trained model was integrated into a Streamlit web dashboard.

### Model Loading

The application loads the saved model using:

```python
joblib.load("model/employee_attrition_model.pkl")
```

### Prediction Process

The Streamlit application follows this process:

```text
User enters employee information
              ↓
Create a pandas DataFrame
              ↓
Saved Pipeline
              ↓
Preprocessing
              ↓
Logistic Regression
              ↓
Prediction + Probability
              ↓
Display Result
```

### Prediction Methods

The application uses:

```text
model.predict()
```

to generate the predicted class.

It also uses:

```text
model.predict_proba()
```

to obtain the probability for each class.

---

## 14. Streamlit Dashboard

The application contains five main pages.

### 14.1 Home / Overview

This page provides:

* Project title
* Project description
* Problem statement
* Objective
* Dataset information
* Technologies used

### 14.2 Model Information

This page displays:

* Final model
* Model architecture
* Input features
* Preprocessing methods
* Training configuration
* Final model performance

### 14.3 Prediction

This page allows users to enter employee information.

After clicking the **Predict** button, the application displays:

* Likely to Stay / Likely to Leave
* Probability of staying
* Probability of leaving

### 14.4 Results / Performance

This page displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Model performance charts

### 14.5 Model Comparison

This page compares the three tuned models:

* Logistic Regression
* Random Forest
* XGBoost

It also explains why Logistic Regression was selected as the final model.

---

## 15. Dashboard Workflow

The complete dashboard workflow is:

```text
Home / Overview
       ↓
Model Information
       ↓
Prediction
       ↓
Results / Performance
       ↓
Model Comparison
```

Users can navigate between pages using the sidebar.

---

## 16. Technologies Used

The project was developed using the following technologies:

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Programming language               |
| Pandas       | Data handling                      |
| NumPy        | Numerical operations               |
| Scikit-learn | Preprocessing and machine learning |
| XGBoost      | Gradient boosting model            |
| Joblib       | Model saving and loading           |
| Matplotlib   | Data visualization                 |
| Streamlit    | Web dashboard and deployment       |
| Google Colab | Model training environment         |

---

## 17. Project Structure

```text
employee_attrition/
│
├── app.py
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
│
└── model/
    └── employee_attrition_model.pkl
```

### File Description

| File                           | Purpose                                 |
| ------------------------------ | --------------------------------------- |
| `app.py`                       | Main Streamlit application              |
| `requirements.txt`             | Required Python packages                |
| `README.md`                    | Project overview and quick-start guide  |
| `DOCUMENTATION.md`             | Detailed project documentation          |
| `employee_attrition_model.pkl` | Final trained machine learning pipeline |

---

## 18. Installation and Setup

Make sure Python 3.9 or higher is installed.

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 19. Usage Guide

1. Start the application using `streamlit run app.py`.
2. Open the Streamlit dashboard in your browser.
3. Use the sidebar to navigate between pages.
4. Open the **Prediction** page.
5. Enter the employee's information.
6. Click the **Predict** button.
7. View the predicted result.
8. Check the prediction probabilities for both outcomes.
9. Use the **Results / Performance** and **Model Comparison** pages to understand model performance.

---

## 20. Project Limitations

### Class Imbalance

The dataset contains significantly more employees who stayed than employees who left.

```text
No  → 83.9%
Yes → 16.1%
```

This makes it more difficult to correctly identify all employees who may leave.

### Model Performance

The final Logistic Regression model achieved approximately 76% accuracy.

Although XGBoost and Random Forest achieved higher accuracy, Logistic Regression was selected because it provided better recall for the attrition class.

### Dataset Size

The dataset contains only 1,470 employee records, which is relatively small for building a model intended for broad real-world use.

### Generalization

The model was trained using a selected-columns version of the IBM HR Employee Attrition dataset.

Therefore, its performance may differ when applied to employee data from different organizations or industries.

### Prediction Interpretation

The prediction probability is a model estimate and should not be treated as a guaranteed prediction of an employee's future behavior.

---

## 21. Future Improvements

Possible future improvements include:

* Collecting a larger and more diverse employee dataset.
* Testing additional machine learning algorithms.
* Applying additional techniques for handling class imbalance.
* Experimenting with different classification thresholds.
* Adding feature importance and model explainability.
* Adding SHAP-based explanations to the Streamlit dashboard.
* Adding more interactive visualizations.
* Monitoring model performance after deployment.
* Periodically retraining the model using new employee data.
* Deploying the application to a cloud platform.

---

## 22. Conclusion

This project demonstrates an end-to-end machine learning workflow for employee attrition prediction.

The project includes:

* Dataset preparation
* Data preprocessing
* Numerical feature scaling
* Categorical feature encoding
* Baseline model training
* Model comparison
* Hyperparameter tuning
* Model evaluation
* Final model selection
* Model saving
* Streamlit dashboard development
* Model deployment

Six classification models were initially evaluated, followed by hyperparameter tuning of Logistic Regression, Random Forest, and XGBoost.

The final **Logistic Regression** model was selected because it achieved a higher recall for the employee attrition class, which is important for identifying employees who may leave the organization.

The final trained pipeline was saved as:

```text
model/employee_attrition_model.pkl
```

and integrated into an interactive Streamlit dashboard for real-time employee attrition prediction.

---

## 23. Guideline Applicability

The original project guideline included requirements for image-based and segmentation projects, such as:

* Image upload
* Ground-truth masks
* Predicted segmentation masks
* Mask overlays
* Image segmentation visualization

These features are **not applicable** to this project because Employee Attrition Prediction is a **tabular binary classification problem**.

The project uses structured employee data in rows and columns rather than images.

Therefore, image and segmentation-specific requirements were intentionally excluded rather than adding unrelated or artificial functionality.

All applicable project requirements, including:

* Home / Overview
* Model Information
* Prediction Interface
* Results / Performance
* Model Comparison
* Project Structure
* README
* Detailed Documentation
* Streamlit Deployment

have been included in the project.
