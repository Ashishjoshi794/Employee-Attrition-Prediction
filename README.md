# Employee Attrition Prediction

A Machine Learning project that predicts whether an employee is likely to leave a company based on employee-related information.

The project includes data preprocessing, machine learning model training, model comparison, hyperparameter tuning, model saving, and a Streamlit dashboard for making predictions.

---

# Project Overview

Employee attrition is an important problem for organizations because losing experienced employees can increase recruitment and training costs.

This project uses machine learning to identify employees who are more likely to leave the company.

The final trained model is integrated into a **Streamlit dashboard**, where users can enter employee information and receive an attrition prediction with probability.

---

## Dataset

The project uses the **IBM HR Employee Attrition dataset**.

* Total records: *1,470*
* Input features: **16**
* Target variable: **Attrition**
* Target classes:

  * `Yes` — Employee leaves the company
  * `No` — Employee stays in the company

### Input Features

The model uses the following employee information:

* Age
* Business Travel
* Department
* Distance From Home
* Education
* Environment Satisfaction
* Job Level
* Job Role
* Job Satisfaction
* Marital Status
* Monthly Income
* Over Time
* Stock Option Level
* Total Working Years
* Work Life Balance
* Years At Company

---

## Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Cleaning & Preparation
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment
```

---

## Data Preprocessing

The dataset contains both numerical and categorical features.

### Numerical Features

Numerical features were standardized using:

* `StandardScaler`

### Categorical Features

Categorical features were converted into numerical form using:

* `OneHotEncoder`
* `handle_unknown="ignore"`

A `ColumnTransformer` was used to apply the appropriate preprocessing to each type of feature.

---

## Machine Learning Models

Three machine learning models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**.

The main scoring metric used during tuning was **F1 Score**.

---

## Model Performance

The following results were obtained on the test dataset:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   75.51% |    35.29% | 63.83% |   45.45% |  76.58% |
| Random Forest       |   85.03% |    55.56% | 31.91% |   40.54% |  78.45% |
| XGBoost             |   86.39% |    70.59% | 25.53% |   37.50% |  76.91% |
<img width="758" height="95" alt="image" src="https://github.com/user-attachments/assets/245f4c07-7b60-4477-9bad-37918c364deb" />


---

## Final Model

**Logistic Regression** was selected as the final model.

Although Random Forest and XGBoost achieved higher overall accuracy, Logistic Regression achieved a much higher **Recall for employees who leave (`Attrition = Yes`)**.

The final model achieved:

* Accuracy: **75.51%**
* Precision: **35.29%**
* Recall: **63.83%**
* F1 Score: **45.45%**
* ROC-AUC: **76.58%**

The final model was saved using Joblib:

```text
model/employee_attrition_model.pkl
```

---

## Streamlit Dashboard

The trained model was integrated into a Streamlit web application.

The dashboard contains the following pages:

### 1. Home / Overview

Provides an overview of the project, dataset, and objective.

### 2. Model Information

Displays information about:

* Final model
* Features
* Preprocessing
* Training details

### 3. Prediction

Users can enter employee information and generate an attrition prediction.

The dashboard displays:

* Predicted outcome
* Probability of staying
* Probability of leaving

### 4. Results / Performance

Displays the actual model performance metrics and visualizations.

### 5. Model Comparison

Provides a comparison between:

* Logistic Regression
* Random Forest
* XGBoost

---

## Project Structure

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

---

## Requirements

* Python 3.9 or higher
* Streamlit
* Pandas
* Scikit-learn
* Joblib
* XGBoost

All required packages are listed in:

```text
requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd employee_attrition
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit dashboard using:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## Prediction Example

The user enters employee information such as:

```text
Age: 30
Business Travel: Travel_Rarely
Department: Research & Development
Job Level: 2
Job Role: Research Scientist
Monthly Income: 5000
OverTime: No
Years At Company: 4
```

The dashboard then provides a prediction such as:

```text
Prediction: Employee is likely to Stay

Probability:
Stay: 72%
Leave: 28%
```

The actual prediction depends on the employee information entered by the user.

---

## Key Learning Outcomes

Through this project, I worked with:

* Data preprocessing
* Feature engineering and feature selection
* Categorical encoding
* Feature scaling
* Classification algorithms
* Logistic Regression
* Random Forest
* XGBoost
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Classification evaluation metrics
* Model comparison
* Model serialization using Joblib
* Streamlit dashboard development
* Machine learning model deployment

---

## Limitations

* The model is trained using a specific HR dataset and may not generalize to every organization.
* The dataset contains historical employee information, so real-world employee behavior may differ.
* Prediction probabilities should be treated as model estimates, not guaranteed outcomes.
* Additional real-world HR data could potentially improve the model.

---

## Future Improvements

Possible future improvements include:

* Testing additional classification algorithms
* Handling class imbalance with additional techniques
* Feature importance and explainability
* Model monitoring
* Using a larger and more diverse HR dataset
* Cloud deployment
* Adding interactive business insights to the dashboard

---

## Documentation

Detailed project documentation is available in:

```text
DOCUMENTATION.md
```

---

## Author

**Ashish Joshi**

Data Science / Machine Learning Project

---

## Project Type

**Machine Learning | Classification | Employee Attrition Prediction | Streamlit Deployment**
