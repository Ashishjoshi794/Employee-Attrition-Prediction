import os
import streamlit as st
import pandas as pd
import joblib
import sklearn.compose._column_transformer as ct


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "employee_attrition_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    # Compatibility for ColumnTransformer pickle
    if not hasattr(ct, "_RemainderColsList"):

        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList

    if not os.path.exists(MODEL_PATH):

        st.error("Model file not found.")
        st.code(MODEL_PATH)
        st.stop()

    try:

        model = joblib.load(MODEL_PATH)

        return model

    except Exception as e:

        st.error("Unable to load the trained model.")
        st.exception(e)
        st.stop()


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("👨‍💼 Employee Attrition Prediction")

st.write(
    "Predict whether an employee is likely to leave the company "
    "based on employee-related information."
)

st.divider()


# ============================================================
# EMPLOYEE INPUT FORM
# ============================================================

st.subheader("Enter Employee Information")

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=35
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=29,
        value=5
    )

    education = st.selectbox(
        "Education Level",
        [1, 2, 3, 4, 5],
        index=2
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4],
        index=2
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5],
        index=1
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4],
        index=2
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=100
    )

    stock_option_level = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3],
        index=0
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=8
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4],
        index=2
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "No",
            "Yes"
        ]
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Attrition",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [{
            "Age": age,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EnvironmentSatisfaction": environment_satisfaction,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "BusinessTravel": business_travel,
            "Department": department,
            "JobRole": job_role,
            "MaritalStatus": marital_status,
            "OverTime": overtime
        }]
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    st.write("### 🔍 Model Debug Information")

    st.write(
        "RAW PREDICTION:",
        repr(prediction)
    )

    st.write(
        "MODEL CLASSES:",
        model.classes_
    )

    st.write(
        "PROBABILITIES:",
        probabilities
    )


       # --------------------------------------------------------
    # Get probability according to model classes
    # --------------------------------------------------------

    classes = list(model.classes_)

    probability_dict = dict(
        zip(classes, probabilities)
    )

    # Model uses:
    # 0 = No / Stay
    # 1 = Yes / Leave

    no_probability = probability_dict.get(0, 0)
    yes_probability = probability_dict.get(1, 0)


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ This employee is LIKELY TO LEAVE the company."
        )

        st.write(
            "**Prediction:** Attrition = Yes"
        )

    else:

        st.success(
            "✅ This employee is LIKELY TO STAY at the company."
        )

        st.write(
            "**Prediction:** Attrition = No"
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader("Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Probability of Staying",
            f"{no_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Probability of Leaving",
            f"{yes_probability * 100:.2f}%"
        )

    st.progress(
        float(yes_probability)
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    if yes_probability >= 0.50:

        st.warning(
            "The model estimates a higher probability "
            "of employee attrition."
        )

    else:

        st.info(
            "The model estimates a higher probability "
            "that the employee will stay."
        )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("View Employee Information"):

        st.dataframe(
            input_data,
            use_container_width=True
        )