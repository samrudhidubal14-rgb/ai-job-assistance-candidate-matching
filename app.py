import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "AI_Job_Assistance_Candidate_Matching_Model.pkl"

model = joblib.load(MODEL_FILE)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Job Assistance",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("💼 AI-Based Intelligent Job Assistance & Candidate Matching System")

st.write(
    "An AI-powered system that predicts candidate suitability "
    "using Machine Learning."
)

st.divider()


# ============================================================
# GET FEATURES FROM TRAINED MODEL
# ============================================================

preprocessor = model.named_steps["preprocessor"]

numeric_features = (
    preprocessor.transformers_[0][2]
)

categorical_features = (
    preprocessor.transformers_[1][2]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 About the System")

st.sidebar.write(
    """
    This application uses a trained Machine Learning model
    to classify candidates according to their suitability.

    The system can support recruiters by providing
    data-driven candidate screening assistance.
    """
)

st.sidebar.info(
    "Developed as an MBA AI & Data Science Project"
)


# ============================================================
# CANDIDATE INFORMATION
# ============================================================

st.header("👤 Candidate Information")

candidate = {}

col1, col2 = st.columns(2)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

st.subheader("📊 Numerical Information")

for i, column in enumerate(numeric_features):

    if i % 2 == 0:
        current_col = col1
    else:
        current_col = col2

    with current_col:

        candidate[column] = st.number_input(
            column,
            value=0.0
        )


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

st.subheader("📝 Candidate Details")

col3, col4 = st.columns(2)

for i, column in enumerate(categorical_features):

    if i % 2 == 0:
        current_col = col3
    else:
        current_col = col4

    with current_col:

        candidate[column] = st.text_input(
            column,
            placeholder=f"Enter {column}"
        )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("🔍 Candidate Suitability Prediction")

if st.button(
    "🚀 Predict Candidate Suitability",
    type="primary",
    use_container_width=True
):

    candidate_df = pd.DataFrame(
        [candidate]
    )

    try:

        # Make prediction
        prediction = model.predict(
            candidate_df
        )[0]

        # Display result
        st.success(
            f"Predicted Candidate Category: {prediction}"
        )


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        try:

            probabilities = model.predict_proba(
                candidate_df
            )[0]

            probability_df = pd.DataFrame({

                "Candidate Category":
                    model.classes_,

                "Probability (%)":
                    probabilities * 100

            })

            probability_df[
                "Probability (%)"
            ] = probability_df[
                "Probability (%)"
            ].round(2)


            st.subheader(
                "📈 Prediction Probability"
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            # Highest probability
            max_probability = max(
                probabilities
            ) * 100

            st.metric(
                "Highest Match Probability",
                f"{max_probability:.2f}%"
            )

        except Exception:

            pass


    except Exception as e:

        st.error(
            "Unable to generate prediction."
        )

        st.write(
            "Error:",
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Intelligent Job Assistance & Candidate Matching System"
)

st.caption(
    "MBA AI & Data Science Project"
)
