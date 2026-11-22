import os
import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(
    page_title="Netflix Top 3 Predictor", 
    page_icon="https://www.citypng.com/public/uploads/preview/vector-netflix-logo-701751694792511ae31ddmjow.png?v=2026111403", 
    layout="centered")

st.title("Netflix Global Top 3 Predictor")

st.write(
    "This app uses a **Scikit-Learn Random Forest Pipeline** trained on Netflix's Global Top 10 dataset "
    "to predict whether a title will rank in the **Top 3**."
)

@st.cache_resource
def load_artifact():
    if not os.path.exists("model.joblib"):
        return None
    try:
        return joblib.load("model.joblib")
    except Exception:
        return None

artifact = load_artifact()

if artifact is None:
    st.error("Model file (`model.joblib`) is missing or invalid. Make sure it is uploaded next to `app.py`.")
    st.stop()

model = artifact["model"]
feature_order = artifact["features"]
model_accuracy = artifact.get("accuracy", 0.0)
feature_importances = artifact.get("importances", {})

st.info(f"**Model's Test Accuracy:** `{model_accuracy:.1%}`")

st.subheader("Enter Title Details")

category = st.selectbox(
    "Content Category",
    options=[
        "Films (English)",
        "Films (Non-English)",
        "TV (English)",
        "TV (Non-English)"
    ]
)

month_names = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
selected_month_name = st.radio(
    "Release Month (Nov & Dec trigger Holiday Season)",
    options=list(month_names.keys()),
    horizontal=True
)

col1, col2 = st.columns(2)

with col1:
    hours_millions = st.slider(
        "Weekly Hours Viewed (Millions)",
        min_value=0.7,
        max_value=575.0,
        value=15.0,
        step=0.5
    )
    runtime = st.slider(
        "Runtime (Hours)",
        min_value=0.5,
        max_value=48.0,
        value=2.2,
        step=0.1
    )

with col2:
    views_millions = st.slider(
        "Weekly Views (Millions)",
        min_value=0.6,
        max_value=68.0,
        value=3.1,
        step=0.2
    )
    weeks_in_top_10 = st.slider(
        "Cumulative Weeks in Top 10",
        min_value=1,
        max_value=65,
        value=2,
        step=1
    )


st.subheader("2. Prediction Result")

if st.button("Predict Top 3 Probability", type="primary"):
    try:
        month_num = month_names[selected_month_name]

        input_data = {
            "weekly_hours_viewed": int(hours_millions * 1_000_000),
            "runtime": runtime,
            "weekly_views": float(views_millions * 1_000_000),
            "cumulative_weeks_in_top_10": weeks_in_top_10,
            "month": month_num,
            "is_holiday_season": month_num,
            "category_Films (English)": (category == "Films (English)"),
            "category_Films (Non-English)": (category == "Films (Non-English)"),
            "category_TV (English)": (category == "TV (English)"),
            "category_TV (Non-English)": (category == "TV (Non-English)")
        }

       
        input_df = pd.DataFrame([input_data])[feature_order]

        prob_top_3 = model.predict_proba(input_df)[0][1] * 100

        st.metric(
            label="Predicted Probability of Top 3 Rank",
            value=f"{prob_top_3:.1f}%",
            delta="Strong Top 3 Contender" if prob_top_3 >= 50 else "Likely Ranks #4–#10"
        )

    except Exception as e:
        st.warning(f"⚠️ Could not generate prediction due to invalid input. Details: {e}")


if feature_importances:
    with st.expander("📊 View What Drives Top 3 Rankings (Feature Importance)"):
        imp_df = pd.DataFrame(
            list(feature_importances.items()),
            columns=["Feature", "Importance"]
        ).sort_values(by="Importance", ascending=True)
        st.bar_chart(imp_df.set_index("Feature"))

st.divider()
st.caption(
    "**About this model:** Built with Scikit-Learn and trained on "
    "10,920 weekly records from the Netflix Global Top 10 dataset. This application is an educational example only and not official streaming advice."
)
