import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

# -------------------------------
# 1. Load Model + Assets
# -------------------------------
@st.cache_resource
def load_model_assets():
    model = joblib.load('tb_model.pkl')
    features = joblib.load('feature_names.pkl')
    cat_features = joblib.load('cat_features.pkl')
    return model, features, cat_features

model, features_list, cat_features = load_model_assets()

# -------------------------------
# 2. Page Setup
# -------------------------------
st.set_page_config(page_title="TB Knowledge Predictor", layout="centered")
st.title("🩺 TB Knowledge Prediction System")
st.markdown("---")
st.write("Enter demographic details to predict TB transmission knowledge based on DHS data.")

# -------------------------------
# 3. User Inputs
# -------------------------------
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        education = st.selectbox(
            "Education Level",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["No Education", "Primary", "Secondary", "Higher"][x]
        )

        wealth = st.selectbox(
            "Wealth Index",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: ["Poorest", "Poorer", "Middle", "Richer", "Richest"][x-1]
        )

    with col2:
        residence = st.radio(
            "Residence Type",
            options=[1, 2],
            format_func=lambda x: "Urban" if x == 1 else "Rural"
        )

        sex = st.radio(
            "Sex",
            options=[1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

    weight = st.slider("Survey Weight (Normalized)", 0.0, 5.0, 1.0)

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Analyze Results", use_container_width=True):

    # Create empty dataframe with correct columns
    input_data = pd.DataFrame(columns=features_list)

    # Fill categorical columns with "Unknown"
    for col in features_list:
        if col in cat_features:
            input_data.loc[0, col] = "Unknown"
        else:
            input_data.loc[0, col] = 0  # numeric default

    # Overwrite user-provided values
    input_data.loc[0, 'Education'] = education
    input_data.loc[0, 'Wealth_index'] = wealth
    input_data.loc[0, 'Residence'] = residence
    input_data.loc[0, 'Sex'] = sex
    input_data.loc[0, 'Weight_Normalized'] = weight

    # Ensure correct data types
    for col in input_data.columns:
        if col in cat_features:
            input_data[col] = input_data[col].astype(str)
        else:
            input_data[col] = pd.to_numeric(input_data[col], errors="coerce").fillna(0)

    # 🔥 IMPORTANT: Use Pool with cat_features
    input_pool = Pool(input_data, cat_features=cat_features)

    probability = model.predict_proba(input_pool)[0][1]

    # -------------------------------
    # 5. Display Results
    # -------------------------------
    st.markdown("### **Prediction Output**")

    if probability >= 0.87:
        st.success(f"✅ High TB Knowledge Confirmed ({probability:.1%})")

    elif probability <= 0.13:
        st.error(f"🚨 Knowledge Gap Detected ({probability:.1%})")

    else:
        st.warning(f"⚠️ Uncertain Knowledge Level ({probability:.1%})")

    # -------------------------------
    # 6. Feature Importance
    # -------------------------------
    st.markdown("---")
    st.subheader("📊 Key Determinants for this Prediction")

    importance = model.get_feature_importance()
    feat_imp_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(5)

    st.bar_chart(feat_imp_df.set_index('Feature'))
    st.caption("Top 5 global factors used by the model.")

# -------------------------------
# 7. Education Section
# -------------------------------
st.markdown("---")
with st.expander("📚 Learn More About TB Transmission"):
    st.write("""
    TB spreads through the air when a person with lung TB coughs or sneezes.
    It is curable with proper antibiotics treatment.
    """)