import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# Load model and encoders
# -------------------------
model_path = os.path.join(os.path.dirname(__file__), '../models/disease_model.pkl')
le_path = os.path.join(os.path.dirname(__file__), '../models/label_encoder.pkl')
feature_columns_path = os.path.join(os.path.dirname(__file__), '../models/feature_columns.pkl')

model = joblib.load(model_path)
le = joblib.load(le_path)
feature_columns = joblib.load(feature_columns_path)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🩺 AI-Based Disease Risk Prediction")
st.write("Select your symptoms from the list below to get a disease prediction.")

# User selects symptoms
selected_symptoms = st.multiselect(
    "Choose symptoms you're experiencing:",
    options=feature_columns,
    help="You can select multiple symptoms"
)

# Show number of selected symptoms
if selected_symptoms:
    st.write(f"**Selected {len(selected_symptoms)} symptom(s)**")

# Predict button
if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom!")
    else:
        # Create input dataframe with all features as 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            if symptom in input_data.columns:
                input_data.at[0, symptom] = 1
        
        # Predict
        prediction_encoded = model.predict(input_data)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        # Display result
        st.success(f"✅ **Predicted Disease:** {prediction}")
        
        # Show selected symptoms
        with st.expander("📋 View selected symptoms"):
            for i, symptom in enumerate(selected_symptoms, 1):
                st.write(f"{i}. {symptom}")

# Footer
st.markdown("---")
st.caption("⚠️ This is an AI prediction tool and should not replace professional medical advice.")