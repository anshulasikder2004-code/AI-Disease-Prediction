This project is an AI-driven digital triage platform that assesses patient symptoms, predicts possible diseases and guides patients to appropriate care pathways. (ongoing)
It is designed for low-resource settings, aiming to reduce hospital overcrowding, improve access to timely care and empower patients with actionable guidance.

Current Work (Implemented):

1.Symptom-based disease prediction using Random Forest Classifier

2.Backend: Python, scikit-learn, pandas

3.Web interface: Streamlit

4.Model artifacts saved as .pkl files (disease model, label encoder, feature columns)

Future Work / Planned Features:

1️⃣ Risk Stratification

Categorize patients into Low / Medium / High severity

Automatically flag urgent cases for immediate care

Display risk scores with visual indicators

2️⃣ Referral Guidance

Suggest the most appropriate facility (clinic, hospital, specialist)

Provide guidance based on patient location & facility availability

Prioritize urgent referrals to reduce ER overload

3️⃣ Resource-Aware Recommendations

Adjust recommendations based on local hospital capacity & nearby clinics

Suggest home care or telemedicine for mild cases

Optimize patient flow and reduce unnecessary hospital visits

4️⃣ Health Equity Features

Support local language interfaces (Bangla)

Accessible for underserved populations with limited health literacy

Include guidance for informal healthcare providers (pharmacies, village doctors)

5️⃣ Advanced AI / ML Features

Expand symptom dataset for better prediction accuracy

Implement NLP-based symptom extraction from patient text

Explore multi-label disease prediction for complex cases

Use model explainability (SHAP / LIME) for patient & doctor understanding

6️⃣ Mobile-Friendly & Scalable Interface

Deploy as mobile-first web app

Allow patients to input symptoms easily on phones

Provide real-time predictions without heavy computational load

7️⃣ Integration with Healthcare Systems

Connect with public hospital databases for live updates

Enable early warning alerts for hospitals on high-risk patients

Provide analytics dashboard for hospital administrators

