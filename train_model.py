import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -------------------------
# 1. Load dataset
# -------------------------
data_path = os.path.join(os.path.dirname(__file__), '../data/symptoms.csv')
df = pd.read_csv(data_path)

# Fill NaN with empty string
df = df.fillna('')

# Strip whitespace from all symptom columns
for col in df.columns:
    if col.startswith("Symptom"):
        df[col] = df[col].astype(str).str.strip()

# Target column
target_column = 'Disease'

# -------------------------
# 2. Gather all unique symptoms
# -------------------------
all_symptoms = set()
for col in df.columns:
    if col.startswith("Symptom"):
        for symptom in df[col].unique():
            symptom = str(symptom).strip()
            if symptom != '' and symptom.lower() != 'nan':
                all_symptoms.add(symptom)

# Convert to sorted list and ensure all are strings
all_symptoms = sorted([str(s) for s in all_symptoms])
print(f"Found {len(all_symptoms)} unique symptoms")

# -------------------------
# 3. Create binary feature matrix
# -------------------------
X = pd.DataFrame(0, index=df.index, columns=[str(s) for s in all_symptoms])

# Populate the matrix
for col in df.columns:
    if col.startswith("Symptom"):
        for idx in df.index:
            symptom = str(df.at[idx, col]).strip()
            if symptom != '' and symptom.lower() != 'nan' and symptom in X.columns:
                X.at[idx, symptom] = 1

# Extra safety: ensure all column names are strings
X.columns = X.columns.astype(str)
print(f"Feature matrix shape: {X.shape}")

# -------------------------
# 4. Encode target
# -------------------------
le = LabelEncoder()
y = le.fit_transform(df[target_column])

# -------------------------
# 5. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 6. Train model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# -------------------------
# 7. Save model, label encoder, and feature columns
# -------------------------
os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)

model_path = os.path.join(os.path.dirname(__file__), '../models/disease_model.pkl')
le_path = os.path.join(os.path.dirname(__file__), '../models/label_encoder.pkl')
feature_columns_path = os.path.join(os.path.dirname(__file__), '../models/feature_columns.pkl')

joblib.dump(model, model_path)
joblib.dump(le, le_path)
joblib.dump(list(X.columns), feature_columns_path)  # Save as list of strings

print("✓ Model saved!")
print("✓ Label encoder saved!")
print("✓ Feature columns saved!")
print(f"✓ Total symptoms: {len(all_symptoms)}")
