# ============================================================
#   Student Stress Level Prediction System
#   Author: Stefanie S. Relos | BSIT-301
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================
print("=" * 50)
print("  STUDENT STRESS LEVEL PREDICTION SYSTEM")
print("=" * 50)

df = pd.read_csv("student_stress_sleep_screen.csv")

print("\n[1] Dataset Loaded Successfully!")
print(f"    Total Records : {len(df)}")
print(f"    Total Columns : {len(df.columns)}")
print(f"\n    Columns: {df.columns.tolist()}")

# ============================================================
# STEP 2 — DATA PREPROCESSING
# ============================================================
print("\n[2] Preprocessing Data...")

# Drop student_id (not needed)
df.drop(columns=["student_id"], inplace=True, errors="ignore")

# Encode categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target from encoding list temporarily
target_col = "stress_level"
feature_cats = [col for col in categorical_cols if col != target_col]

for col in feature_cats:
    df[col] = le.fit_transform(df[col])

# Encode target label
df[target_col] = le.fit_transform(df[target_col])
stress_classes = le.classes_  # Save for display later

print(f"    Encoded Columns : {categorical_cols}")
print(f"    Stress Classes  : {stress_classes}")

# ============================================================
# STEP 3 — SPLIT FEATURES AND TARGET
# ============================================================
print("\n[3] Splitting Features and Target...")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"    Training Set : {len(X_train)} records")
print(f"    Testing Set  : {len(X_test)} records")

# ============================================================
# STEP 4 — TRAIN THE MODEL
# ============================================================
print("\n[4] Training Random Forest Model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("    Model Trained Successfully!")

# ============================================================
# STEP 5 — EVALUATE THE MODEL
# ============================================================
print("\n[5] Evaluating Model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n    Accuracy : {accuracy * 100:.2f}%")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=stress_classes))

# ============================================================
# STEP 6 — CONFUSION MATRIX PLOT
# ============================================================
print("\n[6] Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=stress_classes,
    yticklabels=stress_classes
)
plt.title("Confusion Matrix — Student Stress Level Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("    Saved: confusion_matrix.png")

# ============================================================
# STEP 7 — FEATURE IMPORTANCE PLOT
# ============================================================
print("\n[7] Generating Feature Importance Chart...")

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance — Stress Level Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("    Saved: feature_importance.png")

# ============================================================
# STEP 8 — SAVE THE TRAINED MODEL
# ============================================================
print("\n[8] Saving Trained Model...")

joblib.dump(model, "stress_model.pkl")
print("    Saved: stress_model.pkl")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 50)
print("  TRAINING COMPLETE!")
print(f"  Final Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)