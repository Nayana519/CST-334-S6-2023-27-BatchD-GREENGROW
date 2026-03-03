import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import numpy as np

print("[INFO] Loading dataset.csv...")

data = pd.read_csv("dataset.csv")

print("[SUCCESS] Dataset loaded successfully!")
print("[INFO] Columns:", data.columns)
print(f"[INFO] Dataset shape: {data.shape}")

# Encode categorical columns
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])
data['Fertilizer Name'] = le_fert.fit_transform(data['Fertilizer Name'])

# Features
X = data[['Temparature', 'Humidity', 'Moisture',
          'Soil Type', 'Crop Type',
          'Nitrogen', 'Potassium', 'Phosphorous']]

# Target
y = data['Fertilizer Name']

# Feature scaling for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[INFO] Training optimized Gradient Boosting Model...")

# Split with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Use Gradient Boosting for better accuracy
model = GradientBoostingClassifier(
    n_estimators=500,           # Increased estimators for better accuracy
    learning_rate=0.05,         # Lower learning rate for better convergence
    max_depth=8,                # Optimal tree depth
    min_samples_split=3,        # Stricter split criteria
    min_samples_leaf=1,         # Allow single samples in leaves
    subsample=0.95,             # Use 95% of samples for each tree
    random_state=42,
    verbose=0,
    warm_start=False
)

# Compute sample weights to balance classes
sample_weights = compute_sample_weight('balanced', y_train)

model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"[RESULT] Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"[RESULT] F1 Score (weighted): {f1:.4f}")

# Cross-validation for robust evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
print(f"[RESULT] Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_fert.classes_))

# Check class distribution
print("\n[INFO] Class Distribution in Dataset:")
print(y.value_counts())
print(f"\nClass Balance: {y.value_counts(normalize=True)}")

# Feature importance analysis
print("\n[INFO] Feature Importance:")
feature_names = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
importances = model.feature_importances_
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.4f}")

# Detailed accuracy metrics
print("\n[RESULT] Detailed Metrics:")
print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Weighted F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Micro F1 Score: {f1_score(y_test, y_pred, average='micro'):.4f}")

# Accuracy check
min_accuracy = 0.95
if accuracy < min_accuracy:
    print(f"\n[WARNING] Model accuracy ({accuracy*100:.2f}%) is below {min_accuracy*100:.0f}%")
else:
    print(f"\n[SUCCESS] Model accuracy ({accuracy*100:.2f}%) meets minimum threshold of {min_accuracy*100:.0f}%")

print("\n[INFO] NOTE: Class weighting applied to address imbalance.")
print("   Model trained with balanced class weights to improve F1 scores.")
print("   High accuracy with improved F1 scores indicates model is better balanced.")

# Save everything
print("\n[INFO] Saving model and encoders...")
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save scaler for use in prediction
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")
joblib.dump(le_fert, "fertilizer_encoder.pkl")

print("[SUCCESS] — model.pkl CREATED with optimized accuracy and balanced F1 scores!")
print(f"Model Type: Gradient Boosting Classifier (Optimized with Class Weighting)")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Weighted F1 Score: {f1:.4f}")
print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
