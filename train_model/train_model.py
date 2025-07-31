import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from prepare_dataset import fetch_and_engineer_features

# Step 1: Fetch and engineer data
ticker = 'INFY.NS'  # You can change this ticker or loop over multiple tickers if needed
df = fetch_and_engineer_features(ticker)

# Step 2: Drop rows with missing or infinite values
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Step 3: Define features and labels
X = df.drop('Signal', axis=1)
y = df['Signal']

# Ensure alignment after dropna
y = y[X.index]

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Step 7: Save model and scaler
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Model and scaler saved to 'models/'!")

# Step 8: Evaluate model
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔹 Classification Report:\n", classification_report(y_test, y_pred))
