import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the crop dataset
data = pd.read_csv("crop_data.csv")

# Separate input features and crop labels
X = data.drop("label", axis=1)
y = data["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check model performance
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Crop recommendation model trained successfully!")
print(f"Test accuracy: {accuracy:.2%}")

# Save the trained model
joblib.dump(model, "crop_model.pkl")

print("Model saved as crop_model.pkl")