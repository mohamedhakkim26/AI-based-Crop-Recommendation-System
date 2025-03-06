import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train model (this can be done separately and saved to avoid retraining each time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model (you can comment this out after the first run to avoid re-training)
pickle.dump(model, open("model.pkl", "wb"))

# Load the model (uncomment this line after model is trained)
model = pickle.load(open("model.pkl", "rb"))

# Function to predict crop based on features input
def predict_crop(features):
    # Check if the input features are in the correct format (a list of numbers)
    if len(features) != len(X.columns):
        return "Error: The number of features provided does not match the model input."

    # Convert features to a numpy array and reshape it
    features_array = np.array(features).reshape(1, -1)

    # Create a DataFrame with the same column names as the training set
    features_df = pd.DataFrame(features_array, columns=X.columns)

    # Predict using the trained model
    prediction = model.predict(features_df)

    return prediction[0]  # Return the predicted crop



