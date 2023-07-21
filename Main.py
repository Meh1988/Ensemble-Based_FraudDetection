import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
data = pd.read_csv('IFilecreditcard.csv')

# Split data into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Number of features and classes
num_features = X_train.shape[1]
num_classes = 1  # Binary classification (fraud or not fraud)

# Define the base neural network model
def build_base_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(num_features,)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the ensemble by training multiple base models
num_models = 5
ensemble_models = []
for i in range(num_models):
    model = build_base_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    ensemble_models.append(model)

# Make predictions on the test set using each model
y_pred_list = []
for model in ensemble_models:
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)

# Combine predictions using voting (majority vote)
y_pred_ensemble = np.mean(y_pred_list, axis=0)
y_pred_ensemble = np.where(y_pred_ensemble >= 0.5, 1, 0)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, y_pred_ensemble)
conf_matrix = confusion_matrix(y_test, y_pred_ensemble)

print("Ensemble Model Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
