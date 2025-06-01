import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Chargement des données
X_test = pd.read_csv("data/processed_data/X_test.csv")
X_test_scaled = pd.read_csv("data/scaled_data/X_test_scaled.csv")

y_test = pd.read_csv("data/processed_data/y_test.csv")
y_test = np.ravel(y_test)

# Chargement du modèle entraîné
model_path = "src/models/trained_model.pkl"
model_trained = joblib.load(model_path)

# Prédictions
predictions = model_trained.predict(X_test_scaled)

# Évaluation
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Affichage des résultats
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")

# Sauvegarde des scores
scores = {
    "test_mse": mse,
    "test_mae": mae,
    "test_r2": r2
}
scores_path = "metrics/scores.json"
with open(scores_path, "w") as f:
    json.dump(scores, f, indent=4)
print("Scores enregistrés dans metrics/test_scores.json")

# Sauvegarde des prédictions
df_preds = X_test.copy()
df_preds["true"] = y_test
df_preds["prediction"] = predictions

df_preds.to_csv("models/test_predictions.csv", index=False)
print("Prédictions sauvegardées dans data/predictions/test_predictions.csv")