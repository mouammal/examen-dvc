import pandas as pd
import joblib

# Chargement des données scalées
X_train_scaled = pd.read_csv("data/scaled_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# Chargement du meilleur modele trouvé par GridSearchCV
best_model_path = "src/models/best_model.pkl"
best_model = joblib.load(best_model_path)

# Entraînement du modèle (y_train en 1D)
best_model.fit(X_train_scaled, y_train.values.ravel())

# Sauvegarder le modèle entraîné
model_save_path = "src/models/trained_model.pkl"
joblib.dump(best_model, model_save_path)
print(f"Modèle entraîné sauvegardé dans : {model_save_path}")
