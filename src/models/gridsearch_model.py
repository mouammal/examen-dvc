import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Chargement des données
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# y_train est un dataframe à une seule colonne, on le convertit en vecteur
if y_train.shape[1] == 1:
    y_train = y_train.values.ravel()

# Définition du modèle
model = RandomForestRegressor(random_state=42)

# Grille d'hyperparamètres à tester
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

# Entraînement
grid_search.fit(X_train_scaled, y_train)

# Résultat
print("Best params found:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)

# Sauvegarder le meilleur modèle
best_model_path = "src/models/best_model.pkl"
joblib.dump(grid_search.best_estimator_, best_model_path)
print(f"Best model saved to {best_model_path}")