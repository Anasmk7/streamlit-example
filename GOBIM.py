import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import base64

# Fonction pour charger les données
@st.cache
def load_data(file_path):
    data = pd.read_excel(file_path, skiprows=2)
    data.columns = ['Code', 'Designation', 'Unité', 'Qté Min', 'Qté Max', 'P.U. H.T.V.A.', 'MT Min H.T.V.A.', 'MT Max H.T.V.A.', 'Autres']
    data = data.dropna(subset=['Code', 'Designation'])
    data['Qté Min'] = pd.to_numeric(data['Qté Min'], errors='coerce').fillna(0)
    data['Qté Max'] = pd.to_numeric(data['Qté Max'], errors='coerce').fillna(0)
    data['P.U. H.T.V.A.'] = pd.to_numeric(data['P.U. H.T.V.A.'], errors='coerce').fillna(0)
    data['MT Min H.T.V.A.'] = pd.to_numeric(data['MT Min H.T.V.A.'], errors='coerce').fillna(0)
    data['MT Max H.T.V.A.'] = pd.to_numeric(data['MT Max H.T.V.A.'], errors='coerce').fillna(0)
    return data

# Charger les données
file_path = 'BP ESTIMATIF DATA BINGA .xlsx'  # Remplacez par le chemin de votre fichier
data = load_data(file_path)

# Définir les caractéristiques (features) et les cibles (targets)
features = data[['Qté Min', 'Qté Max', 'P.U. H.T.V.A.']]
target_min = data['MT Min H.T.V.A.']
target_max = data['MT Max H.T.V.A.']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train_min, y_test_min, y_train_max, y_test_max = train_test_split(
    features, target_min, target_max, test_size=0.2, random_state=42
)

# Définir les paramètres de la grille de recherche pour RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Fonction pour effectuer la recherche sur grille avec validation croisée
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Optimiser les modèles RandomForest pour les cibles minimales et maximales
best_rf_min = perform_grid_search(RandomForestRegressor(random_state=42), param_grid_rf, X_train, y_train_min)
best_rf_max = perform_grid_search(RandomForestRegressor(random_state=42), param_grid_rf, X_train, y_train_max)

# Entraîner les modèles optimisés
best_rf_min.fit(X_train, y_train_min)
best_rf_max.fit(X_train, y_train_max)

# Interface Streamlit
st.set_page_config(page_title="Prédiction des Coûts de Construction", page_icon="🏗️")

# Afficher le logo en haut de l'application
st.image("redal_logo.png", width=300)

st.title("Prédiction des Scénarios de Construction")

# Saisir les valeurs des scénarios
st.header("Saisir les valeurs des scénarios")
qte_min_scenario = st.slider("Quantité Minimale (Scénario)", min_value=0, max_value=10000, value=100)
qte_max_scenario = st.slider("Quantité Maximale (Scénario)", min_value=0, max_value=10000, value=200)
pu_htva_scenario = st.slider("Prix Unitaire H.T.V.A. (Scénario)", min_value=0.0, max_value=10000.0, value=50.0)

# Simuler plusieurs scénarios
scenarios = pd.DataFrame({
    'Qté Min': [qte_min_scenario],
    'Qté Max': [qte_max_scenario],
    'P.U. H.T.V.A.': [pu_htva_scenario]
})

# Prédire les coûts minimaux et maximaux pour chaque scénario
predictions_min = best_rf_min.predict(scenarios)[0]
predictions_max = best_rf_max.predict(scenarios)[0]

# Afficher les résultats
st.header("Résultats des Prédictions (MT Min H.T.V.A.)")
st.write(f"RandomForest : {predictions_min:.2f}")

st.header("Résultats des Prédictions (MT Max H.T.V.A.)")
st.write(f"RandomForest : {predictions_max:.2f}")

# Visualisation des prédictions pour les scénarios
st.subheader("Visualisation des Prédictions")
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].bar(['RandomForest'], [predictions_min], color=['green'])
ax[0].set_title("Prédictions MT Min H.T.V.A.")
ax[0].set_ylabel("MT Min H.T.V.A.")

ax[1].bar(['RandomForest'], [predictions_max], color=['red'])
ax[1].set_title("Prédictions MT Max H.T.V.A.")
ax[1].set_ylabel("MT Max H.T.V.A.")

st.pyplot(fig)
