import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
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

# Fonction pour définir l'image d'arrière-plan
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Définir l'image d'arrière-plan
set_background('Redal-1.jpg')

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

# Modèles à tester
models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=100),
    'LinearRegression': LinearRegression()
}

# Entraîner les modèles pour les cibles minimales
for model in models.values():
    model.fit(X_train, y_train_min)

# Interface Streamlit
st.title("Prédiction des Scénarios de Construction")

# Afficher les logos dans l'entête
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("veolia_logo.png", use_column_width=True)
with col2:
    st.title("GOBIM")
with col3:
    st.image("redal_logo.png", use_column_width=True)

st.markdown("---")

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
predictions_min = {}
predictions_max = {}

# Prédire les coûts minimaux
for name, model in models.items():
    predictions_min[name] = model.predict(scenarios)[0]

# Entraîner les modèles pour les cibles maximales
for model in models.values():
    model.fit(X_train, y_train_max)

# Prédire les coûts maximaux
for name, model in models.items():
    predictions_max[name] = model.predict(scenarios)[0]

# Afficher les résultats
st.header("Résultats des Prédictions (MT Min H.T.V.A.)")
for name, prediction in predictions_min.items():
    st.write(f"{name} : {prediction:.2f}")

st.header("Résultats des Prédictions (MT Max H.T.V.A.)")
for name, prediction in predictions_max.items():
    st.write(f"{name} : {prediction:.2f}")

# Visualisation des prédictions pour les scénarios
st.subheader("Visualisation des Prédictions")
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].bar(predictions_min.keys(), predictions_min.values(), color=['green', 'red', 'purple', 'orange'])
ax[0].set_title("Prédictions MT Min H.T.V.A.")
ax[0].set_ylabel("MT Min H.T.V.A.")

ax[1].bar(predictions_max.keys(), predictions_max.values(), color=['green', 'red', 'purple', 'orange'])
ax[1].set_title("Prédictions MT Max H.T.V.A.")
ax[1].set_ylabel("MT Max H.T.V.A.")

st.pyplot(fig)
