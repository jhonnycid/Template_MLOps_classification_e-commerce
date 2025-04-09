import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import io
import plotly.express as px
from PIL import Image

# Configuration de l'application
st.set_page_config(
    page_title="Rakuten Product Classifier",
    page_icon="🛒",
    layout="wide"
)

# Variables globales
API_URL = "http://localhost:8000"  # À modifier selon votre configuration
AUTH_TOKEN = None

# Style CSS
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #FF5722;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour l'authentification
def authenticate(username, password):
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'authentification: {str(e)}")
        return None

# Fonction pour prédire la catégorie
def predict_category(description, image):
    global AUTH_TOKEN
    
    if not AUTH_TOKEN:
        st.error("Vous devez vous authentifier avant de faire une prédiction.")
        return None
    
    files = {"image": image}
    data = {"description": description}
    
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            data=data,
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la prédiction: {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

# Fonction pour obtenir les métriques
def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            # Extraire les métriques importantes
            prediction_total = 0
            prediction_success = 0
            prediction_error = 0
            
            for line in metrics_text.split("\n"):
                if line.startswith("prediction_total{status=\"success\"}"):
                    prediction_success = float(line.split(" ")[1])
                elif line.startswith("prediction_total{status=\"error\"}"):
                    prediction_error = float(line.split(" ")[1])
            
            prediction_total = prediction_success + prediction_error
            success_rate = (prediction_success / prediction_total) * 100 if prediction_total > 0 else 0
            
            return {
                "total": prediction_total,
                "success": prediction_success,
                "error": prediction_error,
                "success_rate": success_rate
            }
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des métriques: {str(e)}")
        return None

# Vérifier si l'API est accessible
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Titre principal
st.markdown('<div class="title">Classificateur de Produits Rakuten</div>', unsafe_allow_html=True)

# Sidebar pour l'authentification
with st.sidebar:
    st.markdown('<div class="subtitle">Authentification</div>', unsafe_allow_html=True)
    
    username = st.text_input("Nom d'utilisateur", value="user")
    password = st.text_input("Mot de passe", type="password", value="password")
    
    if st.button("Se connecter"):
        AUTH_TOKEN = authenticate(username, password)
        if AUTH_TOKEN:
            st.success("Authentification réussie!")
        else:
            st.error("Échec de l'authentification. Vérifiez vos identifiants.")
    
    st.markdown("---")
    st.markdown('<div class="subtitle">Navigation</div>', unsafe_allow_html=True)
    
    page = st.radio("", ["Prédiction", "Tableau de bord"])
    
    st.markdown("---")
    api_status = check_api_health()
    st.markdown(f"**Statut de l'API:** {'🟢 En ligne' if api_status else '🔴 Hors ligne'}")

# Page principale
if page == "Prédiction":
    st.markdown('<div class="subtitle">Classification de Produits</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Description du produit")
        description = st.text_area("Entrez la description du produit", height=150)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Image du produit")
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Classifier le produit"):
        if not description or uploaded_file is None:
            st.error("Veuillez fournir à la fois une description et une image.")
        else:
            with st.spinner("Classification en cours..."):
                # Réinitialiser le curseur du fichier pour pouvoir le lire à nouveau
                uploaded_file.seek(0)
                result = predict_category(description, uploaded_file)
                
                if result:
                    st.markdown('<div class="success">', unsafe_allow_html=True)
                    st.markdown(f"### Résultat de la classification")
                    st.markdown(f"**Catégorie prédite:** {result['category']} (ID: {result['prediction']})")
                    st.markdown(f"**Confiance:** {result['confidence']*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Afficher une jauge pour la confiance
                    fig = px.pie(values=[result['confidence'], 1-result['confidence']], 
                                names=['Confiance', 'Incertitude'],
                                title="Niveau de confiance",
                                color_discrete_sequence=['#28a745', '#f8f9fa'],
                                hole=0.7)
                    st.plotly_chart(fig)

elif page == "Tableau de bord":
    st.markdown('<div class="subtitle">Tableau de Bord des Performances</div>', unsafe_allow_html=True)
    
    if not check_api_health():
        st.error("L'API n'est pas accessible. Impossible de récupérer les métriques.")
    else:
        # Récupérer les métriques
        metrics = get_metrics()
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric(label="Total des prédictions", value=int(metrics["total"]))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric(label="Prédictions réussies", value=int(metrics["success"]))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric(label="Prédictions échouées", value=int(metrics["error"]))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric(label="Taux de réussite", value=f"{metrics['success_rate']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Graphique des prédictions
            fig = px.pie(
                values=[metrics["success"], metrics["error"]],
                names=["Réussies", "Échouées"],
                title="Répartition des prédictions",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
            st.plotly_chart(fig)
            
            # Lien vers Grafana
            st.markdown("""
            <div class="card">
            <h3>Monitoring avancé</h3>
            <p>Pour un monitoring plus détaillé, consultez les tableaux de bord Grafana à l'adresse <code>http://localhost:3000</code> (utilisateur: admin, mot de passe: admin).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Impossible de récupérer les métriques. Vérifiez la connexion à l'API.")

# Footer
st.markdown("---")
st.markdown(
    "Développé dans le cadre du projet MLOps DataScientest | "
    "Documentation disponible sur GitHub"
)
