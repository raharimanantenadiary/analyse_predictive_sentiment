# ========================================
# ÉTAPE 6 : APPLICATION STREAMLIT
# Projet : Analyse de Sentiment Twitter
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment Twitter",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 1. CHARGEMENT DU MODÈLE ET VECTORISEUR
# ========================================

@st.cache_resource
def charger_modele_et_vectoriseur():
    """
    Charge le modèle et le vectoriseur sauvegardés
    """
    try:
        with open('modele_sentiment.pkl', 'rb') as fichier:
            modele = pickle.load(fichier)
        
        with open('vectoriseur_tfidf.pkl', 'rb') as fichier:
            vectoriseur = pickle.load(fichier)
        
        return modele, vectoriseur
    except FileNotFoundError:
        st.error("❌ Fichiers du modèle introuvables. Vérifie que 'modele_sentiment.pkl' et 'vectoriseur_tfidf.pkl' sont dans le même dossier.")
        return None, None

# Charger le modèle
modele, vectoriseur = charger_modele_et_vectoriseur()

# ========================================
# 2. FONCTIONS DE PREPROCESSING
# ========================================

# Initialiser les ressources NLTK
mots_vides = set(stopwords.words('english'))
mots_vides.update({'rt', 'via', 'amp'})
lemmatiseur = WordNetLemmatizer()

def nettoyer_texte(texte):
    """Nettoie le texte"""
    texte = texte.lower()
    texte = re.sub(r'http\S+|www\S+|https\S+', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'@\w+', '', texte)
    texte = re.sub(r'#', '', texte)
    texte = re.sub(r'\d+', '', texte)
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    texte = re.sub(r'\s+', ' ', texte)
    texte = texte.strip()
    return texte

def supprimer_stopwords(texte):
    """Supprime les stopwords"""
    mots = texte.split()
    mots_filtres = [mot for mot in mots if mot not in mots_vides]
    return ' '.join(mots_filtres)

def lemmatiser_texte(texte):
    """Lemmatise le texte"""
    mots = texte.split()
    mots_lemmatises = [lemmatiseur.lemmatize(mot) for mot in mots]
    return ' '.join(mots_lemmatises)

def preprocesser_texte(texte):
    """Pipeline complet de preprocessing"""
    texte = nettoyer_texte(texte)
    texte = supprimer_stopwords(texte)
    texte = lemmatiser_texte(texte)
    return texte

# ========================================
# 3. FONCTION DE PRÉDICTION
# ========================================

def predire_sentiment(texte):
    """
    Prédit le sentiment d'un tweet
    
    Arguments:
        texte (str) : Le tweet à analyser
    
    Retourne:
        tuple : (sentiment, probabilités)
    """
    if modele is None or vectoriseur is None:
        return None, None
    
    # Preprocessing
    texte_preprocesse = preprocesser_texte(texte)
    
    # Vectorisation
    texte_vectorise = vectoriseur.transform([texte_preprocesse])
    
    # Prédiction
    prediction = modele.predict(texte_vectorise)[0]
    probabilites = modele.predict_proba(texte_vectorise)[0]
    
    return prediction, probabilites, texte_preprocesse

# ========================================
# 4. INTERFACE UTILISATEUR
# ========================================

# Titre principal
st.title("🐦 Analyse de Sentiment Twitter")
st.markdown("### 📊 Classification automatique de tweets en 3 catégories")
st.markdown("---")

# Vérifier que le modèle est chargé
if modele is None or vectoriseur is None:
    st.stop()

# ========================================
# 5. SIDEBAR - INFORMATIONS DU MODÈLE
# ========================================

with st.sidebar:
    st.header("📋 Informations du Modèle")
    
    st.metric("🎯 Accuracy", "69.36%")
    st.metric("🎯 Precision", "70.45%")
    st.metric("🎯 Recall", "69.36%")
    st.metric("🎯 F1-Score", "69.35%")
    
    st.markdown("---")
    
    st.subheader("🔧 Technologies utilisées")
    st.markdown("""
    - **Algorithme** : Régression Logistique
    - **Vectorisation** : TF-IDF
    - **Features** : 5000 mots
    - **N-grams** : Unigrammes + Bigrammes
    """)
    
    st.markdown("---")
    
    st.subheader("📚 Classes")
    classes = modele.classes_
    for i, classe in enumerate(classes):
        if classe.lower() in ['positive', 'positif']:
            st.success(f"✅ {classe}")
        elif classe.lower() in ['negative', 'negatif', 'négatif']:
            st.error(f"❌ {classe}")
        else:
            st.info(f"➖ {classe}")

# ========================================
# 6. ZONE PRINCIPALE - PRÉDICTION
# ========================================

# Onglets
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📊 Exemples", "ℹ️ À propos"])

# TAB 1 : PRÉDICTION
with tab1:
    st.header("Analyser un nouveau tweet")
    
    # Zone de texte
    texte_utilisateur = st.text_area(
        "Entrez votre tweet ici :",
        height=150,
        placeholder="Exemple : I love this product! It's amazing and works perfectly!"
    )
    
    # Bouton de prédiction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        bouton_predire = st.button("🚀 Analyser le sentiment", use_container_width=True)
    
    if bouton_predire and texte_utilisateur.strip():
        with st.spinner("Analyse en cours..."):
            # Prédiction
            sentiment, probabilites, texte_preprocesse = predire_sentiment(texte_utilisateur)
            
            # Affichage des résultats
            st.markdown("---")
            st.subheader("📊 Résultats de l'analyse")
            
            # Sentiment prédit
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🎯 Sentiment détecté")
                
                # Couleur selon le sentiment
                if sentiment.lower() in ['positive', 'positif']:
                    st.success(f"# {sentiment.upper()}")
                    emoji = "😊"
                elif sentiment.lower() in ['negative', 'negatif', 'négatif']:
                    st.error(f"# {sentiment.upper()}")
                    emoji = "😞"
                else:
                    st.info(f"# {sentiment.upper()}")
                    emoji = "😐"
                
                st.markdown(f"### {emoji}")
            
            with col2:
                st.markdown("#### 📈 Probabilités")
                
                # Créer un DataFrame pour les probabilités
                df_probas = pd.DataFrame({
                    'Sentiment': modele.classes_,
                    'Probabilité (%)': probabilites * 100
                })
                
                # Afficher les probabilités
                for idx, row in df_probas.iterrows():
                    st.metric(
                        label=row['Sentiment'],
                        value=f"{row['Probabilité (%)']:.2f}%"
                    )
            
            # Graphique des probabilités
            st.markdown("---")
            st.markdown("#### 📊 Visualisation des probabilités")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#2ecc71' if s.lower() in ['positive', 'positif'] 
                     else '#e74c3c' if s.lower() in ['negative', 'negatif', 'négatif']
                     else '#95a5a6' 
                     for s in modele.classes_]
            
            bars = ax.barh(modele.classes_, probabilites * 100, color=colors)
            ax.set_xlabel('Probabilité (%)', fontweight='bold')
            ax.set_title('Distribution des probabilités par sentiment', fontweight='bold', pad=20)
            ax.set_xlim(0, 100)
            
            # Ajouter les valeurs sur les barres
            for i, (bar, proba) in enumerate(zip(bars, probabilites)):
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                       f'{proba*100:.2f}%', 
                       ha='left', va='center', fontweight='bold')
            
            st.pyplot(fig)
            
            # Détails du preprocessing
            with st.expander("🔍 Voir les détails du preprocessing"):
                st.markdown("**Texte original :**")
                st.code(texte_utilisateur)
                
                st.markdown("**Texte après preprocessing :**")
                st.code(texte_preprocesse)
                
                st.markdown("**Transformations appliquées :**")
                st.markdown("""
                1. ✅ Conversion en minuscules
                2. ✅ Suppression des URLs
                3. ✅ Suppression des mentions (@)
                4. ✅ Suppression des hashtags (#)
                5. ✅ Suppression de la ponctuation
                6. ✅ Suppression des stopwords
                7. ✅ Lemmatisation
                """)
    
    elif bouton_predire:
        st.warning("⚠️ Veuillez entrer un tweet à analyser.")

# TAB 2 : EXEMPLES
with tab2:
    st.header("Exemples de tweets par sentiment")
    
    exemples = {
        "Positif": [
            "I love this product! It's amazing and works perfectly!",
            "Best experience ever! Highly recommend to everyone!",
            "Absolutely fantastic service, very happy with my purchase!"
        ],
        "Négatif": [
            "This is the worst experience ever. Totally disappointed.",
            "Terrible product, waste of money. Would not recommend.",
            "Very bad quality and poor customer service."
        ],
        "Neutre": [
            "It's okay, nothing special but not bad either.",
            "The product arrived on time. Standard quality.",
            "Average experience, meets basic expectations."
        ]
    }
    
    for sentiment, tweets in exemples.items():
        if sentiment == "Positif":
            st.success(f"### ✅ {sentiment}")
        elif sentiment == "Négatif":
            st.error(f"### ❌ {sentiment}")
        else:
            st.info(f"### ➖ {sentiment}")
        
        for i, tweet in enumerate(tweets, 1):
            st.markdown(f"**{i}.** {tweet}")
        
        st.markdown("")

# TAB 3 : À PROPOS
with tab3:
    st.header("ℹ️ À propos du projet")
    
    st.markdown("""
    ### 🎯 Objectif
    Ce projet vise à classifier automatiquement des tweets en 3 catégories de sentiment :
    - **Positif** 😊 : Expressions de satisfaction, joie, approbation
    - **Négatif** 😞 : Expressions de mécontentement, tristesse, désapprobation
    - **Neutre** 😐 : Expressions factuelles sans émotion marquée
    
    ### 🛠️ Méthodologie
    
    #### 1. **Preprocessing NLP**
    - Nettoyage des tweets (URLs, mentions, ponctuation)
    - Suppression des stopwords
    - Lemmatisation des mots
    
    #### 2. **Vectorisation TF-IDF**
    - Transformation des textes en vecteurs numériques
    - Identification des mots importants
    - 5000 features extraites
    
    #### 3. **Classification**
    - Algorithme : Régression Logistique
    - Entraînement sur 80% des données
    - Test sur 20% des données
    
    ### 📊 Performances
    - **Accuracy** : 69.36%
    - **Precision** : 70.45%
    - **Recall** : 69.36%
    - **F1-Score** : 69.35%
    
    ### 🚀 Technologies
    - Python 3.x
    - Scikit-learn
    - NLTK
    - Streamlit
    - Pandas
    
    ### 👨‍💻 Développeur
    Projet d'analyse de sentiment sur Twitter
    """)

# ========================================
# 7. FOOTER
# ========================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🐦 Analyse de Sentiment Twitter | Projet NLP | 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)