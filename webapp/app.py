import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io
import pdfplumber

st.set_page_config(
    page_title="Dashboard Matching CV-Offres",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


API_URL = "http://serving-api:8080"

ARTIFACTS_PATH = Path("/app/artifacts")  
DATA_PATH = Path("/app/data")           
HISTORY_FILE = "/app/.analysis_history.json" 

st.markdown("""
<style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .highlight-score {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_pca_data():
    """Charge les données PCA et les embeddings."""
    try:
        if ARTIFACTS_PATH.exists():
            pca_file = ARTIFACTS_PATH / "pca.pickle"
            if pca_file.exists():
                with open(pca_file, 'rb') as f:
                    pca = pickle.load(f)
                return pca
    except Exception as e:
        st.warning(f"Impossible de charger la PCA : {e}")
    return None

def load_reference_data():
    """Charge les données de référence (embeddings des offres)."""
    try:
        csv_file = DATA_PATH / "ref_data.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file)
    except Exception as e:
        st.warning(f"Impossible de charger les données de référence : {e}")
    return None

def load_history():
    """Charge l'historique des analyses."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    """Sauvegarde l'historique des analyses."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def extract_cv_text(pdf_file):
    """Extrait le texte d'un fichier PDF."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du PDF : {e}")
        return None

def call_api_predict(cv_file, job_id, trigger_n8n=True):
    """Appelle l'API de prédiction."""
    try:
        with st.spinner("Calcul de la similarité..."):
            files = {"fichier_cv": (cv_file.name, cv_file.getvalue(), "application/pdf")}
            data = {
                "id_offre": job_id,
                "trigger_n8n": str(trigger_n8n).lower()
            }
            response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Erreur API : {response.text}")
                return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return None

def tab_prediction():
    st.header("🎯 Prédiction de Score de Matching")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Étape 1 : Télécharger un CV")
        uploaded_cv = st.file_uploader("Choisir un fichier PDF", type="pdf", key="cv_upload")
        
        if uploaded_cv:
            st.success(f"✅ Fichier sélectionné : {uploaded_cv.name}")
            with st.expander("👁️ Aperçu du texte extrait"):
                cv_text = extract_cv_text(uploaded_cv)
                if cv_text:
                    st.text_area("Texte du CV", cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text, height=150, disabled=True)
    
    with col2:
        st.subheader("💼 Étape 2 : Sélectionner une offre")
        
        jobs_dir = DATA_PATH / "jobs_corpus"
        job_options = [f.name for f in jobs_dir.glob("*.pdf")] if jobs_dir.exists() else []
        if not job_options:
            job_options = ["offre_1.pdf", "offre_2.pdf", "offre_3.pdf", "offre_4.pdf", "offre_5.pdf"]
            
        selected_job = st.selectbox("Choisir une offre", job_options)
        
        st.info(f"Offre sélectionnée : **{selected_job}**")
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔢 Calculer le score seulement", use_container_width=True):
            if uploaded_cv and selected_job:
                result = call_api_predict(uploaded_cv, selected_job, trigger_n8n=False)
                if result:
                    display_prediction_results(result, uploaded_cv, selected_job, triggered_n8n=False)
            else:
                st.warning("⚠️ Veuillez sélectionner un CV et une offre")

    with col_btn2:
        if st.button("🚀 Lancer l'analyse et notification", use_container_width=True, type="primary"):
            if uploaded_cv and selected_job:
                result = call_api_predict(uploaded_cv, selected_job, trigger_n8n=True)
                if result:
                    display_prediction_results(result, uploaded_cv, selected_job, triggered_n8n=True)
            else:
                st.warning("⚠️ Veuillez sélectionner un CV et une offre")

def display_prediction_results(result, uploaded_cv, selected_job, triggered_n8n=True):
    """Affiche les résultats de la prédiction et gère l'historique."""
    score = result.get('similarity_score', 0)
    
    st.markdown("---")
    col_score1, col_score2 = st.columns(2)
    
    with col_score1:
        st.markdown(f"### Score de similarité")
        progress_val = min(max(score, 0), 1)
        st.progress(progress_val)
        st.markdown(f"<div class='highlight-score'>{score:.2%}</div>", unsafe_allow_html=True)
    
    with col_score2:
        st.markdown("### 📊 Interprétation")
        if score >= 0.8:
            st.success("🟢 **Excellente correspondance**")
        elif score >= 0.6:
            st.info("🟡 **Bonne correspondance**")
        else:
            st.warning("🔴 **Correspondance faible**")
    
    history = load_history()
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "cv_name": uploaded_cv.name,
        "job_id": selected_job,
        "score": float(score),
        "status": "pending" if triggered_n8n else "direct_score"
    }
    history.append(new_entry)
    save_history(history)
    
    st.success("✅ Analyse sauvegardée dans l'historique")
    
    if triggered_n8n:
        st.markdown("---")
        st.markdown("### 📨 Workflow Validation")
        st.info("Un email de validation a été envoyé. Cliquez sur le lien pour confirmer.")
    else:
        st.info("ℹ️ Note : Le workflow d'explication n8n n'a pas été lancé pour ce calcul rapide.")


def tab_pca_visualization():
    st.header("📈 Visualisation PCA Interactive")
    st.markdown("Explorez les embeddings des offres dans un espace 2D")
    
    ref_data = load_reference_data()
    
    if ref_data is None:
        st.warning("⚠️ Les données PCA ne sont pas disponibles. Veuillez exécuter le script d'initialisation.")
        st.code("python scripts/init_model.py")
        return

    required_cols = {"cv_text", "job_text", "similarity_score", "cv_pca_1", "cv_pca_2", "job_pca_1", "job_pca_2"}
    missing_cols = sorted(required_cols - set(ref_data.columns))
    if missing_cols:
        st.warning("⚠️ Colonnes PCA manquantes dans ref_data.csv : " + ", ".join(missing_cols))
        return

    jobs = (
        ref_data.groupby(["job_text", "job_pca_1", "job_pca_2"], as_index=False)["similarity_score"]
        .mean()
        .rename(columns={"job_pca_1": "pca_x", "job_pca_2": "pca_y", "job_text": "label", "similarity_score": "score_moyen"})
    )
    jobs["type_document"] = "Offre"

    cvs = (
        ref_data.groupby(["cv_text", "cv_pca_1", "cv_pca_2"], as_index=False)["similarity_score"]
        .mean()
        .rename(columns={"cv_pca_1": "pca_x", "cv_pca_2": "pca_y", "cv_text": "label", "similarity_score": "score_moyen"})
    )
    cvs["type_document"] = "CV"

    df_pca = pd.concat([jobs, cvs], ignore_index=True)

    def infer_category(text):
        txt = str(text).lower()
        if "data" in txt:
            return "Data"
        if "front" in txt or "react" in txt:
            return "Frontend"
        if "devops" in txt:
            return "DevOps"
        if "back" in txt or "api" in txt:
            return "Backend"
        return "Autre"

    df_pca["categorie"] = df_pca["label"].apply(infer_category)
    df_pca["label_court"] = df_pca["label"].str.slice(0, 80)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        color_by = st.selectbox("Colorer par :", ["Type document", "Score moyen", "Catégorie"])
    with col2:
        point_size = st.slider("Taille des points", 5, 20, 10)
    with col3:
        show_labels = st.checkbox("Afficher les labels", value=True)
    
    st.divider()
    
    color_column = "type_document"
    if color_by == "Score moyen":
        color_column = "score_moyen"
    elif color_by == "Catégorie":
        color_column = "categorie"

    fig = px.scatter(
        df_pca,
        x='pca_x',
        y='pca_y',
        color=color_column,
        hover_data={
            "type_document": True,
            "score_moyen": ":.3f",
            "label_court": True,
            "pca_x": ":.3f",
            "pca_y": ":.3f",
            "label": False,
        },
        title="Projection PCA 2D des CV et Offres",
        labels={'pca_x': 'PC1', 'pca_y': 'PC2'},
        size_max=point_size
    )

    if show_labels:
        fig.update_traces(text=df_pca["type_document"], textposition="top center")
    
    fig.update_layout(
        hovermode='closest',
        height=700,
        showlegend=True,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Statistiques")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Nombre d'offres", len(jobs))
    with col_stats2:
        pca = load_pca_data()
        if pca is not None and hasattr(pca, "explained_variance_ratio_"):
            st.metric("Variance expliquée", f"{sum(pca.explained_variance_ratio_):.1%}")
        else:
            st.metric("Variance expliquée", "N/A")
    with col_stats3:
        st.metric("Nombre de CV", len(cvs))


def tab_history():
    st.header("📝 Historique des Analyses")
    
    history = load_history()
    
    if not history:
        st.info("📭 Aucune analyse enregistrée pour le moment.")
        st.markdown("Lancez une analyse depuis l'onglet **Prédiction** pour commencer.")
        return
    
    df_history = pd.DataFrame(history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    df_history = df_history.sort_values('timestamp', ascending=False)
    
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        status_filter = st.multiselect("Filtrer par statut", df_history['status'].unique(), default=df_history['status'].unique())
    
    with col_filter2:
        score_min = st.slider("Score minimum", 0.0, 1.0, 0.0)
    
    df_filtered = df_history[
        (df_history['status'].isin(status_filter)) &
        (df_history['score'] >= score_min)
    ]
    
    st.divider()
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Total analyses", len(df_history))
    with col_stat2:
        st.metric("Score moyen", f"{df_history['score'].mean():.2%}")
    with col_stat3:
        st.metric("Score max", f"{df_history['score'].max():.2%}")
    with col_stat4:
        pending = len(df_history[df_history['status'] == 'pending'])
        st.metric("En attente de validation", pending)
    
    st.divider()
    
    st.subheader("📋 Liste des analyses")
    
    display_df = df_filtered.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.2%}")
    display_df = display_df[['timestamp', 'cv_name', 'job_id', 'score', 'status']]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("📊 Évolution des scores")
        df_plot = df_history.sort_values('timestamp')
        fig_scores = px.line(
            df_plot,
            x='timestamp',
            y='score',
            markers=True,
            title="Évolution des scores de similarité"
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col_viz2:
        st.subheader("🎯 Distribution des scores")
        fig_dist = px.histogram(
            df_history,
            x='score',
            nbins=20,
            title="Distribution des scores",
            labels={'score': 'Score de similarité', 'count': 'Nombre'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.divider()
    col_manage1, col_manage2, col_manage3 = st.columns(3)
    
    with col_manage1:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()
    
    with col_manage2:
        if st.button("📥 Exporter en CSV", use_container_width=True):
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name="historique_analyses.csv",
                mime="text/csv"
            )
    
    with col_manage3:
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if st.session_state.confirm_clear:
            st.warning("⚠️ Êtes-vous sûr ?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Oui", use_container_width=True):
                    save_history([])
                    st.session_state.confirm_clear = False
                    st.success("Historique supprimé")
                    st.rerun()
            with col_no:
                if st.button("❌ Non", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
        else:
            if st.button("🗑️ Effacer l'historique", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()

def main():
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        st.markdown("### 🔌 Statut API")
        try:
            resp = requests.get(f"{API_URL}/", timeout=2)
            st.success("✅ API connectée")
        except:
            st.error("❌ API indisponible")
            st.info("Lancez le backend : `python serving/api.py`")
        
        st.markdown("---")
        st.markdown("### 📚 À propos")
        st.caption("Dashboard Matching CV-Offres v1.0")
        st.caption("Développé pour le projet n8n automation")
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 Prédiction",
        "📈 Visualisation PCA",
        "📝 Historique"
    ])
    
    with tab1:
        tab_prediction()
    
    with tab2:
        tab_pca_visualization()
    
    with tab3:
        tab_history()

if __name__ == "__main__":
    main()
