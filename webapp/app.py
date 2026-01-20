import streamlit as st
import requests

# Membre 3 : Dashboard (Minimum Vital).
# Interface permettant uniquement de tester l'upload de CV et le déclenchement du workflow n8n.

st.set_page_config(page_title="Test Matching CV & n8n", page_icon="🎯")
st.title("🎯 Testeur de Workflow Match/Feedback")

# Configuration
API_URL = "http://localhost:8000"

st.info("Ce dashboard est une version simplifiée pour vérifier le fonctionnement de n8n.")

# Upload
uploaded_cv = st.file_uploader("Étape 1 : Déposer le CV (PDF)", type="pdf")
id_offre = st.text_input("Étape 2 : ID de l'offre (ex: dev_python.pdf)", value="dev_python.pdf")

if st.button("Étape 3 : Lancer le test n8n"):
    if uploaded_cv:
        with st.spinner("Simulation d'appel API..."):
            files = {"fichier_cv": (uploaded_cv.name, uploaded_cv.getvalue(), "application/pdf")}
            data = {"id_offre": id_offre}
            
            try:
                res = requests.post(f"{API_URL}/predict", files=files, data=data)
                if res.status_code == 200:
                    resultat = res.json()
                    st.success(f"Réponse API : {resultat['message']}")
                    st.write(f"Score calculé : {resultat['similarity_score']}")
                    st.markdown("---")
                    st.markdown("### 🔍 Prochaine Étape :")
                    st.markdown("1. Vérifiez vos **emails** (Gmail).")
                    st.markdown("2. Cliquez sur le lien de validation envoyé par n8n.")
                    st.markdown("3. Revenez voir les logs de l'API pour voir le feedback.")
                else:
                    st.error(f"Erreur API : {res.text}")
            except Exception as e:
                st.error(f"Erreur de connexion : {e}. Le backend FastAPI est-il lancé ?")
    else:
        st.warning("Veuillez sélectionner un fichier PDF.")

# TO-DO : Membre 3 doit implémenter ici la visualisation PCA et les graphiques de score.
st.markdown("---")
st.caption("Mode Simulation - Construction collaborative n8n")
