import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model

# ----------------------------
# Page config (doit être avant tout st.*)
# ----------------------------
st.set_page_config(
    page_title="Poubelle Intelligente",
    page_icon="♻️",
    layout="centered",
)

# ----------------------------
# Styles légers (design plus pro)
# ----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 920px; }
      .small-muted { color: #6b7280; font-size: 0.9rem; }
      .card { padding: 1rem 1.2rem; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar (réglages)
# ----------------------------
with st.sidebar:
    st.header("Réglages")
    threshold = st.slider("Seuil de décision", 0.10, 0.90, 0.50, 0.05)
    st.caption("Si la probabilité *recyclable* ≥ seuil → classé recyclable.")
    show_details = st.toggle("Afficher les détails", value=True)

# ----------------------------
# Titre
# ----------------------------
st.title("♻️ Poubelle Intelligente")
st.markdown('<div class="small-muted">Analyse d’image : estimation “recyclable / non recyclable”.</div>', unsafe_allow_html=True)

# ----------------------------
# Chargement du modèle (cache adapté aux ressources)
# ----------------------------
MODEL_PATH = Path("best_model.h5")

@st.cache_resource
def load_my_model(model_path: str):
    # compile=False : inference uniquement
    return load_model(model_path, compile=False)

if not MODEL_PATH.exists():
    st.error("❌ Modèle introuvable : `best_model.h5` n’est pas dans le même dossier que `app.py`.")
    st.info("Place `best_model.h5` à côté de `app.py`, puis relance : `streamlit run app.py`.")
    st.stop()

model = load_my_model(str(MODEL_PATH))

# ----------------------------
# Prétraitement (sans OpenCV -> simple, stable, moins de warnings)
# ----------------------------
def preprocess_image(uploaded_file, target_size=(224, 224)) -> np.ndarray:
    """
    Retourne un tenseur (1, H, W, 3) normalisé [0,1].
    """
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)  # PIL attend (width, height) mais ici on met carré donc OK
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr

def predict_recyclable_prob(uploaded_file) -> float:
    x = preprocess_image(uploaded_file)
    pred = model.predict(x, verbose=0)

    # Cas le plus courant : sortie sigmoid => shape (1,1)
    # Si jamais ton modèle renvoie autre chose, on sécurise.
    p = float(np.squeeze(pred))

    # On borne au cas où
    p = max(0.0, min(1.0, p))
    return p

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Chargez une image (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

st.markdown('<div class="card">', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👆 Ajoute une image pour lancer la prédiction.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

col1, col2 = st.columns([1, 1], vertical_alignment="top")

with col1:
    st.subheader("Image")
    # ✅ Paramètre moderne : width="stretch" (évite le warning use_column_width)
    st.image(uploaded_file, caption="Image chargée", width="stretch")

with col2:
    st.subheader("Résultat")
    with st.spinner("Analyse en cours…"):
        try:
            p_rec = predict_recyclable_prob(uploaded_file)
        except Exception as e:
            st.error("Erreur pendant la prédiction.")
            st.exception(e)
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

    p_non = 1.0 - p_rec

    # Décision
    is_recyclable = p_rec >= threshold

    # Affichage principal
    if is_recyclable:
        st.success("✅ Classé : **Recyclable**")
    else:
        st.error("🗑️ Classé : **Non recyclable**")

    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Prob. Recyclable", f"{p_rec*100:.1f}%")
    m2.metric("Prob. Non recyclable", f"{p_non*100:.1f}%")

    # Barre de confiance (progress attend 0..100 int)
    st.caption("Confiance (recyclable)")
    st.progress(int(round(p_rec * 100)))

    if show_details:
        with st.expander("Détails & debug"):
            st.write("**Seuil** :", threshold)
            st.write("**Sortie modèle (p_recyclable)** :", p_rec)
            st.write("**Forme attendue entrée** : (1, 224, 224, 3)")
            st.write("⚠️ Important : si ton modèle a été entraîné avec un prétraitement spécifique "
                     "(ex: `preprocess_input` de VGG/ResNet), il faut reproduire exactement le même ici.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown('<div class="small-muted">Astuce : si tu modifies le code, Streamlit recharge automatiquement.</div>', unsafe_allow_html=True)