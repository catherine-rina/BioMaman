import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
#from tensorflow.keras.models import load_model
from tf_keras.models import load_model
import time

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BioMaman · Tri Intelligent",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Global — Thème BioMaman
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a1a0f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8f5e9;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #1b3a1f 0%, #0a1a0f 50%, #0d1f12 100%) !important;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
    background: #0d2212 !important;
    border-right: 1px solid #2a4a2e !important;
}
[data-testid="stSidebar"] * { color: #a5d6a7 !important; }

.block-container {
    padding: 2.5rem 2rem 4rem 2rem !important;
    max-width: 860px !important;
}

/* Hero */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 2.5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: #a5d6a7;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 100px;
    border: 1px solid #4caf50;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    line-height: 1.05;
    letter-spacing: -1px;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #a5d6a7 0%, #66bb6a 40%, #e8f5e9 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #81c784;
    margin-top: 0.5rem;
}
.hero-divider {
    width: 60px; height: 3px;
    background: linear-gradient(90deg, #4caf50, #a5d6a7);
    border-radius: 2px;
    margin: 1.8rem auto 0;
}

/* Upload */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #c8e6c9;
    margin-bottom: 0.5rem;
    display: block;
}
[data-testid="stFileUploader"] {
    background: rgba(46,125,50,0.08) !important;
    border: 2px dashed #2e7d32 !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #66bb6a !important; }
[data-testid="stFileUploader"] * { color: #a5d6a7 !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid #2a4a2e !important;
}

/* Result card */
.result-card {
    background: rgba(27, 62, 32, 0.55);
    border: 1px solid #2a4a2e;
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(10px);
    animation: fadeUp 0.45s ease both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
}

.verdict-recyclable {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(135deg, rgba(27,94,32,0.8), rgba(46,125,50,0.6));
    border: 1px solid #4caf50;
    border-radius: 14px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;
}
.verdict-non {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(135deg, rgba(183,28,28,0.5), rgba(198,40,40,0.3));
    border: 1px solid #ef5350;
    border-radius: 14px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;
}
.verdict-icon  { font-size: 2rem; line-height: 1; }
.verdict-label { font-size:0.72rem; font-weight:500; letter-spacing:2.5px; text-transform:uppercase; color:#a5d6a7; margin:0 0 3px; }
.verdict-text  { font-family:'Syne',sans-serif; font-weight:700; font-size:1.5rem; color:#e8f5e9; margin:0; }

.prob-row { display:flex; gap:12px; margin-bottom:1.2rem; }
.prob-box { flex:1; background:rgba(15,35,18,0.7); border:1px solid #2a4a2e; border-radius:12px; padding:0.85rem 1rem; text-align:center; }
.prob-val { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem; line-height:1.1; margin:0; }
.prob-val-green { color:#66bb6a; }
.prob-val-red   { color:#ef9a9a; }
.prob-desc { font-size:0.72rem; color:#81c784; letter-spacing:1.5px; text-transform:uppercase; font-weight:500; margin:4px 0 0; }

.bar-wrap { margin-bottom:1.2rem; }
.bar-label { display:flex; justify-content:space-between; font-size:0.78rem; color:#81c784; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; font-weight:500; }
.bar-track { background:rgba(15,35,18,0.9); border-radius:100px; height:8px; overflow:hidden; border:1px solid #1b3a1f; }
.bar-fill-green { height:100%; border-radius:100px; background:linear-gradient(90deg,#2e7d32,#66bb6a,#a5d6a7); }
.bar-fill-red   { height:100%; border-radius:100px; background:linear-gradient(90deg,#b71c1c,#e53935,#ef9a9a); }

.cat-pill { display:inline-flex; align-items:center; gap:8px; background:rgba(15,35,18,0.8); border:1px solid #2a4a2e; border-radius:100px; padding:6px 14px; font-size:0.8rem; color:#a5d6a7; font-weight:500; margin:3px 3px 0 0; }

.info-box { background:rgba(15,35,18,0.5); border-left:3px solid #4caf50; border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; font-size:0.88rem; color:#a5d6a7; margin-top:1rem; line-height:1.6; }
.info-box strong { color:#c8e6c9; }

.sec-title { font-family:'Syne',sans-serif; font-weight:700; font-size:0.78rem; letter-spacing:3px; text-transform:uppercase; color:#66bb6a; margin:0 0 0.8rem; }

[data-testid="stExpander"] { background:rgba(15,35,18,0.4) !important; border:1px solid #1b3a1f !important; border-radius:12px !important; }

[data-testid="stMetric"] { background:rgba(15,35,18,0.6) !important; border:1px solid #2a4a2e !important; border-radius:12px !important; padding:0.8rem 1rem !important; }
[data-testid="stMetricLabel"] { color:#81c784 !important; font-size:0.75rem !important; }
[data-testid="stMetricValue"] { color:#c8e6c9 !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; }

[data-testid="stSidebar"] h2 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:#c8e6c9 !important; font-size:1rem !important; border-bottom:1px solid #2a4a2e; padding-bottom:0.5rem; }

.bm-footer { text-align:center; padding:2.5rem 1rem 1rem; color:#2e7d32; font-size:0.75rem; letter-spacing:1.5px; text-transform:uppercase; }

[data-testid="column"] { padding:0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    threshold = st.slider("Seuil de décision", 0.10, 0.90, 0.50, 0.05)
    st.caption("Si P(recyclable) ≥ seuil → classé recyclable.")
    show_details = st.toggle("Détails techniques", value=False)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#4a7a4e; line-height:1.8;'>
    <strong style='color:#81c784'>BioMaman v1.0</strong><br>
    Prototype IA · Classification déchets<br>
    Lomé, Togo · SAFEN 2026<br><br>
    Modèle : MobileNetV2<br>
    Entrée : 224 × 224 px<br>
    Classes : Recyclable / Non recyclable
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">♻ BioMaman · Tri Intelligent</div>
    <h1 class="hero-title">Ce déchet,<br>recyclable ou pas ?</h1>
    <p class="hero-sub">Prends une photo — l'IA répond en 2 secondes.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Chargement modèle
# ──────────────────────────────────────────────
MODEL_PATH = Path("best_model.h5")

@st.cache_resource
def load_my_model(path: str):
    return load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Modèle introuvable</strong><br>
    Place <code>best_model.h5</code> dans le même dossier que <code>app.py</code>,
    puis relance : <code>streamlit run app.py</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model = load_my_model(str(MODEL_PATH))

# ──────────────────────────────────────────────
# Prétraitement
# ──────────────────────────────────────────────
def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(uploaded_file):
    x = preprocess_image(uploaded_file)
    pred = model.predict(x, verbose=0)
    p = float(np.squeeze(pred))
    return max(0.0, min(1.0, p))

# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────
st.markdown('<span class="upload-label">📷 Charger une image du déchet</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown("""
    <div class="info-box" style="margin-top:1rem;">
    👆 <strong>Glisse une image ici</strong> ou clique pour parcourir.<br>
    Formats acceptés : JPG · JPEG · PNG
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="bm-footer">🌿 BioMaman · Lomé, Togo · SAFEN 2026</div>', unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────
# Analyse
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_img, col_res = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<p class="sec-title">Image analysée</p>', unsafe_allow_html=True)
    st.image(uploaded_file, use_container_width=True)

with col_res:
    st.markdown('<p class="sec-title">Résultat</p>', unsafe_allow_html=True)
    with st.spinner("Analyse IA en cours…"):
        time.sleep(0.4)
        try:
            p_rec = predict(uploaded_file)
        except Exception as e:
            st.error("Erreur pendant la prédiction.")
            if show_details:
                st.exception(e)
            st.stop()

    p_non = 1.0 - p_rec
    is_recyclable = p_rec >= threshold

    if is_recyclable:
        verdict_class = "verdict-recyclable"
        icon = "♻️"
        label_text = "Recyclable"
    else:
        verdict_class = "verdict-non"
        icon = "🗑️"
        label_text = "Non recyclable"

    bar_pct = p_rec * 100 if is_recyclable else p_non * 100
    fill_class = "bar-fill-green" if is_recyclable else "bar-fill-red"

    tips_html = ""
    if is_recyclable:
        tips = [("🔵","Plastique PET"),("⚪","Papier · Carton"),("🟤","Métal"),("🟢","Verre")]
        tips_html = '<p style="font-size:0.72rem;color:#81c784;letter-spacing:2px;text-transform:uppercase;font-weight:500;margin:0 0 8px;">Où déposer :</p><div style="display:flex;flex-wrap:wrap;gap:4px;">'
        for ico, lbl in tips:
            tips_html += f'<span class="cat-pill">{ico} {lbl}</span>'
        tips_html += "</div>"
        tips_html += """
        <div class="info-box" style="margin-top:1rem;">
        🌿 <strong>BioHub le plus proche</strong><br>
        Marché Assigamé · 0.4 km · Ouvert 7h–18h
        </div>"""
    else:
        tips_html = """
        <div class="info-box" style="background:rgba(100,0,0,0.2);border-left-color:#ef5350;">
        🌱 <strong>Déchet organique ?</strong><br>
        Ce type de déchet peut être transformé en <strong>biogaz ou compost</strong>
        au BioHub de votre quartier.
        </div>"""

    st.markdown(f"""
    <div class="result-card">
        <div class="{verdict_class}">
            <span class="verdict-icon">{icon}</span>
            <div>
                <p class="verdict-label">Classification IA</p>
                <p class="verdict-text">{label_text}</p>
            </div>
        </div>
        <div class="prob-row">
            <div class="prob-box">
                <p class="prob-val prob-val-green">{p_rec*100:.1f}%</p>
                <p class="prob-desc">♻ Recyclable</p>
            </div>
            <div class="prob-box">
                <p class="prob-val prob-val-red">{p_non*100:.1f}%</p>
                <p class="prob-desc">🗑 Non recyclable</p>
            </div>
        </div>
        <div class="bar-wrap">
            <div class="bar-label">
                <span>Confiance</span><span>{bar_pct:.0f}%</span>
            </div>
            <div class="bar-track">
                <div class="{fill_class}" style="width:{bar_pct:.1f}%"></div>
            </div>
        </div>
        {tips_html}
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Détails techniques
# ──────────────────────────────────────────────
if show_details:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Détails techniques"):
        c1, c2, c3 = st.columns(3)
        c1.metric("P(Recyclable)", f"{p_rec:.4f}")
        c2.metric("Seuil", f"{threshold:.2f}")
        c3.metric("Forme entrée", "1×224×224×3")
        st.caption(f"Décision : {'recyclable' if is_recyclable else 'non recyclable'} "
                   f"(P={p_rec:.4f} {'≥' if is_recyclable else '<'} seuil={threshold})")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="bm-footer">
    🌿 BioMaman · Tri Intelligent · Lomé, Togo · SAFEN 2026 &nbsp;·&nbsp; Prototype IA
</div>
""", unsafe_allow_html=True)
