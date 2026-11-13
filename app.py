import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import urllib.request
import os

# ── Configuration de la page
st.set_page_config(
    page_title="Mask Detection App",
    page_icon="😷",
    layout="wide"
)

# ── Style CSS
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; color: white; }
    h1, h2, h3, p, label { color: white !important; }
    .result-box {
        padding: 15px; border-radius: 10px;
        text-align: center; font-size: 20px;
        font-weight: bold; margin: 8px 0;
    }
    .with-mask    { background-color: #1b5e20; color: white; }
    .without-mask { background-color: #b71c1c; color: white; }
    .stat-box {
        background-color: #16213e; padding: 15px;
        border-radius: 8px; text-align: center; margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ── Téléchargement du modèle DNN
@st.cache_resource
def load_face_detector():
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url    = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    if not os.path.exists("deploy.prototxt"):
        urllib.request.urlretrieve(prototxt_url, "deploy.prototxt")
    if not os.path.exists("face_detector.caffemodel"):
        urllib.request.urlretrieve(model_url, "face_detector.caffemodel")
    return cv2.dnn.readNet("deploy.prototxt", "face_detector.caffemodel")

# ── Chargement du modèle CNN masque
@st.cache_resource
def load_mask_model():
    return load_model("best_model.keras")

face_net = load_face_detector()
model    = load_mask_model()

# ── Prédiction masque
def predict_face(face_img):
    img = cv2.resize(face_img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prob = model.predict(img, verbose=0)[0][0]
    if prob >= 0.5:
        return "Masque OK", (0, 200, 0), prob * 100
    else:
        return "Sans masque !", (0, 0, 220), (1 - prob) * 100

# ── Détection DNN + NMS + classification
def detect_masks(image_np):
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Agrandir si trop petite
    if w < 1200:
        scale = 1200 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Collecter toutes les détections
    boxes = []
    confidences = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2-x1, y2-y1])
            confidences.append(float(conf))

    # NMS pour supprimer les doublons
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    results = []
    faces_found = 0

    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, bw, bh = boxes[idx]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x+bw), min(h, y+bh)

            face_roi = img_bgr[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            faces_found += 1
            label, color, conf = predict_face(face_roi)
            results.append({"label": label, "conf": conf, "color": color})

            face_w = x2 - x1
            font_scale = max(0.4, min(0.8, face_w / 150))
            thickness  = 2

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
            band_h = max(30, int(face_w * 0.2))
            y_band = max(band_h, y1)
            cv2.rectangle(img_bgr, (x1, y_band - band_h),
                          (x2, y_band), color, -1)

            short_label = "Masque" if "Masque OK" in label else "No Mask"
            text = f"{short_label} {conf:.0f}%"
            cv2.putText(img_bgr, text, (x1 + 5, y_band - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness)

    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_result, results, faces_found

# ════════════════════════════════════════
# ── SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Choose Input Mode")
    st.markdown("---")
    mode = st.radio(
        "Select input mode:",
        ["📁 Image Upload", "ℹ️ About the Project"],
        label_visibility="collapsed"
    )

# ════════════════════════════════════════
# ── PAGE : IMAGE UPLOAD
# ════════════════════════════════════════
if mode == "📁 Image Upload":
    st.markdown("# 😷 Mask Detection App")
    st.markdown("Uploadez une image pour détecter automatiquement les masques.")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Choisissez une image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
        image_np  = np.array(pil_image)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🖼️ Image originale")
            st.image(pil_image, use_container_width=True)

        with st.spinner("🔍 Analyse en cours..."):
            result_img, results, nb_faces = detect_masks(image_np)

        with col2:
            st.markdown("### 🎯 Résultat de détection")
            st.image(result_img, use_container_width=True)

        st.markdown("---")

        if nb_faces == 0:
            st.warning("⚠️ Aucun visage détecté. Essayez une image avec des visages bien visibles et de face.")
        else:
            nb_mask   = sum(1 for r in results if "Masque OK" in r["label"])
            nb_nomask = sum(1 for r in results if "Sans masque" in r["label"])

            st.markdown(f"### 📊 Résultats — {nb_faces} visage(s) détecté(s)")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="stat-box">
                    <h2 style="color:#ffffff">{nb_faces}</h2>
                    <p>👤 Visages détectés</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stat-box">
                    <h2 style="color:#4CAF50">{nb_mask}</h2>
                    <p>✅ Avec masque</p>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="stat-box">
                    <h2 style="color:#ef5350">{nb_nomask}</h2>
                    <p>❌ Sans masque</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("### 🔍 Détail par visage")
            for i, r in enumerate(results):
                css  = "with-mask" if "Masque OK" in r["label"] else "without-mask"
                icon = "✅" if "Masque OK" in r["label"] else "❌"
                st.markdown(f"""<div class="result-box {css}">
                    {icon} Visage {i+1} — {r["label"]} — Confiance : {r["conf"]:.1f}%
                </div>""", unsafe_allow_html=True)

    else:
        st.info("👆 Uploadez une image JPG, PNG ou JPEG pour commencer.")
        st.markdown("""
        #### ✅ Ce que le système peut faire :
        - Détecter plusieurs visages dans une même image
        - Classifier chaque visage : avec masque ou sans masque
        - Afficher le pourcentage de confiance pour chaque détection
        - Compter le nombre de personnes avec et sans masque

        #### 📌 Conseils pour de meilleurs résultats :
        - Utilisez des images avec visages bien visibles et de face
        - Privilégiez des images de bonne qualité (+500KB)
        - Évitez les images avec visages très petits ou de profil
        """)

# ════════════════════════════════════════
# ── PAGE : ABOUT
# ════════════════════════════════════════
elif mode == "ℹ️ About the Project":
    st.markdown("# ℹ️ About the Mask Detection Project")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Contexte du Projet
        Projet réalisé dans le cadre du module **Deep Learning**.
        Système de détection automatique du port du masque
        basé sur les réseaux de neurones convolutifs (CNN).

        ### 🧠 Modèle CNN
        - **Architecture** : CNN from scratch (3 blocs convolutifs)
        - **Dataset** : 7 553 images (Kaggle)
        - **Précision** : 97,88% sur les données de test
        - **Framework** : TensorFlow / Keras
        - **Détection visages** : OpenCV DNN (SSD ResNet)
        """)

    with col2:
        st.markdown("""
        ### 🛠️ Technologies Utilisées
        | Bibliothèque | Rôle |
        |---|---|
        | TensorFlow / Keras | Modèle CNN |
        | OpenCV DNN | Détection des visages |
        | NumPy | Traitement des données |
        | Streamlit | Interface web |
        | Matplotlib | Visualisation |
        | Scikit-learn | Métriques |

        ### 📊 Résultats
        - **Accuracy** : 97,88%
        - **F1-Score** : 0.97
        - **Erreurs** : 29 / 1133 images de test
        """)

    st.markdown("---")
    st.markdown("""
    ### 👨‍💻 Réalisé par
    **Votre Nom** — 5ème année Ingénierie Informatique
    Sous la supervision de **Prof. BENLAKHDAR Said**
    """)