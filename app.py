import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Citra Classifier", layout="wide")

# =====================================================
# SOFT PINK THEME (NON KONTRAS)
# =====================================================
st.markdown("""
<style>

/* ===============================
   BACKGROUND
================================ */
.stApp {
    background-color: #fff5f8;
}

/* ===============================
   HEADINGS & TEXT
================================ */
h1, h2, h3 {
    color: #8a1c4a;
    font-weight: 600;
}

label, p, span, div {
    color: #5a1f3c;
}

/* ===============================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background-color: #ffe6ef;
}

/* ===============================
   INPUT BOX
================================ */
input, textarea {
    background-color: #ffffff !important;
    color: #5a1f3c !important;
    border-radius: 8px;
    border: 1px solid #f3a5c4;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #f3a5c4;
}

div[data-baseweb="select"] * {
    color: #5a1f3c !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px dashed #f3a5c4;
    padding: 8px;
}

/* ===============================
   BUTTON
================================ */
.stButton>button {
    background-color: #f48fb1;
    color: #ffffff;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 0.5rem 1.2rem;
}

.stButton>button:hover {
    background-color: #ec6f9f;
}

/* ===============================
   SUCCESS BOX
================================ */
.stSuccess {
    background-color: #ffe6ef;
    color: #7a1c42;
    border-radius: 10px;
    border-left: 5px solid #f48fb1;
}

/* ===============================
   IMAGE
================================ */
img {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# UI
# =====================================================
st.title("🖼️ Citra Classifier – Emotion Detection")

model_choice = st.selectbox(
    "Pilih Model",
    ["base_cnn", "vgg19", "mobilenet"]
)

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "png", "jpeg"]
)

# =====================================================
# PREDICTION
# =====================================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((128,128))
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model(f"saved_models/{model_choice}.h5")
    pred = model.predict(img_array)

    class_labels = ["disgust", "happy", "sad"]
    idx = np.argmax(pred)

    st.success(
        f"Prediksi: {class_labels[idx]} "
        f"(Probabilitas: {pred[0][idx]:.2f})"
    )
