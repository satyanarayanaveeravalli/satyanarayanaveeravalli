import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Google Play App Predictor", page_icon="📱", layout="centered")

# Load model and PCA
model = pickle.load(open('model.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

# Background and font styling
st.markdown("""
    <style>
    /* Background image */
    .stApp {
        background-color:yellow;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Transparent content block */
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: #003366;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        color: #444;
    }

    /* Style input labels */
    label {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="block-container">', unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📱 Google Play App Success Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fill in the details to estimate how successful your app will be</div>', unsafe_allow_html=True)

# Input form in columns
col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("📂 App Category", ['FAMILY', 'GAME', 'TOOLS', 'EDUCATION', 'PRODUCTIVITY'])
    size = st.number_input("📦 App Size (MB)", min_value=1.0)
    installs = st.number_input("⬇️ Number of Installs", min_value=0)
    paid_status = st.selectbox("💲 Paid or Free?", ['Free', 'Paid'])

with col2:
    price = st.number_input("💰 Price (₹)", min_value=0.0)
    rating = st.number_input("⭐ App Rating (0–5)", min_value=0.0, max_value=5.0)
    year = st.number_input("📅 Release Year", min_value=2000, max_value=2030)
    month = st.number_input("📅 Release Month", min_value=1, max_value=12)
    day = st.number_input("📅 Release Day", min_value=1, max_value=31)

# Category encoding
category_map = {'FAMILY': 0, 'GAME': 1, 'TOOLS': 2, 'EDUCATION': 3, 'PRODUCTIVITY': 4}
category_encoded = category_map[category]
paid = 0 if paid_status == 'Free' else 1

# Prepare input
input_data = np.array([[category_encoded, size, installs, paid, price, rating, year, month, day]])
input_pca = pca.transform(input_data)

# Prediction
if st.button("🚀 Predict Success"):
    prediction = model.predict(input_pca)
    st.success(f"📈 Predicted Success Rating / Category: **{prediction[0]}**")

st.markdown('</div>', unsafe_allow_html=True)
