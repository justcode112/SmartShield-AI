import streamlit as st
import joblib
import re
import string
from PIL import Image

st.set_page_config(
    page_title="SmartShield Pro",
    layout="wide"
)

# Dark mode styling
st.markdown("""
    <style>
    body {background-color: #0E1117; color: white;}
    </style>
""", unsafe_allow_html=True)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

st.title("🛡 SmartShield Pro")
st.subheader("Hybrid AI Spam Detection System")
st.write("Built by Numan")

tab1, tab2 = st.tabs(["🔍 Analyze Message", "📊 Model Insights"])

with tab1:
    user_input = st.text_area("Enter message:")

    if st.button("Analyze"):
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        probability = model.predict_proba(vectorized)[0][1]

        st.progress(float(probability))

        if probability > 0.40:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ NOT SPAM")

        st.write("Confidence:", round(probability,3))

with tab2:
    st.subheader("Model Performance Dashboard")

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    # Reload dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = vectorizer.transform(df['message'])
    y = df['label']
    y_pred = model.predict(X)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)

    fig1, ax1 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax1)
    st.pyplot(fig1)

    # Feature Importance
    st.subheader("Top Spam Indicators")

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_spam = np.argsort(coefficients)[-15:]

    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.barh(feature_names[top_spam], coefficients[top_spam])
    st.pyplot(fig2)