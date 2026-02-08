import streamlit as st
import pandas as pd
import joblib

# App title
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam Classifier")
st.write("Enter text to classify as spam or not spam.")

# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load("spam_classifier_model.pkl")

model = load_model()

# User inputs
text_input = st.text_area(
    "Enter text to classify",
    placeholder="Type your text here...",
    height=150
)   

# Predict button
if st.button("Classify Text"):
    if text_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([text_input])[0]
        proba = model.predict_proba([text_input])[0]
        confidence = max(proba)
        st.write(f"Confidence: {confidence:.2f}")
        if prediction == "spam":
            st.error("This message is SPAM")
        else:
            st.success("This message is NOT spam")