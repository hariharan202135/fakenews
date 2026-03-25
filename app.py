import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Prediction function
def predict_news(news):
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()
    
    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
    return label, confidence

# UI Design
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is real or fake.")

# Input
user_input = st.text_area("Enter News Text:")

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text")
    else:
        label, confidence = predict_news(user_input)

        st.subheader("Result:")
        
        if label == "REAL NEWS":
            st.success(label)
        else:
            st.error(label)

        st.write(f"Confidence: {round(confidence * 100, 2)}%")