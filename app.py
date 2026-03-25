import streamlit as st
import pickle
import os

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is real or fake.")

# 🔍 Check if model files exist
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.warning("⚠ Model files not found in deployment.")
    st.info("This is a demo version. Full model works locally.")
    st.stop()

# ✅ Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔮 Prediction function
def predict_news(news):
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()
    
    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
    return label, confidence

# 📝 Input box
user_input = st.text_area("Enter News Text:")

# 🔘 Predict button
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
