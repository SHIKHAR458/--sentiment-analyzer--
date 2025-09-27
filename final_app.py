
import streamlit as st
import joblib


model = joblib.load("logistic_sentiment.pkl")
vectorizer = joblib.load("tfidf.pkl")


emotion_mapping = {0: 'sadness', 1: 'anger', 2: 'joy'}

st.set_page_config(page_title="Emotion Prediction App", page_icon="🎭", layout="centered")

st.markdown("""
<style>
body {
    background-color: #f0f8ff;  /* light blue */
    color: #000000;
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    height: 50px;
    width: 200px;
    border-radius: 10px;
    border: none;
    font-size: 16px;
}
.stTextArea textarea {
    background-color: #e6f2ff;
    color: #000000;
    font-size: 16px;
    height: 150px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Emotion Prediction App")
st.info("Note: This is an ML-based NLP model that detects only sadness, anger, and joy. Predictions may not always be perfect, especially for short or ambiguous texts.")



user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if len(user_input.split()) < 15:
        st.warning("⚠ Please enter at least 15 words for a more accurate prediction.")
    else:
        X_input = vectorizer.transform([user_input])
        pred_index = model.predict(X_input)[0]
        pred_label = emotion_mapping[pred_index]

        color_dict = {'sadness':'#6495ED', 'anger':'#FF6347', 'joy':'#FFD700'}
        color = color_dict.get(pred_label, '#FFFFFF')

        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 15px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #000000;
        ">
            Predicted Emotion: {pred_label}
        </div>
        """, unsafe_allow_html=True)
