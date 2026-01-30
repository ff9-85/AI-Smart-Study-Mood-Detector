import streamlit as st
import pickle
import re

#Load model and vectorizer
model=pickle.load(open("mood_model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

#Suggesions
suggestions={
    "Focused":"Great! Keep going 🚀",
    "Tired":"Take a short break and hydrate 💧",
    "Stressed":"Try deep breathing or plan tasks 🧘",
    "Distracted":"Remove distractions and set a timer ⏱"
}

#Preprocess function
def preprocess(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    return text

#Streamlit UI
st.set_page_config(page_title="Smart Study Mood Detector",page_icon="📚",layout="centered")
st.title("📚 Smart Study Mood Detector")
st.markdown("Type how you feel while studying and get your study mood prediction!")

#user input
user_input=st.text_area("Enter your study feeling here:")
if st.button("Predict Mood"):
    if user_input.strip()=="":
        st.warning("Please enter a sentence!")
    else:
        processed_input=preprocess(user_input)
        vectorized_input=vectorizer.transform([processed_input])
        mood=model.predict(vectorized_input)[0]
        st.success(f"🧠 Detected Mood: **{mood}**")
        st.info(f"💡Suggestion:{suggestions[mood]}")