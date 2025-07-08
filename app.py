import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.title("📩 Spam Email Classifier")

user_input = st.text_area("Enter the email message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 🔥 Vectorize the input before predicting
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        result = "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam"
        st.success(f"The Message is: {result}")


