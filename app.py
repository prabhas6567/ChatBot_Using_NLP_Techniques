import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL and NLTK setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Custom CSS for styling
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        color: #4F8BF9;
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .chat-user {
        background-color: #4F8BF9;
        color: white;
        padding: 10px;
        border-radius: 10px 10px 0 10px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
    }
    .chat-bot {
        background-color: #F0F2F6;
        color: black;
        padding: 10px;
        border-radius: 10px 10px 10px 0;
        margin: 5px 0;
        max-width: 70%;
        float: left;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🤖 TalkAI - The General AI Chatbot Assistant")
    st.markdown("Welcome to the chatbot! Type a message below to start chatting.")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize a counter for the input widget key
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    # Home Menu
    if choice == "Home":
        # Chat input with a unique key
        user_input = st.text_input(
            "You:",
            key=f"user_input_{st.session_state.input_counter}",
            placeholder="Type your message here..."
        )

        if user_input:
            # Get chatbot response
            response = chatbot(user_input)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Append to chat history
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": response,
                "timestamp": timestamp
            })

            # Display chat history
            st.markdown("---")
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="chat-user">You: {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bot">Bot: {chat["bot"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-timestamp">{chat["timestamp"]}</div>', unsafe_allow_html =True)
            st.markdown("---")

            # Increment the input counter to create a new key for the next input
            st.session_state.input_counter += 1

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="chat-user">You: {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bot">Bot: {chat["bot"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-timestamp">{chat["timestamp"]}</div>', unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write("This project aims to create an interactive chatbot that understands and responds to user input based on intents. Built using NLP techniques and Streamlit, it provides a user-friendly interface for engaging conversations.")

        st.subheader("Project Overview:")
        st.write("""
        The project consists of two main components:
        1. **NLP Techniques**: Utilizing Logistic Regression to train the chatbot on labeled intents.
        2. **Streamlit Interface**: A web-based interface for users to interact with the chatbot.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset includes labeled intents and entities, allowing the chatbot to understand user queries effectively.
        - **Intents**: Categories of user input (e.g., greetings, inquiries).
        - **Entities**: Specific information extracted from user input.
        """)

        st.subheader("Conclusion:")
        st.write("This chatbot project demonstrates the integration of NLP and web technologies to create an engaging user experience. Future enhancements could include more advanced NLP techniques and a larger dataset for improved accuracy.")

if __name__ == '__main__':
    main()