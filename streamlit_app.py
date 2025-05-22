import streamlit as st
import os
import requests
from PIL import Image
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Load resources ---
st.set_page_config(page_title="CBT_Based_Chatbot", layout="centered")
#Load and show logo
logo = Image.open("logo.png")
st.image(logo, width=120)

# Title and intro
#st.markdown("<h1 style='text-align: center; color: #20B2AA;h1>", unsafe_allow_html=True)Dr.Brainee</h1>", unsafe_allow_html=True)



@st.cache_resource
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3-3-70b-versatile"
    )

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return Chroma(persist_directory="/content/chroma_db", embedding_function=embeddings)

# --- Preprocessing function ---

# --- Set up LangChain QA pipeline ---
def setup_qa_chain(vector_db, llm):
    prompt_templates = """ You are a compassionate mental health chatbot.
First, give the sentiment of the user query, then respond thoughtfully.

Context:
{context}

User: {question}
Chatbot:
"""
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Main Streamlit App ---

def main():
    st.markdown(
        """
        <h1 style='text-align: center; color: #20B2AA;'>Dr.Brainee!!</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='
            background-color: #2F4F4F;
            padding: 20px;
            border-radius: 10px;
            color: #BDB76B;
            font-size: 18px;
            '>
            <b>Hi, I’m Dr.Brainee</b> — your personal <b>mental health therapy assistant</b>.<br>
            I’m here to <i>listen, support you</i>, and help you through anything on your mind.<br>
            This is a safe, <span style="color: yellow;"><b>judgment‑free space</b></span>. Let’s talk. ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

    user_input = st.text_area("🧠 How are you feeling today?", height=150)


    if st.button("Submit") and user_input.strip():
        with st.spinner("Analyzing..."):
            llm = initialize_llm()
            vector_db = load_vector_db()
            qa_chain = setup_qa_chain(vector_db, llm)
            response = qa_chain.run(user_input)
            
        st.markdown("**Chatbot Response:**")
        st.write(response)

    st.markdown("---")
    st.markdown("🔚 Type 'exit' to end the session.")

if __name__ == "__main__":
    main()

   
    
