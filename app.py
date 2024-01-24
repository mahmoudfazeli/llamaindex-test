"""
Workshop Transcript Chatbot Application

This Streamlit application uses LlamaIndex to create a chatbot that processes and answers questions based on workshop transcripts.
It loads and indexes transcripts from a specified directory and utilizes LlamaIndex's chat engine to augment GPT-3.5 responses with the context from these transcripts.
"""

import streamlit as st
from llama_index import(
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Document
)
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from dotenv import load_dotenv
import tempfile
import os
#import openai

load_dotenv()

llm=OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    system_prompt=
        "You are a workshop facilitator with access to a detailed document.\n"
        "Your job is to answer questions about the workshop topic by referring to specific parts of the document.\n"
        "Please cite the page number or section when providing answers.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    )
embed_model = OpenAIEmbedding()


# Set the header of the Streamlit application
st.header("Workshop Transcript Chatbot")

# Initialize session state to store the chat history
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about DPS Workshops!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(uploaded_files):
    """
    Load and index workshop transcripts uploaded by the user.

    Args:
        uploaded_files: A list of uploaded file objects.

    Returns:
        VectorStoreIndex: An indexed representation of the workshop transcripts.
    """
    with st.spinner(text="Indexing uploaded workshop docs – hang tight! This might take some time."):
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

            reader = SimpleDirectoryReader(input_dir=temp_dir, recursive=True)
            docs = reader.load_data()

            if docs:
                service_context_for_indexing = ServiceContext.from_defaults(embed_model = embed_model)
                index = VectorStoreIndex.from_documents(docs, service_context=service_context_for_indexing)
                return index
            else:
                return None

# Streamlit file uploader
uploaded_files = st.file_uploader("Upload workshop documents", accept_multiple_files=True)

if uploaded_files:
    index = load_data(uploaded_files)
    if index:
        # Set up the ServiceContext with the LLM for the querying stage
        service_context_for_querying = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model
        )
        # Create a chat engine using the indexed data
        chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            service_context=service_context_for_querying,
            verbose=True
        )

        # Chat interface for user input and displaying chat history
        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate and display the response from the chat engine
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve the response from the chat engine based on the user's prompt
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history