import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Document,
    get_response_synthesizer,
    StorageContext
)
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_parse import LlamaParse
import qdrant_client
from qdrant_client.http.models import VectorParams, Distance
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import tempfile
import os
from PIL import Image
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import nest_asyncio
from dotenv import load_dotenv
import uuid
import numpy as np
import shutil

nest_asyncio.apply()

load_dotenv()

llm = OpenAIMultiModal(
    model="gpt-4o",
    temperature=0.2
)

embed_model = OpenAIEmbedding()
image_embedding_model = SentenceTransformer("clip-ViT-B-32")

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="text",
    verbose=True,
)
file_extractor = {".pdf": parser}

client = qdrant_client.QdrantClient(location=":memory:")

# Check if the collection exists and create if not
collection_name = "test_store"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)

# Set the header of the Streamlit application
st.header("Document Chatbot")

# Initialize session state to store the chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your documents!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(uploaded_files, persistent_dir):
    """
    Load and index PDF documents uploaded by the user.

    Args:
        uploaded_files: A list of uploaded file objects.
        persistent_dir: A directory to save persistent data.

    Returns:
        VectorStoreIndex: An indexed representation of the documents.
    """
    with st.spinner(text="Indexing uploaded documents – hang tight! This might take some time."):
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                file_path = os.path.join(persistent_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

        reader = SimpleDirectoryReader(input_dir=persistent_dir, file_extractor=file_extractor, recursive=True)
        docs = reader.load_data()

        if docs:
            settings_for_indexing = {
                "embedding_model": embed_model
            }
            # Execute pipeline and time the process
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

            return index
        else:
            return None

def extract_images_and_tables(pdf_path, persistent_dir):
    """
    Extract images and tables from a PDF file.

    Args:
        pdf_path: The path to the PDF file.
        persistent_dir: The directory to save extracted images and tables.

    Returns:
        A list of paths to the extracted images and tables.
    """
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_path = os.path.join(persistent_dir, f"page_{page_number + 1}_img_{img_index + 1}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            image_paths.append(image_path)
    
    return image_paths

def display_image(image_path):
    """
    Display an image in the Streamlit app.

    Args:
        image_path: Path to the image to be displayed.
    """
    image = Image.open(image_path)
    st.image(image, caption=image_path)

# Streamlit file uploader
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    persistent_dir = "persistent_data"
    os.makedirs(persistent_dir, exist_ok=True)

    index = load_data(uploaded_files, persistent_dir)

    if index:
        # Process images from the uploaded PDFs
        image_paths = []
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(persistent_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.extend(extract_images_and_tables(pdf_path, persistent_dir))

        # Generate embeddings for images and store them in Qdrant
        for image_path in image_paths:
            image = Image.open(image_path)
            image_embedding = image_embedding_model.encode(image)
            # Resize the image embedding to match the document embedding size
            if image_embedding.shape[0] != 1536:
                image_embedding = np.resize(image_embedding, (1536,))
            image_embedding = image_embedding.tolist()
            point = qdrant_client.http.models.PointStruct(
                id=str(uuid.uuid4()),  # Generate a valid UUID for each point
                vector=image_embedding,
                payload={"type": "image", "path": image_path}
            )
            client.upsert(collection_name=collection_name, points=[point])

        # Set up the Settings with the LLM for the querying stage
        settings_for_querying = {
            "llm": llm,
            "embedding_model": embed_model
        }

        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
        )

        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
        )

        # Assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[cohere_rerank]
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True
        )

        tool_config = IndexToolConfig(
            query_engine=query_engine,
            name=f"Vector Index",
            description=f"useful for when you want to answer queries about the uploaded documents",
            tool_kwargs={"return_direct": True},
            memory=memory
        )

        # Create the tool
        tool = LlamaIndexTool.from_tool_config(tool_config)

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
                    response = tool.run(prompt)
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)  # Add response to message history

                    # Check if the response contains a request for an image
                    if "show me the image" in prompt.lower():
                        # Assuming the images are indexed and stored with a path
                        for point in client.search(
                                collection_name="test_store",
                                query_vector=image_embedding,
                                limit=1
                            ):
                            image_path = point.payload["path"]
                            display_image(image_path)

