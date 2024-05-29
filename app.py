import streamlit as st
import logging
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
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
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.youtube_transcript.utils import is_youtube_video
from llama_parse import LlamaParse
import qdrant_client
from qdrant_client.http.models import VectorParams, Distance
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import os
from PIL import Image
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import nest_asyncio
from dotenv import load_dotenv
import uuid
import numpy as np
from io import BytesIO
import tempfile

# TODO: Make RAG Agentic
# TODO: Fix image retrieval
# TODO: Integrate with Coaching AI Code
# TODO: Fix memory issue
# TODO: Modularize the code 

# Setup logging
#logging.basicConfig(level=logging.DEBUG)

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

# Initialize session state to store the chat history and the index
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your documents!"}
    ]

if "index" not in st.session_state:
    st.session_state.index = None

# Initialize an in-memory store for images
if "image_store" not in st.session_state:
    st.session_state.image_store = {}

@st.cache_resource(show_spinner=False)
def load_data(uploaded_files, youtube_links):
    """
    Load and index PDF documents and YouTube video transcripts uploaded by the user.

    Args:
        uploaded_files: A list of uploaded file objects.
        youtube_links: A list of YouTube video links.

    Returns:
        VectorStoreIndex: An indexed representation of the documents and transcripts.
    """
    documents = []
    
    with st.spinner(text="Indexing uploaded documents – hang tight! This might take some time."):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process uploaded PDF files
            if uploaded_files:
                logging.info("Processing uploaded PDF files.")
                for uploaded_file in uploaded_files:
                    if uploaded_file is not None:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                
                reader = SimpleDirectoryReader(input_dir=temp_dir, file_extractor=file_extractor, recursive=True)
                documents.extend(reader.load_data())
    
    # Process YouTube links
    if youtube_links:
        logging.info("Processing YouTube links.")
        youtube_reader = YoutubeTranscriptReader()
        for link in youtube_links:
            if is_youtube_video(link):
                documents.extend(youtube_reader.load_data(ytlinks=[link]))

    if documents:
        logging.info("Creating index from documents.")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index
    else:
        logging.warning("No documents found to index.")
        return None

def extract_images_and_tables(pdf_path):
    """
    Extract images and tables from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A list of keys to the in-memory images.
    """
    doc = fitz.open(pdf_path)
    image_keys = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_key = f"page_{page_number + 1}_img_{img_index + 1}"
            st.session_state.image_store[image_key] = image_bytes
            logging.info(f"Extracted image: {image_key}")

            image_keys.append(image_key)
    
    return image_keys

def display_image(image_key):
    """
    Display an image from the in-memory store in the Streamlit app.

    Args:
        image_key: Key of the image to be displayed.
    """
    logging.info(f"Displaying image: {image_key}")
    image_bytes = st.session_state.image_store[image_key]
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption=image_key)

# Sidebar for file upload and YouTube links
with st.sidebar:
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf'])
    youtube_links = st.text_area("Enter YouTube links (one per line)")

    if st.button("Process"):
        youtube_links = youtube_links.split("\n") if youtube_links else []
        st.session_state.index = load_data(uploaded_files, youtube_links)

# Set up the Settings with the LLM for the querying stage
if st.session_state.index:
    # Process images from the uploaded PDFs
    image_keys = []
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_keys.extend(extract_images_and_tables(pdf_path))

    # Generate embeddings for images and store them in Qdrant
    for image_key in image_keys:
        image_bytes = st.session_state.image_store[image_key]
        image = Image.open(BytesIO(image_bytes))
        image_embedding = image_embedding_model.encode(image)
        # Resize the image embedding to match the document embedding size
        if image_embedding.shape[0] != 1536:
            image_embedding = np.resize(image_embedding, (1536,))
        image_embedding = image_embedding.tolist()
        point = qdrant_client.http.models.PointStruct(
            id=str(uuid.uuid4()),  # Generate a valid UUID for each point
            vector=image_embedding,
            payload={"type": "image", "key": image_key}
        )
        client.upsert(collection_name=collection_name, points=[point])
        logging.info(f"Stored image embedding for: {image_key}")

    settings_for_querying = {
        "llm": llm,
        "embedding_model": embed_model
    }

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=st.session_state.index,
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
                try:
                    # Retrieve the response from the chat engine based on the user's prompt
                    response = tool.run(prompt)
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)  # Add response to message history

                    # Check if the response contains a request for an image
                    if "show me the image" in prompt.lower():
                        logging.info("Request to show image detected.")
                        # Assuming the images are indexed and stored with a key
                        # Perform a search query with a dummy vector, here we use a zero vector
                        dummy_vector = [0.0] * 1536
                        results = client.search(
                            collection_name="test_store",
                            query_vector=dummy_vector,
                            limit=1
                        )
                        for point in results:
                            image_key = point.payload["key"]
                            display_image(image_key)
                except Exception as e:
                    logging.error(f"Error during tool run: {e}")
                    st.write("An error occurred while processing your request. Please try again.")

# Check if there are no uploaded files or YouTube links
if st.session_state.index is None and len(st.session_state.messages) > 1:
    st.write("Please upload documents and/or provide YouTube links to proceed.")
