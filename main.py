import os
import tempfile
import nltk

# Set up a custom NLTK data path
nltk_data_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)
    
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

import streamlit as st
import logging
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
    StorageContext
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.youtube_transcript.utils import is_youtube_video
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from llama_parse import LlamaParse
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import numpy as np
import uuid
import tempfile
from dotenv import load_dotenv
# import os
import nest_asyncio
import base64

from llama_index.indices.managed.vectara import VectaraIndex
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

import requests
import json

# Setup logging
#logging.basicConfig(level=logging.DEBUG)

# Apply nest_asyncio
nest_asyncio.apply()
load_dotenv()

# Define the DeepInfraEmbedder Class
class DeepInfraEmbedder:
    def __init__(self, api_key, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.api_key = api_key
        self.url = 'https://api.deepinfra.com/v1/openai/embeddings'
        self.model_name = model_name

    def embed(self, text):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'input': text,
            'model': self.model_name,
            'encoding_format': 'float'
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f'Error: {response.status_code} - {response.text}')

    def embed_documents(self, texts):
        return [self.embed(text) for text in texts]

class DeepInfraImageEmbedder:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = 'https://api.deepinfra.com/v1/inference/sentence-transformers/clip-ViT-B-32'

    def embed_image(self, image_base64):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'inputs': [f"data:image/png;base64,{image_base64}"],
            'normalize': False
        }
        logging.info("Sending request to DeepInfra API")
        try:
            response = requests.post(self.url, headers=headers, json=data)
            logging.info(f"Received response status code: {response.status_code}")

            if response.status_code == 200:
                response_json = response.json()
                logging.info(f"Response JSON: {response_json}")
                embeddings = response_json.get('embeddings', [])
                if not embeddings:
                    raise ValueError("Embeddings not found in response")
                logging.info(f"Received embeddings: {embeddings}")
                return embeddings[0]
            else:
                logging.error(f"DeepInfra API Error: {response.status_code} - {response.text}")
                raise Exception(f'Error: {response.status_code} - {response.text}')
        except Exception as e:
            logging.error(f"Failed to get embeddings from DeepInfra API: {e}")
            raise

# Define a Callable Wrapper
class CallableDeepInfraEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, text):
        return self.embedder.embed(text)

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

# Sidebar for app selection
st.sidebar.title("Select App")
app_selection = st.sidebar.selectbox("Choose an app", ["Document Chatbot", "Coaching AI"])

def document_chatbot():
    st.header("Document Chatbot")

    # Initialize models and clients
    llm = OpenAIMultiModal(model="gpt-4o", temperature=0.2)
    embed_model = OpenAIEmbedding()

    # Initialize DeepInfraImageEmbedder for image embeddings
    deepinfra_api_key = os.getenv("DEEPINFRA_TOKEN")
    image_embedding_model = DeepInfraImageEmbedder(api_key=deepinfra_api_key)

    # Initialize the parser for PDF files
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="text", verbose=True)
    file_extractor = {".pdf": parser}

    # Initialize Qdrant client and collection
    client = qdrant_client.QdrantClient(location=":memory:")
    collection_name = "test_store"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize Cohere Rerank
    cohere_api_key = os.getenv("COHERE_API_KEY")
    cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)

    # Initialize session state for chat history and index
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your documents!"}
        ]
    if "index" not in st.session_state:
        st.session_state.index = None
    if "image_store" not in st.session_state:
        st.session_state.image_store = {}
    if "image_keys" not in st.session_state:
        st.session_state.image_keys = []

    @st.cache_resource(show_spinner=False)
    def load_data(uploaded_files, youtube_links):
        documents = []
        with st.spinner(text="Indexing uploaded documents – hang tight! This might take some time."):
            with tempfile.TemporaryDirectory() as temp_dir:
                if uploaded_files:
                    logging.info("Processing uploaded PDF files.")
                    for uploaded_file in uploaded_files:
                        if uploaded_file is not None:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                    reader = SimpleDirectoryReader(input_dir=temp_dir, file_extractor=file_extractor, recursive=True)
                    documents.extend(reader.load_data())
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

    def extract_images_and_store(pdf_path):
        doc = fitz.open(pdf_path)
        image_keys = []

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                image_key = f"page_{page_number + 1}_img_{img_index + 1}"
                st.session_state.image_store[image_key] = {
                    "base64": image_base64,
                    "metadata": {
                        "page": page_number + 1,
                        "index": img_index + 1,
                        "description": f"Image on page {page_number + 1}, index {img_index + 1}"
                    }
                }
                logging.info(f"Stored image: {image_key}")

                # Use DeepInfra to get the image embedding
                try:
                    logging.info(f"Requesting embedding for image: {image_key}")
                    embedding = image_embedding_model.embed_image(image_base64)
                    logging.info(f"Received embedding for image: {image_key}")
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "type": "image",
                            "key": image_key,
                            "metadata": st.session_state.image_store[image_key]["metadata"]
                        }
                    )
                    client.upsert(collection_name=collection_name, points=[point])
                    logging.info(f"Stored embedding for image: {image_key}")
                except Exception as e:
                    logging.error(f"Failed to store embedding for image {image_key}: {e}")
                image_keys.append(image_key)
                st.session_state.image_keys.append(image_key)
        
        return image_keys

    def display_image(image_key):
        logging.info(f"Displaying image: {image_key}")
        image_base64 = st.session_state.image_store[image_key]["base64"]
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        st.image(image, caption=f"Image from {st.session_state.image_store[image_key]['metadata']['description']}")

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf'])
        youtube_links = st.text_area("Enter YouTube links (one per line)")

        if st.button("Process"):
            youtube_links = youtube_links.split("\n") if youtube_links else []
            st.session_state.index = load_data(uploaded_files, youtube_links)

    if st.session_state.index:
        for uploaded_file in uploaded_files:
            logging.info(f"Processing uploaded file: {uploaded_file.name}")
            pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            extract_images_and_store(pdf_path)

        settings_for_querying = {
            "llm": llm,
            "embedding_model": embed_model
        }

        retriever = VectorIndexRetriever(
            index=st.session_state.index,
            similarity_top_k=5,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[cohere_rerank]
        )

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["content"].startswith("![Image]"):
                    image_key = message["content"].strip("![Image](").strip(")")
                    display_image(image_key)
                else:
                    st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = query_engine.query(prompt)
                        st.write(response.response)
                        message = {"role": "assistant", "content": response.response}
                        st.session_state.messages.append(message)

                        if "show me the image" in prompt.lower():
                            logging.info("Request to show image detected.")
                            dummy_vector = [0.0] * 1536
                            results = client.search(
                                collection_name="test_store",
                                query_vector=dummy_vector,
                                limit=1
                            )
                            for point in results:
                                image_key = point.payload["key"]
                                display_image(image_key)
                                if f"![Image]({image_key})" not in [msg["content"] for msg in st.session_state.messages]:
                                    st.session_state.messages.append({"role": "assistant", "content": f"![Image]({image_key})"})

                    except Exception as e:
                        logging.error(f"Error during tool run: {e}")
                        st.write("An error occurred while processing your request. Please try again.")

    if st.session_state.index is None and len(st.session_state.messages) > 1:
        st.write("Please upload documents and/or provide YouTube links to proceed.")

def coaching_ai():
    st.title("Coaching AI Chat App")

    ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    VECTARA_CUSTOMER_ID = os.getenv('VECTARA_CUSTOMER_ID')
    VECTARA_API_KEY = os.getenv('VECTARA_API_KEY')

    URLs = ["https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"]

    def display_message(message, is_user=False):
        st.markdown(f"{'User:' if is_user else 'Assistant:'} {message}")

    chat_history = st.container()
    query_input = st.text_input("Enter your query", key="query_input")
    submit_button = st.button("Submit")

    user_query = ""

    if submit_button:
        user_query = query_input

        with st.spinner("Retrieving the answer..."):
            vectara = VectaraIndex(vectara_customer_id=VECTARA_CUSTOMER_ID, vectara_corpus_id="3", vectara_api_key=VECTARA_API_KEY)
            qe = vectara.as_query_engine(similarity_top_k=10)
            response_vectara = qe.query(user_query)

            docs = [AssemblyAIAudioTranscriptLoader(file_path=url, api_key=ASSEMBLYAI_API_KEY).load()[0] for url in URLs]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(docs)
            for text in texts:
                text.metadata = {"audio_url": text.metadata["audio_url"]}

            # Function to create the embedder instance
            def make_embedder():
                api_key = os.getenv('DEEPINFRA_TOKEN')
                embedder = DeepInfraEmbedder(api_key)
                return CallableDeepInfraEmbedder(embedder)
                
            embedder = make_embedder()

            qdrant_client = QdrantClient()  
            qdrant = Qdrant.from_documents(texts, embedder, location=":memory:", collection_name="my_documents")

            def make_qa_chain():
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
                return RetrievalQA.from_chain_type(
                    llm,
                    retriever=qdrant.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}),
                    return_source_documents=True
                )

            qa_chain = make_qa_chain()
            response_langchain = qa_chain({"query": user_query})

        with chat_history:
            display_message(user_query, is_user=True)
            display_message(f"Response from Vectara\n{response_vectara}")
            display_message(f"Response from LangChain\nQ: {response_langchain['query'].strip()}\nA: {response_langchain['result'].strip()}")
            st.write("SOURCES:")
            for idx, elt in enumerate(response_langchain['source_documents']):
                st.write(f"    Source {idx}:")
                st.write(f"        Filepath: {elt.metadata['audio_url']}")

        user_query = ""

if app_selection == "Document Chatbot":
    document_chatbot()
else:
    coaching_ai()