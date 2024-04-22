from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import math
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Indexer:

    def __init__(self, db_client, collection_name):
        self.db_client = db_client
        self.collection_name = collection_name
        self.MODEL_CHUNK_SIZE = 4096

    def fetch_sitemap(self, sitemap_url):
        response = requests.get(sitemap_url)
        response.raise_for_status()  # will raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'lxml')
        urls = [loc.text for loc in soup.find_all('loc')]
        return urls

    def fetch_html_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
    
            # Find the body element and extract its text
            soup = BeautifulSoup(response.content, 'lxml')
            body_text = soup.body.get_text(separator=' ', strip=True)
            return body_text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def index_page(self, sitemap_url):
        lim = 5
        urls = self.fetch_sitemap(sitemap_url)
        for url in urls[:lim]:
            html_content = self.fetch_html_content(url)
            if html_content:
                self.add_html_to_vectordb(html_content, url)


    def create_embedding(self, text):
        embedding_model = 'text-embedding-3-small'
        embedding = openai.Embedding.create(input = [text], model=embedding_model)['data'][0]['embedding']
        return embedding

    def add_html_to_vectordb(self, content, path):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.MODEL_CHUNK_SIZE,
            chunk_overlap  = math.floor(self.MODEL_CHUNK_SIZE/10)
        )

        docs = text_splitter.create_documents([content])

        for doc in docs:
            print(path, len(doc.page_content))
            embedding = self.create_embedding(doc.page_content)
            self.insert_embedding(embedding, doc.page_content, path)

    def insert_embedding(self, embedding, text, path):
        row = {
            'vector': embedding,
            'text': text,
            'url': path
        }

        self.db_client.insert(self.collection_name, data=[row])
