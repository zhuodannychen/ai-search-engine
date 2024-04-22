from dotenv import load_dotenv
from flask import Flask, request, jsonify
from db import connectDB
from indexer import Indexer
from search import Search
from pymilvus import MilvusClient, connections, utility, Collection, DataType, FieldSchema, CollectionSchema
import os

load_dotenv()

DB_CLIENT = MilvusClient(
    uri=os.getenv("DB_ENDPOINT"),
    token=os.getenv("TOKEN")
)
COLLECTION_NAME = "documents"

app = Flask(__name__)
indexer = Indexer(DB_CLIENT, COLLECTION_NAME)
searcher = Search(DB_CLIENT, COLLECTION_NAME)


@app.route('/')
def home():
    return "Welcome to the simple Flask API!"

@app.route('/index', methods=['POST'])
def create_index():
    website = request.get_json()['website']
    indexer.index_page(website)
    return "Finished indexing webpage!"

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', default=None, type=str)
    print(query)
    res = searcher.search_query(query)
    return res

if __name__ == '__main__':
    connections.connect("default",
                        uri=os.getenv("DB_ENDPOINT"),
                        token=os.getenv("TOKEN"))
    check_collection = utility.has_collection(COLLECTION_NAME)
    print("DB connection success!")

    if not check_collection:
        dim = 1536
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192)
        url_field = FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048)
        schema = CollectionSchema(fields=[id_field, vector_field, text_field, url_field])
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"created new collection: {collection}")

        index_params = {
            "metric_type": "L2",  # Euclidean distance
            "index_type": "IVF_FLAT",  # Index type, IVF_FLAT is commonly used
            "params": {"nlist": 100}  # Number of clusters, adjust based on your dataset size and nature
        }
        # Create an index on the vector field, assume the vector field is named "vector"
        collection.create_index(field_name="vector", index_params=index_params)
        print("Created index")



    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    app.run(port=8000, debug=True)
