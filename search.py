from dotenv import load_dotenv
import os
import openai 


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class Search:

    def __init__(self, db_client, collection_name):
        self.db_client = db_client
        self.collection_name = collection_name

    def query_vector_db(self, embedding, lim=10):
        result = self.db_client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=lim,
            output_fields=["url", "text"]
        )

        top_urls = []
        for res in result[0]:
            top_urls.append(res['entity']['url'])
        return top_urls

    def search_query(self, query):
        embedding_model = 'text-embedding-3-small'
        embedding = openai.Embedding.create(input = [query], model=embedding_model)['data'][0]['embedding']

        top_result = self.query_vector_db(embedding)
        print(top_result)
        return top_result


