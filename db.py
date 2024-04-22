import configparser
import time
import random
from pymilvus import connections, utility

def connectDB():
    cfp = configparser.RawConfigParser()
    cfp.read('config.ini')
    milvus_uri = cfp.get('ai-search', 'uri')
    token = cfp.get('ai-search', 'token')
    connections.connect("default",
                        uri=milvus_uri,
                        token=token)
    print(f"Connecting to DB: {milvus_uri}")