import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load keys from env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['MONGO_URI'] = os.getenv('MONGO_URI')

# MongoDB db
DB_NAME = "test"
COLLECTION_NAME = "docs"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
EMBEDDING_FIELD_NAME = "embedding"
client = MongoClient()
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]
