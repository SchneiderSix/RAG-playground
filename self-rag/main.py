import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

# Load keys from env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['MONGO_URI'] = os.getenv('MONGO_URI')

# MongoDB db
DB_NAME = 'test'
COLLECTION_NAME = 'docs'
ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'
EMBEDDING_FIELD_NAME = 'embedding'
client = MongoClient(os.environ['MONGO_URI'])
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]


def store_documents():
    # Load docs
    loader = PyPDFDirectoryLoader('documents/')
    data = loader.load()

    # Generate chunks from docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Store documents in MongoDB Atlas Vector Search
    x = MongoDBAtlasVectorSearch.from_documents(
        documents=docs, embedding=OpenAIEmbeddings(disallowed_special=()), collection=MONGODB_COLLECTION, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )


def get_documents(query):
    # Retrieve documents from vector store comparing query's vector
    results = MONGODB_COLLECTION.aggregate([
        {
            '$vectorSearch': {
                'index': 'vector_index',
                'queryVector': OpenAIEmbeddings().embed_query(query),
                'numCandidates': 200,
                'limit': 20,
                'path': 'embedding'
            }
        }
    ])

    for i in results:
        print(i)


if __name__ == '__main__':
    # store_documents()
    get_documents('napoleon')
