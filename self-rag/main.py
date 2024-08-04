import os
from dotenv import load_dotenv
from typing import Literal
from pymongo import MongoClient
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser

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
    """
    Load docs from folder and generate chunks from docs
    """
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
    """Retrieve documents from vector store comparing query's vector

    Args:
        query (string): Sentence to be vetorized and compared in atlas vector search
    """
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


def get_retriever(chunk_limit=200):
    """Get retriever in langchain format from vector store

    Args:
        chunk_limit (int): Number of chunks to be retrieved

    Returns:
        Retriever: Retriever from Atlas Vector Search
    """
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ['MONGO_URI'],
        DB_NAME + '.' + COLLECTION_NAME,
        OpenAIEmbeddings(disallowed_special=()),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )

    retriever = vector_search.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': chunk_limit,
            'post_filter_pipeline': [{'$limit': 25}]
        }
    )
    return retriever


class RouteQuery(BaseModel):
    """
    Route user query to most relavant datasource
    """
    datasource: Literal['vectorstore', 'websearch'] = Field(
        ...,
        description='Given a user question choose to route it to web search or a vectorstore.',
    )


def routing_chain(query):
    """Generate routing chain to decide to answer from retriver or not

    Args:
        query (string): Sentence to be used in prompt
    """

    # Define model
    llm = OpenAI()

    # Output parser
    routing_parser = PydanticOutputParser(pydantic_object=RouteQuery)

    # Prompts for system and human, specify topic of documents
    routing_system_template = """You are an expert at routing a user question to a vectorstore or websearch.
    The vectorstore contains documents related to black holes.
    Use the vectorstore for questions on these topics. For all else, use websearch."""

    routing_system_message_prompt = SystemMessagePromptTemplate.from_template(
        routing_system_template)

    routing_human_template = "{question}\n\n{format_instructions}"

    routing_human_message_prompt = HumanMessagePromptTemplate.from_template(
        routing_human_template)

    # Combined prompts
    chat_prompt = ChatPromptTemplate.from_messages(
        [routing_system_message_prompt, routing_human_message_prompt]
    )

    # Pass instructions
    routing_format_instructions = routing_parser.get_format_instructions()

    # Create chain
    routing_chain = chat_prompt | llm | routing_parser

    print(routing_chain.invoke(
        {"question": query, "format_instructions": routing_format_instructions}))


class GradeDocuments(BaseModel):
    """
    Boolean assignation for retrieved documents
    """
    score: Literal["true", "false"] = Field(
        description="Documents relevant to question")


def relevance_chain(query):
    """Generate relevance chain to get relevant documents

    Args:
        query (string): Sentence to be used in prompt
    """

    # Define model
    llm = OpenAI()

    # Output parser
    relevance_parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    # System prompt, instructions to answer true | false
    relevance_system_template = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a boolean score 'true' or 'false' score to indicate whether the document is relevant to the question."""

    relevance_system_message_prompt = SystemMessagePromptTemplate.from_template(
        relevance_system_template)

    # Human prompt
    relevance_human_template = "Retrieved document: \n\n {document} \n\n User question: {question}\n\n{format_instructions}"
    relevance_human_message_prompt = HumanMessagePromptTemplate.from_template(
        relevance_human_template)

    # Combined prompts
    chat_prompt = ChatPromptTemplate.from_messages(
        [relevance_system_message_prompt, relevance_human_message_prompt]
    )

    # Pass instructions
    retrieval_format_instructions = relevance_parser.get_format_instructions()

    # Create chain
    retrieval_grader_relevance = chat_prompt | llm | relevance_parser

    # Get documents
    docs_from_retriever = get_retriever().invoke(query)
    doc_txt = docs_from_retriever[0].page_content

    result = retrieval_grader_relevance.invoke({
        "question": query,
        "document": doc_txt,
        "format_instructions": retrieval_format_instructions
    })

    print(result)


if __name__ == '__main__':
    # store_documents()
    # get_documents('napoleon')
    # routing_chain('what do you know about black holes?')
    relevance_chain("what do you know about black holes")
