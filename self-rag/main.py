import os
from dotenv import load_dotenv
from typing import Literal
from pymongo import MongoClient
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import (
    StructuredOutputParser, PydanticOutputParser
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


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

# Docs topic
DOCS_TOPIC = "black holes"


def store_documents():
    """
    Load docs from folder and generate chunks from docs
    """
    # Delete documents from collection
    MONGODB_COLLECTION.delete_many({})
    # loader = PyPDFDirectoryLoader('documents/')
    # data = loader.load()

    # Generate chunks from docs
    # chunk_sizes = [128, 256, 512, 1024, 2048]
    # text_splitter = RecursiveCharacterTextSplitter(
    #    chunk_size=128, chunk_overlap=20)
    # docs = text_splitter.split_documents(data)

    docs_path = [
        "documents/0905.3243v2.pdf",
        "documents/1103.2064v2.pdf",
        "documents/1209.2243v1.pdf",
        "documents/1702.07766v2.pdf",
        "documents/81517305.pdf"
    ]

    loader = UnstructuredLoader(
        docs_path, strategy="auto", chunking_strategy="basic")
    docs = loader.load()

    # Store documents in MongoDB Atlas Vector Search
    x = MongoDBAtlasVectorSearch.from_documents(
        documents=docs, embedding=OpenAIEmbeddings(disallowed_special=()), collection=MONGODB_COLLECTION, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )


def get_documents(query, limit=20):
    """Retrieve documents from vector store comparing query's vector

    Args:
        query (string): Sentence to be vetorized and compared in atlas vector search

    Returns:
        docs_metadata (list): Results from avs
    """
    results = list(MONGODB_COLLECTION.aggregate([
        {
            '$vectorSearch': {
                'index': 'vector_index',
                'queryVector': OpenAIEmbeddings().embed_query(query),
                'numCandidates': 200,
                'limit': limit,
                'path': 'embedding'
            }
        }
    ]))
    return results


def get_retriever(most_relevant_docs=200):
    """Get retriever in langchain format from vector store

    Args:
        docs_limit (int): Number of priority docs to be retrieved

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
        search_kwargs={"k": most_relevant_docs,  # , "pre_filter": {
                       "post_filter_pipeline": [{"$limit": 5}]
                       # "page": {"$eq": 555}}, "post_filter_pipeline": [{"$limit": 1}],
                       # "score_threshold": 0.5,
                       }
    )
    return retriever


def context_chain(query):
    """Generate context chain

    Args:
        query (string): Sentence to be used in prompt

    Returns:
        answer (string): From context chain
    """

    # Define model
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    # Prompt
    context_template = """
    Answer the question based only on the following context: {context}
    ---
    Question: {question}
    Answer:"""

    context_prompt = ChatPromptTemplate.from_template(context_template)

    # Chain
    """chain = (
        {
            "context": get_retriever(),
            "question": RunnablePassthrough()
        }
        | context_prompt
        | llm
        | StrOutputParser()
    )"""

    # return chain.invoke(query)

    # Create buffer for memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Conversational chain, lock maxtoken passed with retriever, skip status 429
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_retriever(),
        memory=memory
    )
    return chain({"question": query})["answer"]


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
    routing_system_template = f"""You are an expert at routing a user question to a vectorstore or websearch.
    The vectorstore contains documents related to {DOCS_TOPIC}.
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
    chain = chat_prompt | llm | routing_parser

    print(chain.invoke(
        {"question": query, "format_instructions": routing_format_instructions}))


class GradeDocuments(BaseModel):
    """
    Boolean assignation for retrieved documents
    """
    relevance_score: Literal["true", "false"] = Field(
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
    chain = chat_prompt | llm | relevance_parser

    # Get documents from vector store related to query
    dc = get_documents(query)
    dc = ([str(x["text"]) for x in dc])

    result = chain.invoke({
        "question": query,
        "document": dc,
        "format_instructions": retrieval_format_instructions
    })

    print(result)


class GradeHallucionations(BaseModel):
    """
    Boolean assignation for hallucination present in answer
    """
    hallucination_score: Literal["true", "false"] = Field(
        ..., description="Don't search for additional information. Answer is supported by facts, 'true' or 'false'")


def grader_hallucination_chain(query):
    """Generate grade hallucination chain to check grade of hallucination

    Args:
        query (string): Sentence to be used in prompt
    """
    # Define model
    llm = OpenAI()

    # Output parser
    grader_hallucination_parser = PydanticOutputParser(
        pydantic_object=GradeHallucionations)

    # System prompt
    grader_hallucination_system_template = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts.
    Restrict yourself to give a boolean score, either 'true' or 'false'. If the answer is supported or partially supported by the set of facts, consider it a 'false'.
    Don't consider calling external APIs for additional information as consistent with the facts."""

    grader_hallucination_system_message_prompt = SystemMessagePromptTemplate.from_template(
        grader_hallucination_system_template)

    # Human prompt
    grader_hallucination_human_template = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}\n\n{format_instructions}"
    grader_hallucination_human_message_prompt = HumanMessagePromptTemplate.from_template(
        grader_hallucination_human_template)

    # Combined prompts
    grader_hallucination_chat_prompt = ChatPromptTemplate.from_messages(
        [grader_hallucination_system_message_prompt,
            grader_hallucination_human_message_prompt]
    )

    # Pass instructions
    grader_hallucination_format_instructions = grader_hallucination_parser.get_format_instructions()

    # Chain
    chain = grader_hallucination_chat_prompt | llm | grader_hallucination_parser

    # Get documents from vector store related to query
    dc = get_documents(query)
    dc = ([str(x["text"]) for x in dc])

    # Get answer from context_chain
    ac = context_chain(query)

    result = chain.invoke({
        "documents": dc,
        "generation": ac,
        "format_instructions": grader_hallucination_format_instructions
    })

    print(result)


class GradeAnswer(BaseModel):
    """Boolean grade if the question was answered right
    """
    answered: Literal["true", "false"] = Field(
        ..., description="Answer addresses the question, 'true' or 'false'"
    )


def grader_answer_chain(query):
    """Chain to check if the answer corresponds to the question

    Args:
        query (string): Sentence to be used in prompt
    """

    # Define model
    llm = OpenAI()

    # Output parser
    grader_answer_parser = PydanticOutputParser(
        pydantic_object=GradeAnswer
    )

    # System prompt
    grader_answer_system_template = """You are a grader assessing whether a LLM generation addresses / resolves a question.
    Give a binary score 'true' or 'false'. 'true' means that the answer resolves the question. Your answer must be in json format"""  # Force answer to be in json format

    grader_answer_system_message_prompt = SystemMessagePromptTemplate.from_template(
        grader_answer_system_template
    )

    # Human prompt
    grader_answer_human_template = "User question: \n\n {question} \n\n LLM generation: {generation}\n\n{format_instructions}"
    grader_answer_human_message_prompt = HumanMessagePromptTemplate.from_template(
        grader_answer_human_template
    )

    # Combined prompts
    grader_answer_chat_prompt = ChatPromptTemplate.from_messages(
        [grader_answer_system_message_prompt,
         grader_answer_human_message_prompt]
    )

    # Pass instructions
    grader_answer_format_instructions = grader_answer_parser.get_format_instructions()

    # Chain
    chain = grader_answer_chat_prompt | llm | grader_answer_parser

    # Get answer from context_chain
    ac = context_chain(query)

    result = chain.invoke({
        "question": query,
        "generation": ac,
        "format_instructions": grader_answer_format_instructions
    })

    print(result)


if __name__ == '__main__':
    # store_documents()
    # get_documents('napoleon')
    # print(context_chain("what are black holes?"))
    # print(len(get_retriever().invoke("what are black holes?")))
    # routing_chain('what do you know about black holes?')
    # relevance_chain("what do you know about black holes?")
    # grader_hallucination_chain("what do you know about black holes?")
    grader_answer_chain("tell me something about black holes")
