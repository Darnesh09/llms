import os
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain import HuggingFaceHub
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

astra_db_token = os.getenv("Astra_DB_token")
astra_db_id = "ea60de45-be82-421c-aca8-e8aaafe31cc6"
huggingface_api_token = os.getenv("huggingface_token")

# Establish connection using existing database and keyspace (modify if needed)
# session = Cassandra.get_session(
#     token=astra_db_token,
#     database_id=astra_db_id,
#     keyspace="default_keyspace"  # Replace with your keyspace name
# )

llm_t5 = HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-248M", huggingfacehub_api_token=huggingface_api_token)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store instance (assuming table already exists)
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="MiniLM_embedd",
    session=None,
    keyspace=None
)

# Create the vector store index wrapper
astra_vectr_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Now you can use astra_vectr_index for your retrieval tasks

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"" % query_text)
    answer = astra_vectr_index.query(query_text, llm=llm_t5).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
