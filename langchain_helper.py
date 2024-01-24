from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS = OpenAIEmbeddings()

COLLECTION_NAME = 'Embedding_Store'
usr_file = 'Saud_Resume.pdf'
qdrant_client = QdrantClient(host='localhost', port=6333)


def instance_Qdrant():
    instance = Qdrant(client= qdrant_client, collection_name= COLLECTION_NAME, embeddings= EMBEDDINGS)
    return instance
                            

def vectorDB_for_embeddings(usr_file):
    if usr_file.endswith('.pdf'):
        file = PyPDFLoader(usr_file)
    elif usr_file.endswith('.docx'):
        file = Docx2txtLoader(usr_file)
    else:
        print('Unsupported File Format')
    loaded_text = file.load_and_split()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = chunks.split_documents(loaded_text)
    instance = instance_Qdrant()
    instance.from_documents(documents= docs, embedding= EMBEDDINGS, collection_name = COLLECTION_NAME)


def get_query_from_user(qdrant_db, query, k=4):
    get_similarContent = qdrant_db.similarity_search(query, k)
    all_docs_content = ''.join([d.page_content for d in get_similarContent])

    client = OpenAI(model= 'gpt-3.5-turbo-instruct', temperature=0.8)

    prompt = PromptTemplate(input_variables=['question', 'content'],
                            template= """
                            You are a helpful assistant that answers according to the similarity search with the content,
                            uploaded by a user:

                            Question is: {question}
                            Content: {content}
                        
                            Steps for Answering:

                            1. Read the content, take your time and think carefully before answering.
                            2. Only answer the question if you have relevant information, otherwise do not answer to that question.
                            3. If you don't find any relevant match for the question in the content, just say 'I don't know the answer, is there any further I can do for you ?'
                            4. If you are asked about yourself, then answer to the question as you are ChatGPT and in that case don't use content information to answer.
                            5. Do not answer to any generic and random questions asked be the user even if the user insists for it. Just say 'It's not allowed to answer to that question,
                            I can only answer according to the content'.
                            6. Do not create generic answers from your own. Just answer from the content provided.
                            7. If the question has no match with the content, Kindly do not answer to it.
                           
                            """)
    chain = LLMChain(llm=client, prompt=prompt, output_key='text')
   
    response = chain.invoke({'question':query, 'content':all_docs_content})
    return response['text']

if __name__ == '__main__':
    vectorDB_for_embeddings(usr_file)
    vec_db = instance_Qdrant()
    print(get_query_from_user(vec_db,'What are the skills of Muhammad Saud Shakeel mentioned in the document ?'))