from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


#Step 1 :- Load Raw files(PDF)


DATA_PATH = "data/"
def load_pdf(data):
    loader = DirectoryLoader(data,
                              glob='*.pdf',
                              loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf(data = DATA_PATH)
# print("documents", len(documents))



#Step 2 :- Create Chunks
def create_chunks(extracted_data):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
    text_chunk = text_splitters.split_documents(extracted_data)
    return text_chunk
text_chunks = create_chunks(extracted_data=documents)
# print("chunks= ",len(text_chunks))


#Step 3 :- Create Vector embedding 

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

#Step 4 :- Storing Embedding in FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)