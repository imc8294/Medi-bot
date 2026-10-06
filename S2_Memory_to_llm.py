from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# step 1 - Setup LLM (Mistral with HuggingFace)

huggingface_repo_id = "meta-llama/Llama-3.1-8B-Instruct"

def load_llm(HF_repo_id):
    llm = HuggingFaceEndpoint(repo_id = HF_repo_id, 
                              temperature = 0.5,
                              max_new_tokens = 512,
                              task = "conversational"
                              )

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model

# step 2 - Connect LLM with FAISS and Create Chain



prompt = """
Use the piece of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up the answer.
Don't provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def Custom_prompt(prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    return prompt


#Load DB

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model,  allow_dangerous_deserialization=True)

#Create QA Chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever = db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": Custom_prompt(prompt)},
    return_source_documents=True

)


# Invoke with single query
user_query = input("Enter your query: ")
response = qa_chain.invoke({'query': user_query})

print("Response: ", response['result'])
print("Sources: ", response['source_documents'])

