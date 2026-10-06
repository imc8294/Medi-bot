import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource
def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=512,
        task="conversational"
    )

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


@st.cache_resource
def load_db():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


@st.cache_resource
def create_rag_chain():
    llm = load_llm()
    db = load_db()
    retriever = db.as_retriever(
        search_kwargs={"k": 3 }
    )


    contextualize_q_system_prompt = """
Given a chat history and the latest user question, which might
reference context in the chat history, formulate a standalone
question that can be understood without the chat history.

Do NOT answer the question.

Only rewrite the question if necessary.
If the question is already standalone, return it unchanged.
"""


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                contextualize_q_system_prompt
            ),

            MessagesPlaceholder(
                "chat_history"
            ),

            (
                "human",
                "{input}"
            ),
        ]
    )


    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )


    qa_system_prompt = """
You are Medi-Bot, a medical information assistant.

Use ONLY the information provided in the context below
to answer the user's question.

If the answer is not available in the context, say:
"I don't know based on the provided medical information."

Do not make up information.

Do not provide a diagnosis.

Keep the answer clear and easy to understand.

Context:

{context}
"""


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                qa_system_prompt
            ),

            MessagesPlaceholder(
                "chat_history"
            ),

            (
                "human",
                "{input}"
            ),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )


    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )


    return rag_chain




def main():

    st.set_page_config(
        page_title="Medi-Bot",
        page_icon="🩺"
    )

    st.title("🩺 Medi-Bot")
    st.write("Medical information assistant")


    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []


    for message in st.session_state.chat_history:

        if isinstance(message, HumanMessage):

            with st.chat_message("user"):
                st.markdown(message.content)

        elif isinstance(message, AIMessage):

            with st.chat_message("assistant"):
                st.markdown(message.content)


    user_query = st.chat_input(
        "Ask your medical question..."
    )


    if user_query:

        # Display user question

        with st.chat_message("user"):
            st.markdown(user_query)

        qa_chain = create_rag_chain()

        response = qa_chain.invoke(
            {
                "input": user_query,

                "chat_history":
                    st.session_state.chat_history
            }
        )


        # Get answer

        answer = response["answer"]


        with st.chat_message("assistant"):

            st.markdown(answer)


        st.session_state.chat_history.append(
            HumanMessage(
                content=user_query
            )
        )

        st.session_state.chat_history.append(
            AIMessage(
                content=answer
            )
        )


        if "context" in response:

            with st.expander("📚 Sources"):

                for i, doc in enumerate(
                    response["context"],
                    start=1
                ):

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        None
                    )

                    if page is not None:
                        page = page + 1

                    st.write(
                        f"**Source {i}:** {source}"
                    )

                    if page is not None:

                        st.write(
                            f"**Page:** {page}"
                        )


if __name__ == "__main__":
    main()









# import streamlit as st
# from dotenv import load_dotenv

# from langchain_huggingface import (
#     HuggingFaceEmbeddings,
#     HuggingFaceEndpoint,
#     ChatHuggingFace
# )

# from langchain_classic.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS


# load_dotenv()


# # --------------------------------------------------
# # Configuration
# # --------------------------------------------------

# DB_FAISS_PATH = "vectorstore/db_faiss"

# HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"


# # --------------------------------------------------
# # Load FAISS
# # --------------------------------------------------

# @st.cache_resource
# def load_db():

#     embedding_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     db = FAISS.load_local(
#         DB_FAISS_PATH,
#         embedding_model,
#         allow_dangerous_deserialization=True
#     )

#     return db


# # --------------------------------------------------
# # Load LLM
# # --------------------------------------------------

# @st.cache_resource
# def load_llm():

#     llm = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         temperature=0.5,
#         max_new_tokens=512,
#         task="conversational"
#     )

#     chat_model = ChatHuggingFace(
#         llm=llm
#     )

#     return chat_model


# # --------------------------------------------------
# # Create prompt
# # --------------------------------------------------

# def get_prompt():

#     prompt = """
# You are Medi-bot, a medical information assistant.

# Use ONLY the information provided in the context to answer the user's question.

# If the answer is not present in the context, say:
# "I don't know based on the provided medical information."

# Do not make up information.
# Do not use outside knowledge.

# Context:
# {context}

# Question:
# {question}

# Answer directly and clearly.
# """

#     return PromptTemplate(
#         template=prompt,
#         input_variables=["context", "question"]
#     )


# # --------------------------------------------------
# # Create QA Chain
# # --------------------------------------------------

# @st.cache_resource
# def load_qa_chain():

#     db = load_db()

#     llm = load_llm()

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(
#             search_kwargs={"k": 3}
#         ),
#         chain_type_kwargs={
#             "prompt": get_prompt()
#         },
#         return_source_documents=True
#     )

#     return qa_chain


# # --------------------------------------------------
# # Streamlit UI
# # --------------------------------------------------

# def main():

#     st.set_page_config(
#         page_title="Medi-bot",
#         page_icon="🩺",
#         layout="centered"
#     )

#     st.title("🩺 Medi-bot")

#     st.caption(
#         "Medical information assistant powered by RAG"
#     )

#     # Load chain
#     qa_chain = load_qa_chain()


#     # Chat history

#     if "messages" not in st.session_state:
#         st.session_state.messages = []


#     # Display previous messages

#     for message in st.session_state.messages:

#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])


#     # User input

#     user_query = st.chat_input(
#         "Ask your medical question..."
#     )


#     if user_query:

#         # Display user message

#         with st.chat_message("user"):
#             st.markdown(user_query)


#         st.session_state.messages.append(
#             {
#                 "role": "user",
#                 "content": user_query
#             }
#         )


#         # Get response

#         with st.chat_message("assistant"):

#             with st.spinner("Searching medical information..."):

#                 response = qa_chain.invoke(
#                     {
#                         "query": user_query
#                     }
#                 )

#                 answer = response["result"]

#                 st.markdown(answer)


#         # Save assistant response

#         st.session_state.messages.append(
#             {
#                 "role": "assistant",
#                 "content": answer
#             }
#         )


# if __name__ == "__main__":
#     main()







