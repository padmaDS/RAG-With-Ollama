import os
import time

# User prompt
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# VectorDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

# LLMs
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Retrieval
from langchain.chains import RetrievalQA

# Ensure directories exist
if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')

if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

# Set up initial state variables
template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

Context: {context}
History: {history}

User: {question}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question",
)

vectorstore = Chroma(persist_directory='vectorDB',
                     embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                         model="llama3")
                     )

llm = Ollama(base_url="http://localhost:11434",
             model="llama3",
             verbose=True,
             callback_manager=CallbackManager(
                 [StreamingStdOutCallbackHandler()]),
             )

chat_history = []

def process_pdf(uploaded_file_path):
    print("File uploaded successfully")

    loader = PyPDFLoader(uploaded_file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="llama3")
    )

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )

    return qa_chain

def chat_with_pdf(qa_chain, user_input):
    chat_history.append({"role": "user", "message": user_input})
    print(f"User: {user_input}")

    print("Assistant is typing...")
    response = qa_chain(user_input)
    full_response = ""

    for chunk in response['result'].split():
        full_response += chunk + " "
        time.sleep(0.05)
        print(full_response + "▌", end="\r", flush=True)

    print(full_response)
    chat_history.append({"role": "assistant", "message": response['result']})

# Example usage
uploaded_file_path = r'data\20 columns.pdf'  # Replace with your actual PDF path
qa_chain = process_pdf(uploaded_file_path)

# Simulate user input
user_input = "give me number of participants who never received mmg by income and race?"
response = chat_with_pdf(qa_chain, user_input)

print(response)
