from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import os
import chromadb

app = Flask(__name__)

chat_history = []
folder_path = "db"

# Initialize model and embedding
cached_llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()

# Text splitter for documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Prompt template for retrieval
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are an intelligent chatbot, you are good at retrieving information from the documents. If you do not have an answer from the provided information, say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Global variable to store the vector store client
vector_store_from_client = None


@app.route("/ai", methods=["POST"])
def ai_post():
    try:
        json_content = request.json
        query = json_content.get("query")
        print(f"query: {query}")

        # Invoke cached LLM to get the response
        response = cached_llm.invoke(query)
        print(f"LLM response: {response}")

        return jsonify({"answer": response})
    except Exception as e:
        print(f"Error in /ai: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/loading-pdf", methods=["POST"])
def pdf_post():
    global vector_store_from_client
    try:
        file = request.files["file"]
        file_name = file.filename
        save_file = os.path.join("pdf", file_name)
        file.save(save_file)
        print(f"File saved as: {save_file}")

        # Load PDF and split into chunks
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        print(f"Loaded {len(docs)} documents.")

        # Split documents into smaller chunks
        chunks = text_splitter.split_documents(docs)
        print(f"Generated {len(chunks)} document chunks.")

        # Initialize Chroma with persistence support
        persistent_client = chromadb.PersistentClient()
        vector_store_from_client = Chroma(
            client=persistent_client,
            collection_name="collection_name",
            embedding_function=embedding,
        )

        # Add documents (chunks) to the vector store
        vector_store_from_client.add_documents(documents=chunks)
        
        return jsonify({
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": len(chunks),
        })
    except Exception as e:
        print(f"Error in /loading-pdf: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/query-pdf", methods=["POST"])
def askPDFPost():
    global vector_store_from_client
    try:
        if vector_store_from_client is None:
            return jsonify({"error": "Vector store is not initialized. Please upload a PDF first."}), 400

        print("Post /query_pdf called")
        json_content = request.json
        query = json_content.get("query")
        print(f"query: {query}")

        print("Creating chain")
        retriever = vector_store_from_client.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,
                "score_threshold": 0.1,
            },
        )

        print(f"created a chain {retriever}")

        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                (
                    "human",
                    "Given the above conversation, generate a search query to look up information relevant to the conversation",
                ),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=cached_llm, retriever=retriever, prompt=retriever_prompt
        )

        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain,
        )

        result = retrieval_chain.invoke({"input": query})
        print(result["answer"])
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

        sources = []
        for doc in result["context"]:
            sources.append(
                {"source": doc.metadata["source"], "page_content": doc.page_content}
            )

        response_answer = {"answer": result["answer"], "sources": sources}
        return jsonify(response_answer)
    except Exception as e:
        print(f"Error in /query-pdf: {str(e)}")
        return jsonify({"error": str(e)}), 500


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
