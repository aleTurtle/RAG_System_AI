import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Definizione della classe RunnablePassThrough

class RunnablePassThrough(Runnable):
    def invoke(self, *args, **kwargs):
        return args[0]

# ============================================================
# Funzioni per la gestione dei PDF
# ============================================================
def get_pdf_text(pdf_paths):
    """Legge il testo dai PDF e lo restituisce come stringa unica."""
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    """Divide il testo in chunk più piccoli per facilitare l'elaborazione."""
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len
    )
    return text_splitter.split_text(text)



def ingest_pdf_files(pdf_files):
    """Carica e divide in chunk il contenuto dei PDF."""
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())  # Carica tutte le pagine del PDF
    
    # Dividere in chunk uniformi
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=250,
        separators=["\n\n", ".", "!", "?", " ", ""], #evita di spezzare il testo a metà di una parola
        add_start_index=True  # Forza la divisione in chunk regolari

    )
    
    return text_splitter.split_documents(docs)

# ============================================================
# Funzioni per la gestione delle pagine web
# ============================================================
def load_web_documents(urls):
    """Carica documenti da URL specificati."""
    loader = WebBaseLoader(web_paths=urls, encoding="utf-8")
    docs = loader.load()
    return docs

def split_web_documents(docs):
    """Divide i documenti web in chunk."""
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

# ============================================================
# Funzione per creare o aggiornare il vectorstore
# ============================================================
def get_or_update_vectorstore(text_chunks, db_path="C:/Users/User/Desktop/data4"):
    """Carica il vectorstore se esiste, altrimenti lo crea e lo salva."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("Database FAISS caricato con successo.")
        
        new_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.merge_from(new_vectorstore)
        vectorstore.save_local(db_path)
        print("Nuovi embedding aggiunti al database FAISS.")
    else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local(db_path)
        print("Nuovo database FAISS creato e salvato.")

    return vectorstore


# Funzione helper per formattare i documenti recuperati

def format_documents(docs):
    """Unisce il contenuto dei documenti recuperati."""
    return "\n\n".join(doc.page_content for doc in docs)

# ============================================================
# Impostazione del prompt template
# ============================================================
template = """Utilizza i seguenti elementi di contesto per rispondere alla domanda posta dall'utente. 
Se non conosci la risposta, rispondi dicendo che non la sai e non cercare di inventare una risposta.
Utilizza almeno tre frasi per rispondere e mantieni la risposta il più concisa possibile.
{context}
Domanda: {question}
Risposta:"""

prompt = PromptTemplate.from_template(template)


# Caricamento del modello 

model = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    temperature=0.0,
    api_key="not needed",
    model_name="gemma"
)

# ============================================================
# Costruzione della catena RAG
# ============================================================
def build_rag_chain(retriever):
    """Costruisce la catena RAG."""
    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassThrough()
        }
        | prompt
        | model
    )
    return rag_chain


# Funzione per interrogare il sistema

def query_pdf_system(rag_chain, question):
    """Esegue una query sulla catena RAG e restituisce solo il contenuto della risposta."""
    answer = rag_chain.invoke(question)
    
    if hasattr(answer, 'content'):
        return answer.content  
    
    if isinstance(answer, dict) and "content" in answer:
        return answer["content"]

    return str(answer) 


# Funzione per rispondere alle domande con il contesto della conversazione

def answer_question(question, context):
    """Genera una risposta basata sul contesto e sulla domanda."""
    prompt_chain = prompt | model
    return prompt_chain.invoke({"question": question, "context": context})



# ============================================================
# INTERFACCIA STREAMLIT
# ============================================================
st.title("RAG System")

# Memoria della conversazione
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Caricamento del vectorstore solo se non è già in memoria
if "vectorstore" not in st.session_state:

    pdf_files = [r"C:\\Users\\User\\Desktop\\Guida_L-INF_ita.pdf", 
                 r"C:\\Users\\User\\Desktop\\informatica_comun_digit.pdf"]
    
    web_urls = [
        "https://computerscience.unicam.it/organizzazione",
        "https://computerscience.unicam.it/people"
        
    ]

   # raw_text = get_pdf_text(pdf_files)
    #text_chunks_pdf = get_text_chunks(raw_text)

    text_chunks_pdf = ingest_pdf_files(pdf_files)
    print(f"Numero di chunk dai PDF: {len(text_chunks_pdf)}")

    web_docs = load_web_documents(web_urls)
    text_chunks_web = split_web_documents(web_docs)
    print(f"Numero di chunk dalle pagine web: {len(text_chunks_web)}")
    
    #all_text_chunks = text_chunks_pdf + [doc.page_content for doc in text_chunks_web]

    # Unire i chunk provenienti da PDF e web
    all_text_chunks = text_chunks_pdf + text_chunks_web 
    print(f"Totale chunk processati: {len(all_text_chunks)}")

    #st.session_state.vectorstore = get_or_update_vectorstore(all_text_chunks)
    st.session_state.vectorstore = get_or_update_vectorstore([doc.page_content for doc in all_text_chunks])

vectorstore = st.session_state.vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain = build_rag_chain(retriever)


# Mostra cronologia della chat
if st.session_state.conversation_history:
    for message in st.session_state.conversation_history:
        if message.startswith("Utente:"):
            st.chat_message("user").write(message[7:])
        elif message.startswith("Assistente:"):
            st.chat_message("assistant").write(message[11:])

# Input utente
question = st.chat_input("Fai una domanda:")

if question:
    st.session_state.conversation_history.append(f"Utente: {question}")
    st.chat_message("user").write(question)

    # Creazione del contesto
    context = "\n".join(st.session_state.conversation_history[-5:])
    
    # Recupero documenti rilevanti
    retrieved_docs = retriever.invoke(question)
    context += "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Generazione risposta
    answer = answer_question(question, context)


    def clean_response(answer):
    
        if hasattr(answer, 'content'):
            return answer.content  
    
        if isinstance(answer, dict) and "content" in answer:
            return answer["content"]

        return str(answer) 
    
    answer_text = clean_response(answer)
    
    st.session_state.conversation_history.append(f"Assistente: {answer_text}")
    st.chat_message("assistant").write(answer_text)
