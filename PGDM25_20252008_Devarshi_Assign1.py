import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
# 1. Load .env
# ---------------------------
st.set_page_config(
    page_title="Indian Customs RAG Assistant",
    layout="wide",
    page_icon="📦"
)
load_dotenv("my.env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in my.env")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------
# 2. LLM + Embeddings
# ---------------------------
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# ---------------------------
# 3. Predefined documents
# ---------------------------
PDF_FILES = ["pdf_c_3_merged.pdf"]
CSV_FILES = ["customs_duty_table.csv"]
URLS = [
    "https://www.india-briefing.com/doing-business-guide/india/taxation-and-accounting/customs-duty-and-import-export-taxes-in-india",
    "https://cleartax.in/s/customs-duty-india"
]

DB_DIR = "rag_db"

# ---------------------------
# 4. Load and cache documents
# ---------------------------
@st.cache_data(show_spinner=False)
def load_all_documents():
    docs = []

    # PDFs
    for pdf in PDF_FILES:
        if os.path.exists(pdf):
            loader = PyPDFLoader(pdf)
            docs.extend(loader.load())
        else:
            st.warning(f"⚠ PDF not found: {pdf}")

    # CSVs
    for csv in CSV_FILES:
        if os.path.exists(csv):
            loader = CSVLoader(csv, encoding="utf-8")
            docs.extend(loader.load())
        else:
            st.warning(f"⚠ CSV not found: {csv}")

    # URLs (fetched once)
    for url in URLS:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"⚠ Failed to load URL {url}: {e}")

    return docs

all_docs = load_all_documents()
st.write(f"✅ Loaded {len(all_docs)} documents.")

# ---------------------------
# 5. Build or load vectorstore (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    if os.path.exists(DB_DIR):
        return Chroma(embedding_function=embeddings, persist_directory=DB_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    db.persist()
    return db

db = get_vectorstore()

# ---------------------------
# 6. Retrieval + Answer
# ---------------------------
def ask_llm(query):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    if not results:
        return "❌ No relevant documents found."

    context = "\n\n".join([doc.page_content for doc in results])
    llm = get_llm()
    answer = llm.invoke(f"Use ONLY the context to answer. Also provide sample calculations. Generate output in table.\n\nContext:\n{context}\n\nQuestion: {query}")
    return answer.content

# ---------------------------
# 7. Streamlit UI
# ---------------------------


st.markdown(
    """
    <div style="text-align:center; padding:20px; background-color:#2E86C1; border-radius:10px;">
        <h1 style="color:white;">📦 Indian Customs RAG Assistant</h1>
        <p style="color:white; font-size:18px;">
            Ask anything about Indian Customs Duty, Import-Export Taxes, Rules & Regulations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Initialize Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # stores tuples (question, answer)



st.info("""
    **How to use this RAG bot:**  
    1. Type your question about customs duty in the input box below.  
    2. Press **Ask** to get an answer based on the documents and reliable sources.  
    3. The bot will give a summarized response and, if relevant, example calculations.  
    """)

st.write("Ask any question related to Indian customs duty:")

# --- INPUT BOX ---
user_query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the customs duty for importing laptops?"
)

# --- SEND BUTTON ---
if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("⚠ Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = ask_llm(user_query)
        st.markdown("## 🧾 Answer")
        st.markdown(f"<div style='padding:15px; background:#F7F9F9; border-radius:10px'>{answer}</div>", unsafe_allow_html=True)


# --- DISPLAY QA PAIRS (ONE AFTER ANOTHER) ---
for q, a in st.session_state.history:
    st.markdown(f"### ❓ Question:")
    st.write(q)

    st.markdown(f"### ✅ Answer:")
    st.write(a)

    st.markdown("---")
    


# ---------------------------
# Footer Section
# ---------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
        Made with ❤️ using Streamlit & LangChain | Powered by Google Generative AI
    </div>
    """,
    unsafe_allow_html=True
)


