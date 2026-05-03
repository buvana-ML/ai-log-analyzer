import streamlit as st
import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# 1. LOAD & CLEAN DATA
# -----------------------------
df = pd.read_csv("data/logs.csv")

# Keep required columns
df = df[['message', 'severity', 'status', 'category']]

# Remove missing values
df = df.dropna(subset=['message'])

# Remove duplicates
df = df.drop_duplicates(subset=['message'])

# Keep only FAILURE logs 
df = df[df['status'] == "FAILURE"]

# Remove noisy / useless logs
df = df[~df['message'].str.contains("Press CTRL", na=False)]
df = df[~df['message'].str.contains("test", na=False)]

# -----------------------------
# 2. CREATE DOCUMENTS
# -----------------------------
texts = df['message'].astype(str).tolist()
docs = [Document(page_content=text) for text in texts]

# -----------------------------
# 3. EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# 4. VECTOR DATABASE
# -----------------------------
db = FAISS.from_documents(docs, embeddings)

# -----------------------------
# 5. EXPLANATION
# -----------------------------
def explain_log(log):
    log = log.lower()

    if "exception" in log and "db" in log:
        return f"Issue detected: {log}. Likely due to database failure."
    elif "exception" in log:
        return "Application crashed due to an exception."
    elif "timeout" in log:
        return "External service timeout or delayed response detected."
    elif "db" in log or "database" in log:
        return "Database connectivity issue detected."
    elif "http" in log:
        return "HTTP request failure — API or server issue."
    elif "memory" in log:
        return "Possible memory leak or high memory usage."
    else:
        return "Unknown anomaly — requires further investigation."

# -----------------------------
# 6. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Log Analyzer", layout="wide")

st.title("🧠 AI Log Analyzer")
st.write("Detect and understand system failures using semantic search and AI-based explanations.")

query = st.text_input("🔍 Enter your issue (e.g. database failure)")

# -----------------------------
# 7. SEARCH + DISPLAY
# -----------------------------
if query:
    results = db.similarity_search_with_score(query, k=3)

    st.subheader("📊 Top Relevant Failure Logs")

    for i, (r, score) in enumerate(results):
        st.markdown(f"### 🔹 Result {i+1}")

        st.markdown("### 📄 Log")
        st.code(r.page_content)

        st.markdown(f"**🧠 Explanation:** {explain_log(r.page_content)}")

        st.markdown(f"**📉 Similarity Score:** {round(score, 3)}")

        st.markdown("---")
        
# -----------------------------
# 8. SIDEBAR INFO
# -----------------------------
st.sidebar.title("About")
st.sidebar.write(
    """
    This project uses:
    - Semantic search (FAISS)
    - Sentence embeddings (MiniLM)
    - Rule-based AI explanations
    
    Built for anomaly detection in system logs.
    """
)
