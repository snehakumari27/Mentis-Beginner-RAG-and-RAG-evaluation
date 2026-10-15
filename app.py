import streamlit as st
import os
import tempfile
from ingest import DocumentIngestor
from vector_store import add_documents_to_db
from rag_pipeline import answer_query, DEFAULT_MODEL

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mentis:AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Mentis:AI Study Assistant")

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_chunks" not in st.session_state:
    st.session_state.ingested_chunks = []

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

if "retrieved_sources" not in st.session_state:
    st.session_state.retrieved_sources = []


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_document_text(doc):
    if hasattr(doc, "page_content"):
        return doc.page_content
    if isinstance(doc, dict):
        return doc.get("page_content", str(doc))
    return str(doc)

def get_document_metadata(doc):
    if hasattr(doc, "metadata"):
        return doc.metadata
    if isinstance(doc, dict):
        return doc.get("metadata", {})
    return {}


# ==========================================
# SIDEBAR: SOURCE INGESTION
# ==========================================

with st.sidebar:
    st.header("Add Sources")

    st.write("Paste a YouTube Video or Playlist link:")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/..."
    )

    st.write("Or upload local documents:")
    uploaded_files = st.file_uploader(
        "Upload PDF or PPTX files",
        type=["pdf", "pptx"],
        accept_multiple_files=True
    )

    if st.button("Process & Add to Database"):
        if youtube_url.strip() or uploaded_files:
            
            with st.spinner("Processing sources and creating chunks..."):
                ingestor = DocumentIngestor()
                all_docs = []
                
                # Handle YouTube
                if youtube_url.strip():
                    try:
                        yt_docs = ingestor.ingest_youtube(youtube_url)
                        if yt_docs:
                            all_docs.extend(yt_docs)
                        else:
                            st.warning("Could not find any transcripts for the YouTube link.")
                    except Exception as e:
                        st.error(f"Error processing YouTube source: {e}")

                # Handle Uploads
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                            
                        try:
                            if file_ext == "pdf":
                                file_docs = ingestor.ingest_pdf(tmp_path)
                                all_docs.extend(file_docs)
                            elif file_ext == "pptx":
                                file_docs = ingestor.ingest_ppt(tmp_path)
                                all_docs.extend(file_docs)
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                        finally:
                            os.remove(tmp_path)

                # Save to Database
                if all_docs:
                    st.session_state.ingested_chunks = all_docs
                    chunks_added = add_documents_to_db(all_docs)
                    st.success(f"Success! Added {chunks_added} chunks to the database.")
                else:
                    if not st.session_state.get('ingested_chunks'):
                        st.session_state.ingested_chunks = []
                    
        else:
            st.warning("Please enter a URL or upload a file first.")


# ==========================================
# MAIN PAGE
# ==========================================

st.header("Ask Your Question")
st.caption(f"Currently using local model: `{DEFAULT_MODEL}`")


# ==========================================
# CHUNK INSPECTION
# ==========================================

if st.session_state.ingested_chunks:
    st.subheader("📦 Document Chunking")
    st.write(f"Total chunks created: **{len(st.session_state.ingested_chunks)}**")

    with st.expander("🔍 View all chunks created during ingestion", expanded=False):
        for i, doc in enumerate(st.session_state.ingested_chunks, start=1):
            text = get_document_text(doc)
            metadata = get_document_metadata(doc)
            title = metadata.get("title", metadata.get("source", "Unknown Source"))
            
            st.markdown(f"### Chunk {i}")
            st.caption(f"Source: {title}")
            
            if metadata:
                with st.expander(f"Metadata - Chunk {i}", expanded=False):
                    st.json(metadata)
            
            st.text_area(f"Chunk {i} content", text, height=180, key=f"ingested_chunk_{i}")
            st.divider()


# ==========================================
# CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================================
# USER QUESTION & RAG EXECUTION
# ==========================================

if prompt := st.chat_input("Ask a question about your videos or documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching database and thinking..."):
            try:
                result = answer_query(prompt, model_name=DEFAULT_MODEL)
                answer = result["answer"]
                sources = result.get("sources", [])
                context_texts = result.get("context_texts", [])

                st.session_state.retrieved_chunks = context_texts
                st.session_state.retrieved_sources = sources

                st.markdown("### 🤖 Answer")
                st.markdown(answer)

                st.markdown("### 🔗 Sources")
                if not sources:
                    st.write("No specific source chunks retrieved.")
                else:
                    seen_sources = set()
                    for meta in sources:
                        title = meta.get("title", "Unknown Title")
                        url = meta.get("source", "#")
                        source_key = f"{title}|{url}"
                        if source_key not in seen_sources:
                            st.markdown(f"- [{title}]({url})")
                            seen_sources.add(source_key)

                st.markdown("### 🔎 Retrieved Context")
                st.caption(f"{len(context_texts)} chunks were retrieved and provided to the LLM.")

                if context_texts:
                    for i, context in enumerate(context_texts, start=1):
                        with st.expander(f"📄 Retrieved Chunk {i}", expanded=False):
                            st.write(context)
                            if i <= len(sources) and sources[i - 1]:
                                st.caption("Metadata")
                                st.json(sources[i - 1])
                else:
                    st.warning("The retriever did not return any context chunks.")

                with st.expander("🛠️ RAG Debug Information", expanded=False):
                    st.write("**Question:**")
                    st.code(prompt)
                    st.write("**Model:**")
                    st.code(DEFAULT_MODEL)
                    st.write("**Number of retrieved chunks:**")
                    st.code(str(len(context_texts)))
                    st.write("**Number of source metadata entries:**")
                    st.code(str(len(sources)))

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error while generating answer: {e}")