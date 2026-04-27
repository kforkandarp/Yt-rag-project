from dotenv import load_dotenv
load_dotenv()  # loads the .env file so os.environ.get() can find the key

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFacePipeline     # ✅ updated (fix deprecation, was langchain_community)
from langchain_community.vectorstores import FAISS
# from transformers import pipeline, AutoTokenizer          # ✅ added AutoTokenizer for chat template
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os                                                 # ✅ added to read cookie file path from environment
import requests                                           # ✅ added to build a cookie-loaded HTTP session
import http.cookiejar                                     # ✅ added to parse Netscape-format cookies.txt file
import re

# ============================================================
# PAGE CONFIG
# Must be the very first Streamlit call in the script
# ============================================================
st.set_page_config(
    page_title="VidChat — Chat with YouTube",
    page_icon="🎬",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# Injects styling directly into the Streamlit page
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Overall background */
.stApp { background-color: #0f0f0f; color: #f0f0f0; font-family: 'DM Sans', sans-serif; }

/* Title */
h1 { font-family: 'Space Mono', monospace !important; color: #f0f0f0 !important; letter-spacing: -1px; }

/* Input boxes */
.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}

/* Chat message bubbles — user */
.user-bubble {
    background: #1e3a5f;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
}

/* Chat message bubbles — assistant */
.bot-bubble {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.95rem;
}

/* Source chunk expander */
.source-box {
    background: #111;
    border-left: 3px solid #e63946;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-green { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
.badge-red   { background: #3a1a1a; color: #f87171; border: 1px solid #991b1b; }
.badge-blue  { background: #1a2a3a; color: #60a5fa; border: 1px solid #1d4ed8; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER: extract video ID from a full YouTube URL or raw ID
# Handles formats like: youtu.be/ID, ?v=ID, /embed/ID
# ============================================================
def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",       # ?v=ID
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})", # youtu.be/ID
        r"(?:embed/)([a-zA-Z0-9_-]{11})",     # /embed/ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # if nothing matched, assume the raw string is already a video ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    return None


# ============================================================
# HELPER: build a requests.Session loaded with cookies.txt
# ✅ the new youtube-transcript-api version removed cookie_path;
# cookies must be passed via http_client instead
# ============================================================
def build_session_with_cookies(cookie_file: str) -> requests.Session:
    session = requests.Session()
    cookiejar = http.cookiejar.MozillaCookieJar(cookie_file)
    # MozillaCookieJar reads Netscape format exported by browser extensions
    cookiejar.load(ignore_discard=True, ignore_expires=True)
    session.cookies = cookiejar
    return session


# ============================================================
# CACHED: load the LLM once and reuse across reruns
# st.cache_resource keeps heavy objects (models) alive in memory
# so Streamlit doesn't reload them on every interaction
# ============================================================
# ------------------------------------------------------------------------------------------------------------------------
# @st.cache_resource(show_spinner="⚙️ Loading model (first time only, ~600MB)...")
# def load_llm():
#     model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # ✅ load tokenizer separately so we can apply the chat template
#     pipe = pipeline(
#         "text-generation",
#         model=model_name,
#         tokenizer=tokenizer,   # ✅ pass tokenizer explicitly
#         max_new_tokens=256     # limits how many tokens the model can generate
#     )
#     return HuggingFacePipeline(pipeline=pipe)
#     # wraps HuggingFace pipeline into LangChain-compatible LLM interface
# ------------------------------------------------------------------------------------------------------------------------

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2   # low temperature = more factual, less creative/random
    )

# ============================================================
# CACHED: fetch + embed a video transcript
# keyed by video_id so switching videos re-runs this automatically
# ============================================================
@st.cache_resource(show_spinner=False)
def load_video(_llm, video_id: str):
    """
    Fetches transcript, chunks it, embeds it, and builds retriever.
    Returns (retriever, transcript_preview) or raises on failure.
    _llm is passed only to tie cache lifetime to the model instance.
    """
    COOKIES_PATH = os.environ.get("YOUTUBE_COOKIES_PATH", "cookies.txt")

    # ----------- FETCH TRANSCRIPT -----------
    try:
        if os.path.exists(COOKIES_PATH):
            # ✅ build authenticated session so YouTube doesn't block the request
            http_client = build_session_with_cookies(COOKIES_PATH)
            fetched = YouTubeTranscriptApi(http_client=http_client).fetch(video_id, languages=['en'])
        else:
            # fallback: try without cookies (works on non-blocked IPs)
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        # Fetch transcript object

        transcript_list = fetched.to_raw_data()
        # Convert transcript into list of dictionaries

        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        # Flatten all chunks into one continuous text string

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No English transcript found for this video.")
    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {e}")

    # ----------- CHUNKING -----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    # Creates overlapping chunks to preserve context
    documents = splitter.create_documents([transcript])
    # Converts text into LangChain Document objects

    # ----------- EMBEDDINGS + VECTOR STORE -----------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Converts text chunks into numerical vectors (semantic meaning)

    vector_store = FAISS.from_documents(documents, embeddings)
    # Stores vectors in FAISS for efficient similarity search

    # ----------- RETRIEVER -----------
    retriever = vector_store.as_retriever(
        search_type="similarity",   # uses cosine similarity between embeddings
        search_kwargs={"k": 4}      # return top 4 relevant chunks
    )

    return retriever, transcript[:400]  # also return a short preview of the transcript


# ============================================================
# PROMPT TEMPLATE
# ============================================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    # variables that will be dynamically filled
    template="""You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

)



# ============================================================
# RAG PIPELINE
# ============================================================
def rag_pipeline(query: str, retriever, llm) -> tuple[str, list]:
    docs = retriever.invoke(query)
    # retrieves top-k relevant chunks using semantic search

    context = "\n\n".join(doc.page_content for doc in docs)
    # merges multiple chunks into one context string for the LLM

    if not context.strip():
        return "I don't know.", []
    # ✅ gracefully handle empty retrieval before calling the LLM

    final_prompt = prompt.format(context=context, question=query)
    # injects retrieved context + user query into prompt template

    response = llm.invoke(final_prompt)
    # LLM generates answer based only on provided context

    # ✅ extract only the generated answer, stripping the echoed prompt
    return response.content, docs


# ============================================================
# SESSION STATE INIT
# st.session_state persists values across Streamlit reruns
# (Streamlit reruns the whole script on every interaction)
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # list of {"role": "user"/"assistant", "content": "...", "sources": [...]}

if "loaded_video_id" not in st.session_state:
    st.session_state.loaded_video_id = None
    # tracks which video is currently loaded so we can detect changes

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ============================================================
# UI — HEADER
# ============================================================
st.markdown("# 🎬 VidChat")
st.markdown("<p style='color:#666; margin-top:-12px; font-family:Space Mono,monospace; font-size:0.8rem;'>chat with any youtube video</p>", unsafe_allow_html=True)
st.divider()


# ============================================================
# UI — VIDEO URL INPUT
# ============================================================
col1, col2 = st.columns([4, 1])
with col1:
    url_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
with col2:
    load_btn = st.button("Load", use_container_width=True, type="primary")


# ============================================================
# LOAD VIDEO on button click
# ============================================================
if load_btn and url_input:
    video_id = extract_video_id(url_input)

    if not video_id:
        # couldn't parse a valid ID from the input
        st.markdown('<span class="badge badge-red">❌ Invalid URL</span>', unsafe_allow_html=True)
    elif video_id == st.session_state.loaded_video_id:
        # same video already loaded, no need to re-embed
        st.markdown('<span class="badge badge-blue">ℹ️ Already loaded</span>', unsafe_allow_html=True)
    else:
        # new video — fetch, embed, build retriever
        with st.spinner("Fetching transcript and building knowledge base..."):
            try:
                llm = load_llm()
                retriever, preview = load_video(llm, video_id)

                # save to session so chat can use them
                st.session_state.retriever = retriever
                st.session_state.loaded_video_id = video_id
                st.session_state.chat_history = []  # clear old chat on new video

                st.markdown(f'<span class="badge badge-green">✅ Video loaded — {video_id}</span>', unsafe_allow_html=True)
                # show a short transcript preview so user knows it worked
                with st.expander("📄 Transcript preview"):
                    st.caption(preview + "...")

            except ValueError as e:
                st.markdown(f'<span class="badge badge-red">❌ {e}</span>', unsafe_allow_html=True)


# ============================================================
# UI — CHAT AREA
# Only shown once a video is loaded
# ============================================================
if st.session_state.retriever:

    st.divider()

    # render all past messages from history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            # show source chunks the answer was built from
            if msg.get("sources"):
                with st.expander("📎 Sources used", expanded=False):
                    for i, doc in enumerate(msg["sources"]):
                        st.markdown(f'<div class="source-box"><b>Chunk {i+1}</b><br>{doc.page_content[:300]}...</div>', unsafe_allow_html=True)

    st.divider()

    # chat input at the bottom
    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        # immediately show the user's message
        st.markdown(f'<div class="user-bubble">🧑 {user_input}</div>', unsafe_allow_html=True)

        # run RAG and show a spinner while the model thinks
        with st.spinner("Thinking..."):
            llm = load_llm()
            answer, sources = rag_pipeline(user_input, st.session_state.retriever, llm)

        # show the answer
        st.markdown(f'<div class="bot-bubble">🤖 {answer}</div>', unsafe_allow_html=True)

        # show sources
        if sources:
            with st.expander("📎 Sources used", expanded=False):
                for i, doc in enumerate(sources):
                    st.markdown(f'<div class="source-box"><b>Chunk {i+1}</b><br>{doc.page_content[:300]}...</div>', unsafe_allow_html=True)

        # save to history so messages persist across reruns
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

else:
    # placeholder shown before any video is loaded
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #444;'>
        <div style='font-size: 3rem;'>🎬</div>
        <p style='font-family: Space Mono, monospace; font-size: 0.85rem; margin-top: 12px;'>
            paste a youtube url above and hit load<br>then chat with the video
        </p>
    </div>
    """, unsafe_allow_html=True)