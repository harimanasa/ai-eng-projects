#!/usr/bin/env python
# coding: utf-8

# # Project 2: Customer-Support Chatbot for an E-Commerce Store

# Welcome! In this project, you'll build a **chatbot** that answers customer service questions about Everstorm Outfitters, an imaginary e-commerce store.
# 
# Run each cell in order. Feel free to modify them as you go to better understand each tool and search the web or look online for documentation.

# ## Learning Objectives  
# * **Ingest & chunk** unstructured docs  
# * **Embed** chunks and **index** with *FAISS*  
# * **Retrieve** context and **craft prompts**  
# * **Run** an open-weight LLM locally with *Ollama*  
# * **Build** a Retrieval-Augmented Generation (RAG) chain
# * **Package** the chat loop in a minimal **Streamlit** web UI

# ## Roadmap  
# We will build a RAG-based chatbot in **six** steps:
# 
# 1. **Environment setup**
# 2. **Data preparation**  
#    a. Load source documents  
#    b. Chunk the text  
# 3. **Build a retriever**  
#    a. Generate embeddings  
#    b. Build a FAISS vector index  
# 4. **Build a generation engine**. Load the *Gemma3-1B* model through Ollama and run a sanity check.  
# 5. **Build a RAG**. Connect the system prompt, retriever, and LLM together. 
# 6. **(Optional) Streamlit UI**. Wrap everything in a simple web app so users can chat with the bot.
# 

# ## 1 - Environment setup
# 
# We use conda to manage our project dependencies and ensure everyone has a consistent setup. Conda is an open-source package and environment manager that makes it easy to install libraries and switch between isolated environments. To learn more about conda, you can read: https://docs.conda.io/en/latest/
# 
# Create and activate a clean *conda* environment and install the required packages. If you don't have conda installed, visit https://www.anaconda.com/docs/getting-started/miniconda/main.
# 
# 
# Open your terminal, navigate to the project folder where this notebook is located, and run the following commands.
# 
# ```bash
# conda env create -f environment.yml && conda activate rag-chatbot
# 
# # (Optional but recommended) Register this environment as a Jupyter kernel
# python -m ipykernel install --user --name=rag-chatbot --display-name "rag-chatbot"
# ```
# Once this is done, you can select “rag-chatbot” from the Kernel → Change Kernel menu in Jupyter or VS Code.
# 
# 
# > Behind the scenes:
# > * Conda reads `environment.yml`, solves all pinned dependencies, and builds an isolated environment named `rag-chatbot`.
# > * When it reaches the file’s "pip:" section, Conda automatically invokes pip to install any remaining Python-only packages so the whole stack be available for the project.
# > * Registering the kernel makes your new environment visible to Jupyter, so the notebook runs inside the same environment you just created.

# Let's import required libraries and print a message if we're not **missing packages**.

# In[ ]:


# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

print("✅ Libraries imported! You're good to go!")


# ## 2 - Data preparation
# The goal of this step is to turn all reference documents into small chunks of text that a retriever can index and search. These documents typically come from:
# * PDF files: local documents such as policies, user manuals, or guides.
# * Web pages (HTML): online documentation, blog posts, or help articles.
# 
# In this step, we perform two actions:
# * **Ingesting**: load every PDF and collect the raw text in a list named `raw_docs`.
# * **Chunking**: split each document into small, overlapping chunks so later steps can match a user query to the most relevant passage.

# ### 2.1 - Ingest source documents

# We can use different libraries to load and process PDFs. A quick web search will show several options, each with its own strengths. In this case, we’ll use PyPDFLoader from LangChain, which makes it easy to extract text from PDF files for downstream processing. To learn more about how to use it, refer to: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
# 
# Use **PyPDFLoader** to load every PDF whose filename matches `data/Everstorm_*.pdf` and collect all pages in a list called `raw_docs`. The content of these PDFs is synthetically generated for educational purposes.

# In[ ]:


pdf_paths = glob.glob("data/Everstorm_*.pdf")
raw_docs = []

# Load all pages from each matching PDF into raw_docs
for p in pdf_paths:
    loader = PyPDFLoader(p)
    raw_docs.extend(loader.load())

print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")


# ### (Optional) 2.1 - Load web pages
# You can also pull content straight from the web. Various libraries support reading and parsing web pages directly into text, which is useful for building custom knowledge bases. One example is **UnstructuredURLLoader** from LangChain, which can extract readable content from raw HTML pages and return them in a structured format. To learn more, see: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.url.UnstructuredURLLoader.html
# 
# To practice, load each HTML page below and store the results in a list called `raw_docs`. We’ve included a few sample URLs, but you can replace them with any links you prefer.
# 
# For robustness, add an offline fallback in case a URL fails. In real projects, we typically cache fetched pages to disk, handle rate limits, and track fetch timestamps so content can be refreshed periodically without relying on live network calls during development. For this project, we don’t have offline HTML copies available, but you can still practice by loading any PDFs from the data/ folder using PyPDFLoader and appending them to raw_docs.

# In[ ]:


URLS = [
    # --- BigCommerce – shipping & refunds ---
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds"
]

try:
    url_loader = UnstructuredURLLoader(urls=URLS)
    web_docs = url_loader.load()
    raw_docs.extend(web_docs)
    print(f"Fetched {len(web_docs)} documents from the web. Total docs now: {len(raw_docs)}")
except Exception as e:
    print("⚠️  Web fetch failed, using offline copies:", e)
    # As a fallback, just reload PDFs so the pipeline still works
    raw_docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        raw_docs.extend(loader.load())
    print(f"Loaded {len(raw_docs)} offline documents.")


# ### 2.2 - Chunk the text

# Long documents won’t work well directly with most LLMs. They can easily exceed the model’s context window, making it impossible for the model to read or reason over the full text at once. Even if they fit, processing long inputs can be inefficient and lead to weaker retrieval results.
# 
# To handle this, we split large documents into smaller, overlapping chunks. Several libraries can help with text splitting, each designed to preserve structure or balance chunk size. A popular choice is `RecursiveCharacterTextSplitter` from LangChain, which splits text intelligently while keeping paragraph or sentence boundaries intact. To familiarize youself with the library, visit: https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
# 
# In this project, we’ll split each document into chunks of roughly 300 tokens with a 30-token overlap using `RecursiveCharacterTextSplitter`. (Approximate this as ~1200 characters with ~120-char overlap.)

# In[ ]:


chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
chunks = text_splitter.split_documents(raw_docs)
print(f"✅ {len(chunks)} chunks ready for embedding")


# ## 3 -Build a retriever
# 
# A *retriever* lets the RAG pipeline efficiently look up small, relevant pieces of context at query-time. This step has two parts:
# 1. **Load a model to generate embeddings**: convert each text chunk from the reference documents into a fixed-length vector that captures its semantic meaning.  
# 2. **Build vector database**: store these embeddings in a vector database.
# 

# ### 3.1 - Load a model to generate embeddings

# The goal of this step is to convert each document chunk into a numerical vector (an embedding) that captures its semantic meaning. These embeddings allow our retriever to find and compare similar pieces of text efficiently.
# 
# There are models trained specifically for this purpose, called embedding models. One popular example is OpenAI’s `text-embedding-3-small`, which produces high-quality embeddings that work well for retrieval and semantic search.
# 
# If you prefer running everything locally, you can use smaller open-source models such as `gte-small` (77 M parameters). These local models load quickly, don’t require internet access, and are ideal for experimentation or environments without API access. However, they’re typically less powerful than hosted models.
# 
# Alternatively, you can connect to an API service to access stronger models like OpenAI’s. These require setting an API key (for example, OPENAI_API_KEY) in your environment. OpenAI allows you to create a free account and sometimes offers limited trial credits for new users, but ongoing access requires a billing setup. 
# 
# In this exercise, we’ll stick to the smaller gte-small model for simplicity and reproducibility. We'll use our imported `SentenceTransformerEmbeddings` library to load the model and use it to embed queries. To learn more about lagnchain's embedding support, visit: https://python.langchain.com/docs/integrations/text_embedding/

# In[ ]:


embedding_vector = []

# Embed the sentence "Hello world!" and store it in embedding_vector.
embedder = SentenceTransformerEmbeddings(model_name="sentence-transformers/gte-small")
embedding_vector = embedder.embed_query("Hello world!")
print(len(embedding_vector))


# ### 3.2 - Build a vector database
# 
# Once we have embeddings, we need a way to store and search them efficiently. A simple list wouldn’t scale well, especially when we have thousands of chunks and need to quickly find the most relevant ones.
# 
# To solve this, we use **FAISS**, an open-source similarity search library developed by Meta. FAISS is optimized for fast nearest-neighbor search in high-dimensional spaces, making it ideal for tasks like semantic retrieval and recommendation. It’s strongly encouraged to visit their quickstart guide to understand how FAISS works and how to use it effectively: https://github.com/facebookresearch/faiss/wiki/getting-started
# 
# In this step, we’ll feed all our document embeddings into FAISS, which builds an in-memory vector index. This index allows us to efficiently query for the *k* most similar chunks to any given question.
# 
# During inference, we’ll use this index to retrieve the top-k relevant chunks and pass them to the LLM as context, enabling it to answer questions grounded in our documents.
# 
# 

# In[ ]:


# 1) Build the FAISS index from chunks
embedding_fn = SentenceTransformerEmbeddings(model_name="sentence-transformers/gte-small")
vectordb = FAISS.from_documents(chunks, embedding_fn)

# 2) Create a retriever (k=8)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# 3) Save the vector store locally
vectordb.save_local("faiss_index")

# 4) Confirmation
print("✅ Vector store with", vectordb.index.ntotal, "embeddings")


# ## 4 - Build the generation engine
# At the core of any RAG system lies an **LLM**. The retriever finds relevant information, and the LLM uses that information to generate coherent, context-aware responses.
# 
# In this project, we’ll use **Gemma 3* (1B), a small but capable open-weight model, and run it entirely on your local machine using Ollama. This means you won’t need API keys or internet access to generate responses once the model is downloaded.
# 
# **What is Ollama?**
# 
# Ollama is a lightweight runtime for managing and serving open-weight LLMs locally. It provides:
# * A simple REST API running at localhost:11434, so your code can interact with the model via standard HTTP calls.
# * A model registry and command-line tool** to pull, run, and manage models easily.
# * Support for a wide variety of models (e.g., Gemma, Llama, Mistral, Phi, etc.), making it ideal for experimentation.
# 
# To learn more about Ollama, visit: https://github.com/ollama/ollama. You can browse all supported models and their sizes here: https://ollama.com/library
# 
# 
# ### 4.1 - Install `ollama` and serve `gemma3`
# 
# Follow these steps to set up Ollama and start the model server:
# 
# **1 - Install**
# ```bash
# # macOS (Homebrew)
# brew install ollama
# # Linux
# curl -fsSL https://ollama.com/install.sh | sh
# ```
# 
# If you’re on Windows, install using the official installer from https://ollama.com/download.
# 
# **2 - Start the Ollama server (keep this terminal open)**
# ```bash
# ollama serve
# ```
# This command launches a local server at http://localhost:11434, which will stay running in the background.
# 
# 
# **3 - Pull the Gemma mode (or the model of your choice) in a new terminal**
# ```bash
# ollama pull gemma3:1b
# ```
# 
# This downloads the 1B version of Gemma 3, a compact model suitable for running on most modern laptops. Once downloaded, Ollama will automatically handle model loading and caching.
# 
# 
# After this setup, your system is ready to generate responses locally using the Gemma model through the Ollama API.
# 

# ### 4.2 - Test an LLM with a random prompt (Sanity check)
# 

# In[ ]:


# Initialize the local LLM and run a quick test
llm = Ollama(model="gemma3:1b", temperature=0.1)
print(llm.invoke("In one sentence, explain what a RAG chatbot does."))


# ## Build a RAG

# ### 5.1 - Define a system prompt

# At this stage, we need to tell the model how to behave when generating answers. The **system prompt** acts as the model’s rulebook. It should clearly instruct the model to answer only using the retrieved context and to admit when it doesn’t know the answer. This helps prevent hallucination and keeps the responses grounded in the provided documents.
# 
# In general, a good RAG prompt emphasizes three things: stay within context, stay factual, and stay concise. This is important because RAG works by grounding the LLM in retrieved text. If the prompt is vague, the model may invent details. A precise system prompt reduces hallucinations and keeps answers aligned with your corpus.

# In[ ]:


SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: {source}] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""


# ### 5.2 Create a RAG chain
# Now that we have a retriever, a prompt, and a language model, we can connect them into a single RAG pipeline. The retriever finds the most relevant chunks from our vector index, the prompt injects those chunks into the system message, and the LLM uses that context to produce the final answer. (retriever → prompt → model)
# 
# This connection is handled through LangChain’s `ConversationalRetrievalChain`, which combines retrieval and generation. To familiarize yourself with the library, visit: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html

# In[ ]:


# Build the Conversational Retrieval Chain with our custom prompt
prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
llm = Ollama(model="gemma3:1b", temperature=0.1)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)
print("✅ RAG chain ready!")


# When you ask a question, the retriever pulls the top few relevant text chunks, the model reads them through the system prompt, and then it generates an answer based on that context.
# 
# This structure makes the system transparent and easy to debug. You can inspect what text was retrieved, tune parameters like k, and experiment with different prompts to see how they affect the output quality.
# 

# ### 5.3 - Validate the RAG chain

# We run a few questions to make sure everything behaves as expecte. Experiment by adding you own questions.

# In[ ]:


test_questions = [
    "If I'm not happy with my purchase, what is your refund policy and how do I start a return?",
    "How long will delivery take for a standard order, and where can I track my package once it ships?",
    "What's the quickest way to contact your support team, and what are your operating hours?",
]

chat_history = []
for q in test_questions:
    result = chain.invoke({"question": q, "chat_history": chat_history})
    answer = result.get("answer", result)
    print("\nQ:", q)
    print("A:", answer)
    chat_history.append((q, answer))


# ### 6 - Build the Streamlit UI (optional)

# The goal here is to create a tiny demo so you can interact with your RAG system. The focus is not on UI design. We will build a very small interface only to demonstrate the end-to-end flow.
# 
# There are many ways to make a UI. Some frameworks are powerful but take longer to set up, while others are simple and good for quick experiments. Streamlit is a common choice for fast prototyping because it lets you make a usable interface with only a few lines of Python. If you want to learn the basics, see the Streamlit Quickstart: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/quickstart
# 
# This step is optional. If it is not useful for your work, you can skip it. We will also complete this part together during the live session.
# 
# In this cell, we write a minimal **`app.py`** that starts a simple chat UI and calls your RAG chain.

# In[ ]:


app_code = r'''import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: {source}] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

st.set_page_config(page_title="Everstorm Support Chatbot", page_icon="🛍️")
st.title("🛍️ Everstorm Outfitters — Support Chatbot")

@st.cache_resource(show_spinner=False)
def load_chain():
    embedder = SentenceTransformerEmbeddings(model_name="sentence-transformers/gte-small")
    vectordb = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    llm = Ollama(model="gemma3:1b", temperature=0.1)
    prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

chain = load_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_q = st.chat_input("Ask about refunds, shipping, or contacting support…")

for q, a in st.session_state.chat_history:
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"): st.write(a)

if user_q:
    with st.chat_message("user"): st.write(user_q)
    result = chain.invoke({"question": user_q, "chat_history": st.session_state.chat_history})
    answer = result.get("answer", "(No answer)")
    st.session_state.chat_history.append((user_q, answer))
    with st.chat_message("assistant"): st.write(answer)
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)
print("✅ Wrote app.py. Run: streamlit run app.py")


# Run `streamlit run app.py` from your terminal.

# ## 🎉 Congratulations!
# 
# You’ve just built, tested, and demoed a fully working **customer-support chatbot**.  
# In one project you:
# 
# * **Prepared policy docs**: loaded and chunked them for fast retrieval.  
# * **Built a vector store**: created a FAISS index with text embeddings.  
# * **Plugged in an LLM**: wrapped Gemma3 with LangChain and a prompt-aware RAG chain.  
# * **Validated end-to-end**: answered refund, shipping, and contact questions with confidence.  
# 
# Swap in new documents, tweak the prompt, and your store’s customers get instant, accurate answers.
# 
# 👏 **Great job!** Take a moment to celebrate. The skills you used here power most RAG-based chatbots you see everywhere.
# 

# 
