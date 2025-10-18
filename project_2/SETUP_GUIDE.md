# RAG Chatbot - Complete Setup Guide

This guide provides step-by-step instructions to run the `rag_chatbot.ipynb` notebook successfully.

## Quick Start (For Users with Internet Access)

```bash
# 1. Navigate to project directory
cd project_2

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install and setup Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # Start in background
ollama pull gemma3:1b

# 4. Run the test script to verify setup
python test_notebook_requirements.py

# 5. Launch Jupyter and open the notebook
jupyter notebook rag_chatbot.ipynb
```

## Detailed Setup Instructions

### Step 1: System Requirements

Ensure you have:
- Python 3.11 or higher
- pip package manager
- At least 4GB of available disk space (for models)
- Internet connectivity (for initial setup)

Check your Python version:
```bash
python3 --version
```

### Step 2: Install Python Dependencies

#### Option A: Using pip (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Using conda (Alternative)
```bash
conda env create -f environment.yml
conda activate rag-chatbot
python -m ipykernel install --user --name=rag-chatbot --display-name "rag-chatbot"
```

### Step 3: Install Ollama

Ollama is required to run the local LLM (Gemma3:1b).

#### On Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On macOS:
```bash
brew install ollama
```

#### On Windows:
Download the installer from [ollama.com](https://ollama.com)

#### Verify Installation:
```bash
ollama --version
```

### Step 4: Download the LLM Model

```bash
# Pull the required model (this may take a few minutes)
ollama pull gemma3:1b

# Verify the model is available
ollama list
```

### Step 5: Start Ollama Service

```bash
# Start the Ollama service
ollama serve
```

**Note**: Keep this terminal open, or run in background:
```bash
# Run in background
nohup ollama serve > ollama.log 2>&1 &
```

### Step 6: Verify Setup

Run the test script to ensure everything is configured correctly:

```bash
python test_notebook_requirements.py
```

Expected output:
```
✅ Library Imports
✅ PDF Loading
✅ Text Chunking
✅ Ollama
```

### Step 7: Run the Notebook

#### Option A: Jupyter Notebook (Interactive)
```bash
jupyter notebook rag_chatbot.ipynb
```

Then select the "rag-chatbot" kernel if you used conda, or your default Python kernel.

#### Option B: Run as Python Script
```bash
# Convert notebook to script
jupyter nbconvert --to script rag_chatbot.ipynb

# Run the script
python rag_chatbot.py
```

## Troubleshooting

### Issue: "Cannot connect to HuggingFace"

**Cause**: No internet connectivity or HuggingFace is blocked

**Solution 1** - Use a pre-cached model:
```bash
# On a connected machine, download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/gte-small')"

# Copy the cache directory to your offline machine
# Source: ~/.cache/huggingface/
```

**Solution 2** - Use a different embedding model (requires modifying notebook):
```python
# Replace in notebook:
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

### Issue: "Ollama not found"

**Cause**: Ollama not installed or not in PATH

**Solution**:
```bash
# Check if Ollama is installed
which ollama

# If not found, install it:
curl -fsSL https://ollama.com/install.sh | sh

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin
```

### Issue: "Ollama connection refused"

**Cause**: Ollama service is not running

**Solution**:
```bash
# Start the service
ollama serve

# Or check if it's already running
ps aux | grep ollama
```

### Issue: "Model 'gemma3:1b' not found"

**Cause**: Model not downloaded

**Solution**:
```bash
# Pull the model
ollama pull gemma3:1b

# Verify
ollama list
```

### Issue: Deprecation Warnings

**Cause**: Using old LangChain import paths

**Solution**: Use the fixed notebook:
```bash
# Use the fixed version with updated imports
jupyter notebook rag_chatbot_fixed.ipynb
```

### Issue: PDF Loading Errors

**Cause**: PDF files not found or corrupted

**Solution**:
```bash
# Check if PDF files exist
ls -l data/Everstorm_*.pdf

# Should show 4 PDF files
```

## Running in Offline Mode

If you need to run the notebook without internet:

### 1. Pre-cache Models (on connected machine)

```bash
# Download HuggingFace model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/gte-small')"

# Pull Ollama model
ollama pull gemma3:1b
```

### 2. Transfer Cached Data

Copy these directories to the offline machine:
- HuggingFace cache: `~/.cache/huggingface/`
- Ollama models: `~/.ollama/models/`

### 3. Run Offline

```bash
# Start Ollama (doesn't need internet once model is downloaded)
ollama serve

# Run the notebook
jupyter notebook rag_chatbot.ipynb
```

## Testing the Streamlit App

After running all notebook cells:

```bash
# The notebook generates app.py
# Run it with Streamlit
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with the chatbot.

## Notebook Structure

The notebook is organized into 6 main sections:

1. **Environment Setup** (Cells 1-6)
   - Import libraries
   - Verify installation

2. **Data Preparation** (Cells 7-12)
   - Load PDF files
   - Optional: Load web pages
   - Chunk documents

3. **Build Retriever** (Cells 13-17)
   - Create embeddings
   - Build FAISS vector index

4. **Build Generation Engine** (Cells 18-20)
   - Initialize Ollama
   - Test LLM

5. **Build RAG Chain** (Cells 21-28)
   - Create prompt template
   - Connect components
   - Test with questions

6. **Streamlit UI** (Cells 29-35)
   - Generate web app
   - Deploy locally

## Expected Runtime

- First run (with downloads): 10-15 minutes
- Subsequent runs: 2-3 minutes
- Per-cell execution: 5-30 seconds each

## Files Generated During Execution

- `faiss_index/` - Vector store index (persisted)
- `app.py` - Streamlit web application
- `rag_chatbot.py` - Converted script (if using nbconvert)

## Tips for Success

1. **Run cells sequentially** - Each cell depends on previous ones
2. **Wait for downloads** - First run takes longer due to model downloads
3. **Keep Ollama running** - The service must be active for LLM calls
4. **Check logs** - If something fails, check `ollama.log`
5. **Use the test script** - Run `test_notebook_requirements.py` before starting

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Support

If you encounter issues:

1. Run the test script: `python test_notebook_requirements.py`
2. Check the TESTING_REPORT.md for known issues
3. Review the troubleshooting section above
4. Ensure all prerequisites are met

## Next Steps

After successfully running the notebook:

1. Experiment with different questions
2. Try adding your own PDF documents
3. Modify the prompt template for different behaviors
4. Customize the Streamlit UI
5. Deploy the app to a server
