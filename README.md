# 🦠 Gut Health Assistant (RAG-based Chatbot)

A **Retrieval-Augmented Generation (RAG)** powered conversational assistant that provides **accurate, supportive, and evidence-based answers** to gut health–related queries.  
It uses **FAISS vector search**, **HuggingFace embeddings**, and **NVIDIA’s Llama 3–8B model** for natural, friendly, and empathetic responses.

---

## 🚀 Features

- **Streamlit Chatbot UI** (`app.py`)  
  Clean, responsive chat interface with source display, system status, and quick-start suggestions.

- **RAG Pipeline** (`gut_rag.py`)  
  Combines FAISS retrieval with NVIDIA’s LLM API to generate contextual and empathetic answers.

- **Dataset Indexing** (`build_index.py`)  
  Builds a FAISS vector index from gut health datasets (`train.jsonl`, `val.jsonl`, `test.jsonl`).

- **Evaluation Framework** (`evaluate.py`)  
  Automated evaluation of:
  - Relevance of answers  
  - Source inclusion  
  - Tone (supportive, empathetic, friendly)  
  - Safety against harmful responses  

- **Optimized Performance**  
  - Caching for embeddings and results  
  - Async pipeline loading  
  - Efficient session state management in Streamlit  

---

## 📂 Project Structure

```
├── app.py              # Streamlit chatbot app
├── build_index.py      # Build FAISS index from dataset
├── gut_rag.py          # RAG pipeline implementation
├── evaluate.py         # Evaluation script
├── raw/
│   └── dataset/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       ├── tone_examples.jsonl
│       └── negatives.jsonl
├── faiss_index/        # Generated FAISS index
└── requirements.txt    # Dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/gut-health-assistant.git
cd gut-health-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

---

## 🏗️ Usage

### Step 1: Build FAISS Index
Make sure dataset files (`train.jsonl`, `val.jsonl`, `test.jsonl`) are in `raw/dataset/`.

```bash
python build_index.py
```

This creates a `faiss_index/` folder with the embeddings.

### Step 2: Run Chatbot App
```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

### Step 3: Evaluate System
Run automated evaluation on test and negative datasets:

```bash
python evaluate.py
```

Results are saved in `eval_results.jsonl`.

---

## 📊 Evaluation Metrics

- **Relevance** → Does the answer match the expected content?  
- **Source Inclusion** → Are references or links included?  
- **Tone Appropriateness** → Is the response supportive, empathetic, and friendly?  
- **Safety** → Avoids generating harmful or misleading medical advice.  

---

## 📌 Example Questions

- "What foods promote a healthy gut microbiome?"  
- "How do antibiotics affect gut bacteria?"  
- "What's the difference between probiotics and prebiotics?"  
- "Can stress affect digestive health?"  

---

## 🛡️ Disclaimer

This assistant is for **educational purposes only**.  
It does **not** provide professional medical advice. Always consult a healthcare professional for medical concerns.

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share with attribution.

---

## ✨ Contributors

- **Lokesh Lohar** – AI & ML Developer  
- OpenAI, HuggingFace, NVIDIA APIs – Model providers  
