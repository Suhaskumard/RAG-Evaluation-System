 RAG Evaluation and Improvement System

 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** based
question answering system and evaluates its performance using quantitative
metrics. The system focuses on identifying weaknesses in RAG pipelines such as
poor retrieval, hallucinations, and inconsistent answers.

The project is designed as an **evaluation layer on top of RAG**, rather than
just another chatbot.


## 🎯 Problem Statement
Most RAG systems are deployed without proper evaluation. This leads to:
- Irrelevant document retrieval
- Hallucinated answers
- Unstable responses for the same query

This project addresses the problem by **measuring and visualizing RAG quality**.

🧠 System Architecture
1. Document ingestion  
2. Text chunking  
3. Embedding generation  
4. Vector search using FAISS  
5. Reranking retrieved documents  
6. Answer generation  
7. Evaluation using faithfulness and stability metrics  
8. Visualization using Streamlit dashboard  

 📊 Evaluation Metrics

 Faithfulness
Measures how well the generated answer is supported by retrieved documents.
Low scores may indicate hallucinations.

Stability
Measures consistency of answers when the same query is asked multiple times.
Low stability indicates unreliable generation.

 🛠️ Tech Stack
- Python  
- Sentence Transformers  
- FAISS  
- HuggingFace Transformers  
- Streamlit  
- ngrok (for dashboard exposure)  

 📁 Project Structure

RAG-Evaluation-System/
│
├── data/
│   └── docs.txt
│
├── reports/
│   └── failures.jsonl
│
├── main.py
├── app.py
├── requirements.txt
└── README.md

 ▶️ How to Run
 1️⃣ Install Dependencies
pip install -r requirements.txt

 2️⃣ Run Evaluation Pipeline
python main.py

 3️⃣ Launch Dashboard
streamlit run app.py

(Optional) Use ngrok to expose the dashboard publicly.

 📈 Output

* Evaluated answers with faithfulness and stability scores
* Logged failure cases
* Interactive dashboard showing evaluation results

 🎓 Academic Relevance
This project demonstrates:
* Applied Natural Language Processing
* Vector similarity search
* Model evaluation techniques
* System-level design thinking

 🔮 Future Work

* Add Recall@K metric for retrieval evaluation
* Compare multiple embedding models
* Automate chunk size optimization
* Deploy dashboard using Streamlit Cloud

