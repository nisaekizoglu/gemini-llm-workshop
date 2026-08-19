1. LLM Pratikleri
    1. Projenin Oluşturulması ve sanal ortam
        python -m venv venv
        .\venv\Scripts\activate
    2. kütüphane kurulması
        langchain
        langchain-google-genai
        python-dotenv
        pip install -r requirements.txt
    3. gemini api key: 
        .env: GOOGLE_API_KEY=""
    4. İlk gemini çağrısı yapılması
2. RAG
    1. Proje yapısı
        mil_std_document.py
        create_vector_db.py
        rag.py
    2. kütüphane kur
        langchain
        langchain-google-genai
        langchain-community
        langchain-text-splitters
        faiss-cpu
        python-dotenv
        pip install -r requirements.txt
    3. mil-std doküman oluşturma
        https://github.com/turkiyeyapayzekaakademisi/llm-rag-memory-ai-agents

3. Agent