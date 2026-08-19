# 🤖 Gemini LLM Workshop

> **Gemini, RAG ve AI Agent'ları sıfırdan keşfetmek için uygulamalı bir LLM workshop projesi. Roketsan Coderspace ortaklığı ile geliştirildi.**

Bu repository, **Google Gemini** ve **LangChain / LangGraph** ekosistemini kullanarak modern LLM uygulamalarının temel yapı taşlarını adım adım göstermeyi amaçlar.

Workshop boyunca basit bir Gemini API çağrısından başlayarak:

**LLM → Prompt Engineering → RAG → Vector Search → Tool Calling → AI Agent**

akışını gerçek bir teknik doküman üzerinden uyguluyoruz.

Projenin örnek senaryosunda **RKT-MIL-STD-001** teknik dokümanı bilgi kaynağı olarak kullanılıyor. Agent, kullanıcının sorusunu analiz ederek genel sorular için Gemini'yi, teknik dokümanla ilgili sorular için ise RAG tabanlı arama sistemini kullanıyor.

---

## 🎯 Workshop'un Amacı

Bu projenin amacı yalnızca Gemini API kullanmayı göstermek değil; modern LLM uygulamalarının nasıl **modüler ve agentic sistemlere dönüştürülebileceğini** uygulamalı olarak anlatmaktır.

Workshop sonunda aşağıdaki kavramları daha iyi anlayabilirsiniz:

* 🧠 Large Language Models
* ✍️ Prompt Engineering
* 🔗 LangChain
* 📚 Retrieval-Augmented Generation (RAG)
* 🔎 Semantic Search
* 🧮 Embeddings
* 🗂️ FAISS Vector Database
* 🛠️ Tool Calling
* 🤖 AI Agents
* 🧩 LangGraph Memory / Checkpointing
* 🔐 Environment Variables ve API Key yönetimi

---

# 📚 Workshop İçeriği

## 1. 🧠 İlk Gemini Çağrısı

İlk aşamada Gemini modeline doğrudan istek göndererek temel LLM kullanımını öğreniyoruz.

Bu bölümde:

* Gemini modelinin çağrılması
* API key yönetimi
* Environment variables
* LangChain üzerinden Gemini kullanımı
* Prompt oluşturma
* Model response'larının alınması

ele alınıyor.

Temel örnek:

```python
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
```

Ardından bir prompt hazırlanarak model çağrılıyor.

Bu aşama, sonraki RAG ve Agent örneklerinin temelini oluşturuyor.

---

# 2. 📚 RAG — Retrieval-Augmented Generation

LLM'ler her zaman kendi başlarına özel kurum veya doküman bilgilerine erişemez.

RAG yaklaşımı bu problemi iki aşamada çözer:

```text
Question
   │
   ▼
Retrieve relevant information
   │
   ▼
Add information to prompt
   │
   ▼
LLM generates answer
```

Bu projede RAG sistemi **RKT-MIL-STD-001** teknik dokümanı üzerinde kurulmuştur.

### RAG Pipeline

```text
RKT-MIL-STD-001
       │
       ▼
Document
       │
       ▼
Text Splitting
       │
       ▼
Chunks
       │
       ▼
Gemini Embeddings
       │
       ▼
FAISS Vector Store
       │
       ▼
Similarity Search
       │
       ▼
Relevant Context
       │
       ▼
Gemini
       │
       ▼
Answer
```

Doküman, `RecursiveCharacterTextSplitter` ile parçalara ayrılıyor.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
```

Daha sonra Gemini embedding modeli kullanılarak vektörlere dönüştürülüyor:

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
```

Ve FAISS üzerinde saklanıyor:

```python
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

Sonuç olarak `faiss_index/` altında lokal bir vector store oluşturuluyor.

---

# 3. 🔎 Semantic Search

RAG sisteminin önemli parçalarından biri doğru doküman parçalarını bulabilmektir.

Bu projede FAISS retriever kullanılıyor:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

Kullanıcının sorusu embedding'e dönüştürülerek dokümandaki en alakalı parçalar aranıyor.

Örneğin:

```text
Soru:
"Elektronik kontrol biriminin maksimum çalışma sıcaklığı kaç derecedir?"
```

Retriever ilgili doküman parçalarını bulur.

Bu parçalar daha sonra Gemini'ye context olarak gönderilir.

---

# 4. 🛠️ Tool Calling

Bir sonraki aşamada sistemi biraz daha akıllı hale getiriyoruz.

Agent'a iki farklı tool tanımlıyoruz:

### `general_question_tool`

Genel sorular için kullanılır.

Örnek:

```text
"RAG nedir?"
"Python'da decorator nedir?"
"Bugün hava nasıl?"
```

### `mil_std_rag_tool`

`RKT-MIL-STD-001` teknik dokümanıyla ilgili sorular için kullanılır.

Örnek:

```text
"Çalışma sıcaklığı nedir?"
"Depolama sıcaklığı kaç derece?"
"Nem gereksinimleri nelerdir?"
"Titreşim test kriterleri nelerdir?"
```

Tool'ların tanımları `tools.py` içerisinde bulunuyor.

---

# 5. 🤖 AI Agent

Workshop'un en önemli bölümlerinden biri Agent mimarisidir.

Agent, kullanıcının sorusunu analiz ederek hangi tool'un kullanılacağına karar verir.

```text
                 User Question
                       │
                       ▼
                ┌─────────────┐
                │    Agent    │
                └──────┬──────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        General Question     MIL-STD Question
              │                 │
              ▼                 ▼
        Gemini Tool          RAG Tool
              │                 │
              └────────┬────────┘
                       ▼
                    Answer
```

Agent'ın sistem prompt'u, teknik doküman sorularında RAG tool'unu; diğer sorularda genel Gemini tool'unu kullanmasını sağlar.

Bu yapı sayesinde uygulama yalnızca bir chatbot olmaktan çıkıp **hangi bilgi kaynağını kullanacağına karar verebilen bir AI Agent** haline gelir.

---

# 🧠 Conversation Memory

Agent tarafında LangGraph'ın `InMemorySaver` yapısı kullanılarak konuşma geçmişi tutuluyor.

```python
memory = InMemorySaver()
```

Agent'a bir `thread_id` verilerek aynı konuşma içerisinde önceki mesajların kullanılabilmesi sağlanıyor.

Bu sayede Agent, önceki konuşmalarda verilen bilgileri takip ederek daha bağlamsal cevaplar üretebiliyor.

---

# 📂 Proje Yapısı

```text
gemini-llm-workshop/
│
├── agent.py
├── main.py
├── tools.py
├── rag.py
├── create_vector_db.py
├── mil_std_document.py
├── requirements.txt
│
├── faiss_index/
│   └── ...
│
└── README.md
```

### Dosyaların Görevi

| Dosya                 | Açıklama                                                      |
| --------------------- | ------------------------------------------------------------- |
| `main.py`             | Temel Gemini + LangChain örneği                               |
| `create_vector_db.py` | Dokümanı parçalayıp FAISS vector database oluşturur           |
| `mil_std_document.py` | RAG sisteminde kullanılan teknik doküman içeriğini barındırır |
| `rag.py`              | Embedding, FAISS retrieval ve RAG chain'i oluşturur           |
| `tools.py`            | Agent'ın kullanabileceği tool'ları tanımlar                   |
| `agent.py`            | Gemini tabanlı AI Agent'ı oluşturur                           |
| `requirements.txt`    | Python bağımlılıklarını içerir                                |
| `faiss_index/`        | Oluşturulan lokal vector database                             |

---

### Teknik Doküman Soruları

```text
RKT-MIL-STD-001 nedir?

Elektronik kontrol biriminin maksimum çalışma sıcaklığı nedir?

Depolama sıcaklığı gereksinimleri nelerdir?

Titreşim test kriterleri nelerdir?

Nem gereksinimleri nelerdir?
```

Agent, teknik dokümanla ilgili sorularda RAG pipeline'ını kullanır ve cevabı getirilen context ile sınırlar.

Dokümanda cevap bulunmadığında sistem:

```text
Bu bilgi dokümanda bulunmuyor.
```

şeklinde cevap vermek üzere tasarlanmıştır.

---

# 🧩 Kullanılan Teknolojiler

| Teknoloji        | Kullanım Alanı                 |
| ---------------- | ------------------------------ |
| 🟦 Google Gemini | LLM ve Embedding               |
| 🔗 LangChain     | LLM application framework      |
| 🧠 LangGraph     | Agent state / memory           |
| 🔎 FAISS         | Vector similarity search       |
| 📚 RAG           | Doküman tabanlı soru-cevap     |
| 🐍 Python        | Uygulama geliştirme            |
| 🔐 python-dotenv | API key / environment yönetimi |

---
