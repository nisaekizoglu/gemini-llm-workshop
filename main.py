import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from lanchain_core.prompts import ChatPromptTemplate

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("Api key not found. Please set the GOOGLE_API_KEY environment variable.")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt=ChatPromptTemplate(
    [
        ("system",
        "Sen teknik konuları anlatan bir eğitim asistanısın."),
        ("human", 
        """Konu: {topic}"""),
    ]
)
messages = prompt.format_messages(
    topic="Büyük dil modelleri nedir, 1 cümle ile cevap ver."
)

response = model.invoke(
    messages)

print(response.content)