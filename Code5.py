from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of France?")
]

result = llm.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)