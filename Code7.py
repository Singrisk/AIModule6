from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

# schema
class Review(TypedDict):

    summary: Annotated[str]
    sentiment: str

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software 
feels bloated. There are too many pre-installed apps that I can't 
remove. Also, the UI looks outdated compared to other brands. Hoping 
for a software update to fix this.""")

print(result)
print(result['summary'])
print(result['sentiment'])