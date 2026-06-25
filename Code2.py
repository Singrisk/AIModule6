from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("In the battle of lapento, Who is the commander of the right, left and center fleet of ottoman army in english? ")

print(result.content)