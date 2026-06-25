from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import os

# Load environment variables
load_dotenv()

# Create Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create Semantic Chunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

# Sample text
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Create chunks
docs = text_splitter.create_documents([sample])

# Display chunks
print(f"Number of chunks: {len(docs)}")

for i, doc in enumerate(docs, start=1):
    print(f"\n{'='*50}")
    print(f"Chunk {i}")
    print(f"{'='*50}")
    print(doc.page_content)