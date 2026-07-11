
from src.llm import get_llm


llm = get_llm()


response = llm.invoke(
    "Explain machine learning in one sentence"
)


print(response.content)