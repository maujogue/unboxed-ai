import os

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()


def test_chat_mistralai_simple_invoke() -> None:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Set MISTRAL_API_KEY to run this integration test.")

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0,
    )
    response = llm.invoke("Reply with exactly: hey!")
    print(response.content)


if __name__ == "__main__":
    test_chat_mistralai_simple_invoke()
