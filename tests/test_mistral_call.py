import os
import unittest

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

class TestMistralAPIKey(unittest.TestCase):
    def test_mistral_api_key_is_set(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=api_key,
            temperature=0,
        )
        response = llm.invoke("Reply with exactly: hey!")
        print(response.content)
        self.assertIsNotNone(
            api_key, "MISTRAL_API_KEY environment variable must be set"
        )
        self.assertIsInstance(api_key, str)
        self.assertNotEqual(api_key.strip(), "", "MISTRAL_API_KEY must not be empty")
