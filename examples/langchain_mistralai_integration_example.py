import os
import unittest

from langchain_mistralai import ChatMistralAI


class TestLangchainMistralAIIntegration(unittest.TestCase):
    def test_chat_mistralai_simple_invoke(self) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            self.skipTest("Set MISTRAL_API_KEY to run this integration test.")

        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=api_key,
            temperature=0,
        )
        response = llm.invoke("Reply with exactly: pong")

        self.assertIsInstance(response.content, str)
        self.assertTrue(response.content.strip())
