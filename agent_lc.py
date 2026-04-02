"""LangChain + Google Gemini CLI agent (conversation memory via message list)."""

import lc_transformers_shim  # noqa: F401 — before langchain (optional tokenizer vs. broken torch)

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Guidelines:
- Be concise
- Ask clarifying questions if needed
- Provide structured and clear answers
"""


def run_agent() -> None:
    print("LangChain + Gemini CLI Agent (type 'exit' to quit)\n")

    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=SYSTEM_PROMPT.strip()),
    ]

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else str(response.content)

        print(f"Agent: {text}\n")

        messages.append(AIMessage(content=text))


if __name__ == "__main__":
    run_agent()
