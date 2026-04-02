from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# 🔑 Init client (GEMINI_API_KEY in .env or environment)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Model id (override with GEMINI_MODEL in .env)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# 🧠 System instruction (correct modern way)
SYSTEM_PROMPT = """
You are a helpful AI assistant.

Guidelines:
- Be concise
- Ask clarifying questions if needed
- Provide structured and clear answers
"""

# 🧾 Conversation memory
history = []

def run_agent():
    print("🧠 Gemini CLI Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        # 🧱 Build conversation
        contents = []

        # Add history
        contents.extend(history)

        # Add current user input (each part must be a text object, not a bare string)
        contents.append({
            "role": "user",
            "parts": [{"text": user_input}],
        })

        # 🤖 Call Gemini
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config={
                "system_instruction": SYSTEM_PROMPT
            }
        )

        text = response.text

        print(f"Agent: {text}\n")

        # 💾 Save memory
        history.append({
            "role": "user",
            "parts": [{"text": user_input}],
        })
        history.append({
            "role": "model",
            "parts": [{"text": text}],
        })


if __name__ == "__main__":
    run_agent()