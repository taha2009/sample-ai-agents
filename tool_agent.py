from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# 🔑 Init client (GEMINI_API_KEY in .env or environment)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Model id (override with GEMINI_MODEL in .env)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Hardcoded weather for supported cities (case-insensitive match on name)
_WEATHER_BY_CITY = {
    "pune": "Sunny, 32°C, light breeze",
    "mumbai": "Humid, 30°C, partly cloudy",
    "delhi": "Hazy, 28°C, moderate air quality",
}


def get_weather(city_name: str) -> str:
    """Return weather for the given city name."""
    key = city_name.strip().lower()
    if key in _WEATHER_BY_CITY:
        return _WEATHER_BY_CITY[key]
    return "Weather data not available"


# Hardcoded “latest news” lines per topic (demo data)
_LATEST_NEWS_BY_TOPIC = {
    "general": (
        "Markets mixed as investors weigh policy signals.\n"
        "City officials outline upcoming infrastructure upgrades.\n"
        "Health agencies remind residents about seasonal precautions."
    ),
    "world": (
        "Diplomatic talks continue on regional cooperation.\n"
        "Aid organizations report progress on relief distribution.\n"
        "Weather agencies track developing storm systems."
    ),
    "tech": (
        "Cloud providers expand capacity in major regions.\n"
        "Developers adopt new safety checks for AI-assisted coding.\n"
        "Consumer gadget launches highlight longer battery life."
    ),
    "technology": (
        "Cloud providers expand capacity in major regions.\n"
        "Developers adopt new safety checks for AI-assisted coding.\n"
        "Consumer gadget launches highlight longer battery life."
    ),
    "sports": (
        "Domestic league standings tighten after weekend fixtures.\n"
        "National squad announces training camp dates.\n"
        "Injury updates expected before the next matchday."
    ),
}


def get_latest_news(topic: str = "general") -> str:
    """Return the latest news headlines for a topic (demo hardcoded feed)."""
    key = topic.strip().lower().replace(" ", "_")
    if not key:
        key = "general"
    if key in _LATEST_NEWS_BY_TOPIC:
        return _LATEST_NEWS_BY_TOPIC[key]
    return "News data not available for this topic."


# 🧠 System instruction
SYSTEM_PROMPT = """
You are a helpful AI assistant.

You can call get_weather(city_name) when the user asks about weather in a city.
You can call get_latest_news(topic) when the user asks for news; use topic "general"
if they do not specify one (supported demo topics: general, world, tech, sports).
Summarize tool results clearly for the user.

Guidelines:
- Be concise
- Ask clarifying questions if needed
- Provide structured and clear answers
"""


def run_agent():
    chat = client.chats.create(
        model=GEMINI_MODEL,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "tools": [get_weather, get_latest_news],
        },
    )

    print(
        "🧠 Gemini CLI Agent + get_weather & get_latest_news tools "
        "(type 'exit' to quit)\n"
    )

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        response = chat.send_message(user_input)
        text = response.text if response.text else ""

        print(f"Agent: {text}\n")


if __name__ == "__main__":
    run_agent()
