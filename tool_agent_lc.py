"""LangChain + Google Gemini agent with bound tools (weather + news demos)."""

import lc_transformers_shim  # noqa: F401 — before langchain (optional tokenizer vs. broken torch)

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

_WEATHER_BY_CITY = {
    "pune": "Sunny, 32°C, light breeze",
    "mumbai": "Humid, 30°C, partly cloudy",
    "delhi": "Hazy, 28°C, moderate air quality",
}

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


@tool
def get_weather(city_name: str) -> str:
    """Return weather for the given city name."""
    key = city_name.strip().lower()
    if key in _WEATHER_BY_CITY:
        return _WEATHER_BY_CITY[key]
    return "Weather data not available"


@tool
def get_latest_news(topic: str = "general") -> str:
    """Return the latest news headlines for a topic (demo hardcoded feed)."""
    key = topic.strip().lower().replace(" ", "_")
    if not key:
        key = "general"
    if key in _LATEST_NEWS_BY_TOPIC:
        return _LATEST_NEWS_BY_TOPIC[key]
    return "News data not available for this topic."


TOOLS = [get_weather, get_latest_news]
_TOOL_BY_NAME = {t.name: t for t in TOOLS}

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


def _run_tool_calls(ai: AIMessage) -> list[ToolMessage]:
    out: list[ToolMessage] = []
    for call in ai.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        tool_call_id = call.get("id") or ""
        tool_fn = _TOOL_BY_NAME.get(name)
        if tool_fn is None:
            content = f"Unknown tool: {name}"
        else:
            content = str(tool_fn.invoke(args))
        out.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    return out


def run_agent() -> None:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    llm_tools = llm.bind_tools(TOOLS)

    system = SystemMessage(content=SYSTEM_PROMPT.strip())
    history: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [system]

    print(
        "LangChain + Gemini + get_weather & get_latest_news "
        "(type 'exit' to quit)\n"
    )

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        history.append(HumanMessage(content=user_input))
        ai = llm_tools.invoke(history)

        while ai.tool_calls:
            history.append(ai)
            history.extend(_run_tool_calls(ai))
            ai = llm_tools.invoke(history)

        text = ai.content if isinstance(ai.content, str) else str(ai.content)
        history.append(ai)

        print(f"Agent: {text}\n")


if __name__ == "__main__":
    run_agent()
