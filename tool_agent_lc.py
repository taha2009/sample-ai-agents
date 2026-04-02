"""LangGraph + Google Gemini agent with bound tools (weather + news demos)."""

import lc_transformers_shim  # noqa: F401 — before langchain (optional tokenizer vs. broken torch)

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
    """Return current weather for the given city (demo data for supported cities)."""
    key = city_name.strip().lower()
    if key in _WEATHER_BY_CITY:
        return _WEATHER_BY_CITY[key]
    return "Weather data not available"


@tool
def get_latest_news(topic: str = "general") -> str:
    """Return the latest news headlines for a topic (demo hardcoded feed).

    `topic`: one of general, world, tech, sports (technology is an alias for tech).
    Omit or use general when the user does not specify a topic.
    """
    key = topic.strip().lower().replace(" ", "_")
    if not key:
        key = "general"
    if key in _LATEST_NEWS_BY_TOPIC:
        return _LATEST_NEWS_BY_TOPIC[key]
    return "News data not available for this topic."


TOOLS = [get_weather, get_latest_news]

SYSTEM_PROMPT = """
You are a helpful assistant.
Use tools when they are the right way to answer; summarize tool results clearly for the user.
Be concise, structured, and ask clarifying questions when needed.
"""


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    llm_tools = llm.bind_tools(TOOLS)

    def call_model(state: MessagesState):
        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT.strip()), *msgs]
        response = llm_tools.invoke(msgs)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


def run_agent() -> None:
    app = build_agent()
    thread_id = "cli-session"
    config = {"configurable": {"thread_id": thread_id}}

    print(
        "LangGraph + Gemini + get_weather & get_latest_news "
        "(MemorySaver checkpointer; type 'exit' to quit)\n"
    )

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        text = _last_ai_text(result["messages"])
        print(f"Agent: {text}\n")


if __name__ == "__main__":
    run_agent()
