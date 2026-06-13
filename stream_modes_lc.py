import json
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StreamWriter
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Dummy weather for {city}: 32C Sunny"


tools = [get_weather]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,  # required when thinking is enabled
    streaming=True,
    thinking_budget=1024,
    include_thoughts=True,
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State, writer: StreamWriter):
    writer({"event": "llm_start", "msg_count": len(state["messages"])})
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        writer(
            {
                "event": "tool_calls_planned",
                "calls": [tc["name"] for tc in response.tool_calls],
            }
        )
    else:
        writer({"event": "final_answer", "content": str(response.content)})
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("agent", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile(checkpointer=MemorySaver())


def print_event(i: int, mode: str, event):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Event #{i}  |  mode={mode}")
    print(sep)

    if mode == "messages":
        chunk, metadata = event
        print(f"  type       : {type(chunk).__name__}")
        print(
            f"  node       : {metadata.get('langgraph_node')}  (step {metadata.get('langgraph_step')})"
        )

        if isinstance(chunk.content, list):
            for block in chunk.content:
                btype = block.get("type", "?")
                if btype == "thinking":
                    print(f"  [thinking] : {block.get('thinking', '')!r}")
                elif btype == "text":
                    print(f"  [text]     : {block.get('text', '')!r}")
                else:
                    print(f"  [{btype}]  : {block!r}")
            # print(f"  content    : {json.dumps(chunk.content, indent=4)}")
        else:
            print(f"  content    : {chunk.content!r}")

        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            print(f"  tool_calls : {chunk.tool_calls}")
        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            print(f"  tool_call_chunks: {chunk.tool_call_chunks}")
        if hasattr(chunk, "name") and chunk.name:
            print(f"  tool_name  : {chunk.name}")
        if hasattr(chunk, "tool_call_id") and chunk.tool_call_id:
            print(f"  tool_call_id: {chunk.tool_call_id}")
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            print(f"  usage      : {chunk.usage_metadata}")
        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
            print(f"  resp_meta  : {chunk.response_metadata}")

    elif mode == "custom":
        print(f"  data       : {event}")

    elif mode == "updates":
        for node_name, diff in event.items():
            print(f"  node       : {node_name}")
            for msg in diff.get("messages", []):
                print(f"  msg_type   : {type(msg).__name__}")
                print(f"  content    : {msg.content!r}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"  tool_calls : {msg.tool_calls}")
                if hasattr(msg, "name") and msg.name:
                    print(f"  tool_name  : {msg.name}")
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    print(f"  usage      : {msg.usage_metadata}")


def run():
    config = {"configurable": {"thread_id": "cli-session"}}
    print("Streaming Agent  (get_weather tool)  —  type 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        print()
        for i, (mode, event) in enumerate(
            graph.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode=["messages"],  # can also include "custom" and "updates"
            )
        ):
            print_event(i, mode, event)

        print()


if __name__ == "__main__":
    run()
