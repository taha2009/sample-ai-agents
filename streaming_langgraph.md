# LangGraph Streaming — Notes

## Stream Modes

Pass a single mode or a list to `graph.stream()`:

```python
graph.stream(input, config, stream_mode=["messages", "updates", "custom"])
```

When multiple modes are used, each event is a `(mode, event)` tuple.

---

## `messages` mode

Yields `(chunk, metadata)` per token/message as they are produced.

```python
for chunk, metadata in graph.stream(..., stream_mode="messages"):
    ...
```

**`chunk` types:**

| Type | When | Key fields |
|---|---|---|
| `AIMessageChunk` | LLM generating text | `content`, `tool_calls`, `tool_call_chunks`, `usage_metadata` |
| `AIMessageChunk` | LLM deciding a tool call | `content=''`, `tool_calls`, `tool_call_chunks` |
| `AIMessageChunk` | End-of-stream sentinel | `content=''`, `chunk_position='last'` |
| `ToolMessage` | Tool node executed | `content`, `name`, `tool_call_id` |

**`metadata` keys:**

```python
metadata["langgraph_node"]   # which node emitted this chunk
metadata["langgraph_step"]   # step number in the graph execution
```

**Requires `streaming=True` on the LLM** — otherwise `.invoke()` waits for the full response and no token chunks are emitted.

---

## `updates` mode

Yields one event per node after it finishes, containing the state diff it produced.

```python
for update in graph.stream(..., stream_mode="updates"):
    for node_name, diff in update.items():
        for msg in diff.get("messages", []):
            ...
```

- `AIMessage` — full assembled response (text or tool call) after the agent node
- `ToolMessage` — tool result after the tools node
- No streaming chunks; you get the complete message in one shot

---

## `custom` mode

Lets nodes push arbitrary data to the stream at any point during execution via `StreamWriter`.

```python
from langgraph.types import StreamWriter

def my_node(state: State, writer: StreamWriter):
    writer({"event": "started", "detail": "..."})   # emits immediately
    result = do_work()
    writer({"event": "done"})
    return {"messages": [result]}
```

Received as:

```python
for mode, event in graph.stream(..., stream_mode=["messages", "custom"]):
    if mode == "custom":
        print(event)   # whatever was passed to writer()
```

Useful for progress signals, intermediate results, or debug info without touching graph state.

---

## Event sequence for a tool-calling agent

For a query like "What's the weather in Mumbai?":

```
[messages]  AIMessageChunk  — thinking block (if enabled)
[messages]  AIMessageChunk  — tool call decision (content='', tool_calls=[...])
[messages]  AIMessageChunk  — end-of-stream sentinel
[custom]    {event: tool_calls_planned, calls: [...]}   ← StreamWriter
[updates]   {agent: {messages: [AIMessage with tool_calls]}}
[messages]  ToolMessage     — tool result
[updates]   {tools: {messages: [ToolMessage]}}
[custom]    {event: llm_start, msg_count: 3}            ← StreamWriter
[messages]  AIMessageChunk  — first text token
[messages]  AIMessageChunk  — more text tokens ...
[messages]  AIMessageChunk  — end-of-stream sentinel
[custom]    {event: final_answer, content: "..."}       ← StreamWriter
[updates]   {agent: {messages: [AIMessage with final text]}}
```

---

## `AIMessageChunk` content

Can be a **string** (regular text) or a **list of blocks** (when thinking is enabled):

```python
# string
chunk.content == "The weather in Mumbai is..."

# list of blocks
chunk.content == [
    {"type": "thinking", "thinking": "I need to call get_weather..."},
    {"type": "text",     "text": "The weather in Mumbai is..."}
]
```

Thinking and text arrive in **separate chunks** when streaming; the assembled `AIMessage` in `updates` mode has both in the list.

---

## Thinking / Reasoning tokens

### Gemini

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,        # required when thinking is enabled
    thinking_budget=1024, # token budget for reasoning
    include_thoughts=True # surface thinking text in response
)
```

- Thinking text is returned as a `{"type": "thinking", ...}` block in `chunk.content`
- Visible in both `messages` chunks and the assembled `AIMessage` in `updates`

### OpenAI (o-series)

```python
llm = ChatOpenAI(model="o4-mini", reasoning_effort="medium")
```

- Thinking text is **never exposed** — OpenAI discards it before returning the response
- Only the token count is visible: `usage_metadata["output_token_details"]["reasoning"]`

---

## Combining modes

```python
for mode, event in graph.stream(input, config, stream_mode=["messages", "updates", "custom"]):
    if mode == "messages":
        chunk, metadata = event
    elif mode == "updates":
        # {node_name: state_diff}
        pass
    elif mode == "custom":
        # whatever writer() emitted
        pass
```
