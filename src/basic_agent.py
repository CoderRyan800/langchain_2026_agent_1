import base64
import mimetypes
import sys
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

tavily_client = TavilyClient()

system_prompt = """
You are a friendly and helpful agent named Bob. You are here to help the user with their questions and requests.
You can use the web_search tool to search the web for information.
When the user uploads an image, describe and analyze it thoroughly.
When the user uploads an audio file, transcribe and analyze its contents.
"""

# Supported file types for /UPLOAD
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
# Audio requires model="gpt-4o-audio-preview" — see MODEL constant below
SUPPORTED_AUDIO_TYPES = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".opus"}

# gpt-4o supports vision (images). For audio, switch to "gpt-4o-audio-preview".
MODEL = "gpt-4o"


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


def build_upload_content(file_path: str) -> List[Dict]:
    """Read a file and return a LangChain multimodal content list."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    all_supported = SUPPORTED_IMAGE_TYPES | SUPPORTED_AUDIO_TYPES
    if suffix not in all_supported:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(all_supported))}"
        )

    with open(path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    if suffix in SUPPORTED_IMAGE_TYPES:
        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        return [
            {"type": "text", "text": f"I uploaded this image: {path.name}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            },
        ]
    else:
        # Audio format name for OpenAI: m4a -> mp4, others use extension as-is
        fmt_map = {"m4a": "mp4", "flac": "flac", "ogg": "ogg", "opus": "opus"}
        fmt = fmt_map.get(suffix.lstrip("."), suffix.lstrip("."))
        return [
            {"type": "text", "text": f"I uploaded this audio file: {path.name}"},
            {"type": "input_audio", "input_audio": {"data": b64_data, "format": fmt}},
        ]


def print_response(response: Dict) -> None:
    print(f"\nResponse length: {len(response['messages'])}\n")
    for message in response["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}\n")
        elif isinstance(message, HumanMessage):
            # Content may be a list (multimodal) — show just the text part
            content = message.content
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            print(f"You: {content}\n")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content}\n")
        else:
            print(f"[{type(message).__name__}]: {message}\n")
    print()


# SQLite checkpointer persists conversation history across restarts.
# The DB file is created in the working directory as agent_memory.db.
with SqliteSaver.from_conn_string("agent_memory.db") as checkpointer:
    agent = create_agent(
        model=MODEL,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        tools=[web_search],
        middleware=[
            SummarizationMiddleware(
                model=MODEL,
                trigger=("messages", 10),
                keep=("messages", 3),
            )
        ],
    )

    config = {"configurable": {"thread_id": "1"}}

    print("Agent Bob ready.")
    print("Commands:  /STOP               — quit")
    print("           /UPLOAD <filepath>  — upload an image or audio file\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input == "/STOP":
            print("Goodbye!")
            break

        if user_input.startswith("/UPLOAD "):
            file_path = user_input[len("/UPLOAD "):].strip()
            try:
                content = build_upload_content(file_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"Upload error: {e}\n")
                continue
            message = HumanMessage(content=content)
        else:
            message = HumanMessage(content=user_input)

        response = agent.invoke({"messages": [message]}, config=config)
        print_response(response)
