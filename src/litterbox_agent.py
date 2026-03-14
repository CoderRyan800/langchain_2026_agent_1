"""
Litter Box Monitoring Agent
===========================
Two invocation modes:

  Interactive (cat registration, confirmation, queries):
    python src/litterbox_agent.py

  Sensor-triggered (called by camera/sensor system):
    python src/litterbox_agent.py --event entry --image images/captures/entry.jpg
    python src/litterbox_agent.py --event exit  --image images/captures/exit.jpg
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Ensure src/ is on the path so `litterbox.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from litterbox.db import init_db          # noqa: E402
from litterbox.tools import ALL_TOOLS     # noqa: E402

load_dotenv()
init_db()

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "agent_litterbox_memory.db"

SYSTEM_PROMPT = """
You are a litter box monitoring assistant for a household cat health tracking system.

Your responsibilities:
1. REGISTER CAT IMAGES — When a user uploads a reference photo of a cat, call
   register_cat_image(image_path, cat_name). You MUST always have the cat's name
   before registering. If no name is given, ask for it before proceeding.

2. RECORD ENTRY EVENTS — When the sensor system notifies you of a litter box entry,
   call record_entry(image_path) immediately.

3. RECORD EXIT EVENTS — When the sensor system notifies you of a litter box exit,
   call record_exit(image_path) immediately.

4. CONFIRM IDENTITIES — Help the owner review unconfirmed visits and call
   confirm_identity(visit_id, cat_name) when they identify a cat.

5. ANSWER QUERIES — Use the available query tools to answer questions about visit
   history, health flags, and cat records.

Important rules:
- Health findings from exit analysis are ALWAYS preliminary. Always remind the owner
  that a licensed veterinarian must review any flagged concerns.
- Never speculate beyond what the tools return.
- Orphan exit records (no matching entry) must always be flagged for human review.
"""


def _print_last_ai_message(response: dict) -> None:
    """Print the final AI message and any tool outputs from the response."""
    for message in response["messages"]:
        if isinstance(message, ToolMessage):
            print(f"  [tool result] {message.content}\n")
        elif isinstance(message, AIMessage) and message.content:
            print(f"Assistant: {message.content}\n")


def run_sensor_event(event: str, image_path: str, checkpointer) -> None:
    """Handle a non-interactive sensor trigger (entry or exit)."""
    agent = create_agent(
        model="gpt-4o",
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        tools=ALL_TOOLS,
    )
    # Sensor events use a dedicated thread so they don't pollute interactive history
    config = {"configurable": {"thread_id": "sensor"}}

    if event == "entry":
        prompt = (
            f"SENSOR EVENT: A cat has entered the litter box. "
            f"Entry image path: {image_path}"
        )
    else:
        prompt = (
            f"SENSOR EVENT: A cat has exited the litter box. "
            f"Exit image path: {image_path}"
        )

    response = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    _print_last_ai_message(response)


def run_interactive(checkpointer) -> None:
    """Drop into the interactive chat loop."""
    agent = create_agent(
        model="gpt-4o",
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        tools=ALL_TOOLS,
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o",
                trigger=("messages", 10),
                keep=("messages", 3),
            )
        ],
    )
    config = {"configurable": {"thread_id": "interactive"}}

    print("Litter Box Agent ready.")
    print("Commands:")
    print("  /UPLOAD <filepath>  — register a reference photo for a cat")
    print("  /STOP               — quit\n")

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
            # Pass the path as text; the agent will call register_cat_image
            # and ask for the cat's name if it wasn't provided inline.
            message = HumanMessage(
                content=f"I want to register this cat reference image. File path: {file_path}"
            )
        else:
            message = HumanMessage(content=user_input)

        response = agent.invoke({"messages": [message]}, config=config)
        _print_last_ai_message(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Litter box monitoring agent")
    parser.add_argument(
        "--event", choices=["entry", "exit"],
        help="Sensor event type (entry or exit)"
    )
    parser.add_argument(
        "--image",
        help="Path to the captured image (required for sensor events)"
    )
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        if args.event:
            if not args.image:
                print("Error: --image is required for sensor events", file=sys.stderr)
                sys.exit(1)
            run_sensor_event(args.event, args.image, checkpointer)
        else:
            run_interactive(checkpointer)
