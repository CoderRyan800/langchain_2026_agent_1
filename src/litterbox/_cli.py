"""Console-script entry points installed by setup.py."""
import sys


def main():
    """``litterbox-agent`` command — equivalent to ``python src/litterbox_agent.py``."""
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv
    from langgraph.checkpoint.sqlite import SqliteSaver

    load_dotenv()

    parser = argparse.ArgumentParser(description="Litter box monitoring agent")
    parser.add_argument("--event", choices=["entry", "exit"],
                        help="Sensor event type")
    parser.add_argument("--image", help="Path to the captured image")
    parser.add_argument("--data-dir", help="Override default data directory")
    parser.add_argument("--images-dir", help="Override default images directory")
    parser.add_argument("--weight-pre", type=float, metavar="G")
    parser.add_argument("--weight-entry", type=float, metavar="G")
    parser.add_argument("--weight-exit", type=float, metavar="G")
    parser.add_argument("--ammonia-peak", type=float, metavar="PPB")
    parser.add_argument("--methane-peak", type=float, metavar="PPB")
    args = parser.parse_args()

    from litterbox.api import LitterboxAgent, _SYSTEM_PROMPT
    from litterbox.tools import ALL_TOOLS
    from langchain.messages import HumanMessage, AIMessage, ToolMessage

    agent_obj = LitterboxAgent(
        data_dir=args.data_dir,
        images_dir=args.images_dir,
    )

    def _print_response(response):
        for msg in response["messages"]:
            if isinstance(msg, ToolMessage):
                print(f"  [tool result] {msg.content}\n")
            elif isinstance(msg, AIMessage) and msg.content:
                print(f"Assistant: {msg.content}\n")

    if args.event:
        if not args.image:
            print("Error: --image is required for sensor events", file=sys.stderr)
            sys.exit(1)
        if args.event == "entry":
            print(agent_obj.record_entry(
                args.image,
                weight_pre_g=args.weight_pre,
                weight_entry_g=args.weight_entry,
                ammonia_peak_ppb=args.ammonia_peak,
                methane_peak_ppb=args.methane_peak,
            ))
        else:
            print(agent_obj.record_exit(
                args.image,
                weight_exit_g=args.weight_exit,
                ammonia_peak_ppb=args.ammonia_peak,
                methane_peak_ppb=args.methane_peak,
            ))
        agent_obj.close()
        return

    # Interactive mode
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware

    lg_agent = create_agent(
        model="gpt-4o",
        system_prompt=_SYSTEM_PROMPT,
        checkpointer=agent_obj._checkpointer,
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
            agent_obj.close()
            sys.exit(0)

        if not user_input:
            continue
        if user_input == "/STOP":
            print("Goodbye!")
            break
        if user_input.startswith("/UPLOAD "):
            file_path = user_input[len("/UPLOAD "):].strip()
            message = HumanMessage(
                content=f"I want to register this cat reference image. File path: {file_path}"
            )
        else:
            message = HumanMessage(content=user_input)

        response = lg_agent.invoke({"messages": [message]}, config=config)
        _print_response(response)

    agent_obj.close()


def bob():
    """``litterbox-bob`` command — start the Bob general-purpose assistant."""
    import runpy
    import sys
    from pathlib import Path

    # Locate the src directory relative to this file
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    runpy.run_path(str(src_dir / "basic_agent.py"), run_name="__main__")
