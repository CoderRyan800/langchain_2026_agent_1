"""Console-script entry points installed by setup.py."""
import sys


def main():
    """``litterbox-agent`` command backed by the direct Python API.

    Sensor events call the API methods directly so they can use custom data
    directories and avoid an extra LangGraph turn.  ``python src/litterbox_agent.py``
    remains the project-root LangGraph script.
    """
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    # override=True so the project .env (e.g. LANGSMITH_TRACING=false) wins
    # over any conflicting values inherited from the parent shell.
    load_dotenv(override=True)

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

    from litterbox.api import LitterboxAgent
    from langchain.messages import HumanMessage, AIMessage, ToolMessage

    agent_obj = LitterboxAgent(
        data_dir=args.data_dir,
        images_dir=args.images_dir,
    )

    def _print_response(response):
        # Print only this turn's tool calls and the final assistant reply.
        # response["messages"] is the entire thread, so anything before the
        # last HumanMessage is prior history and must not be re-echoed.
        messages = response["messages"]
        turn_start = 0
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                turn_start = i + 1
                break
        for msg in messages[turn_start:]:
            if isinstance(msg, ToolMessage):
                print(f"  [tool result] {msg.content}\n")
        for msg in reversed(messages[turn_start:]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"Assistant: {msg.content}\n")
                return

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
    lg_agent = agent_obj._get_agent()
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
    from basic_agent import main as bob_main

    bob_main()


if __name__ == "__main__":
    main()
