from dotenv import load_dotenv

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from pprint import pprint

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

load_dotenv()

system_prompt = """
You are a friendly and helpful agent named Bob. You are here to help the user with their questions and requests.
You can use the web_search tool to search the web for information.
"""

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

agent = create_agent(
    model="gpt-5-nano",
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    tools=[web_search],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 10),
            keep=("messages", 3)
        )
    ],
)

flag_stop = False

config = {"configurable": {"thread_id": "1"}}

while not flag_stop:
    user_input = input("You: ")
    if user_input == "/STOP":
        flag_stop = True
    else:
        response = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
            },
            config=config
        )
        print("\n\n")
        print(f"Response length: {len(response['messages'])}\n\n")
        for message in response['messages']:
            if isinstance(message, AIMessage):
                print(f"AI Message: {message.content}\n\n")
            elif isinstance(message, HumanMessage):
                print(f"Human Message: {message.content}\n\n")
            elif isinstance(message, ToolMessage):
                print(f"Tool Message: {message.content}\n\n")
            else:
                print(f"Unknown Message: {message}\n\n")

        print("\n\n")




