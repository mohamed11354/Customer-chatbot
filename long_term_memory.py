from langgraph.store.memory import InMemoryStore
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_together import ChatTogether
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.graph import  MermaidDrawMethod
import os
import json



class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    trials: int

config = {"configurable": {"user_id": "1", "thread_id": "1"}}

long_memory = InMemoryStore()

with open('schema.json', 'r') as f:
    schema = json.load(f)

#long_memory.put(("user_info",config["configurable"]["user_id"]), config["configurable"]["user_id"], schema)
print(long_memory.search(("user_info", config["configurable"]["user_id"])))


llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Change model if needed
    temperature=0.0,
)


def employee(state: State):
    """It is responsible for support the customer"""
    summary = state.get("summary", "")
    trials = state.get("trials", 0)
    if summary:
        sysmsg = SystemMessage(content=f"You are a helpful customer service! Respond to each inquiry as best you can. Consider the summary of earlier conversation: {summary}")
    else:
        sysmsg = SystemMessage(content="You are a helpful customer service! Respond to each inquiry as best you can.")
    
    if trials == 0: # new inqury
        inquery = HumanMessage(content=input("User: "))
        context = [sysmsg] + state["messages"] + [inquery]
        return {"messages": [inquery, llm.invoke(context)], 'trials': trials+1, 'summary':summary}
    else: # try to solve the same issue with another sugesstion
        context = state['messages'] + [HumanMessage(content="Give me another method to solve it")]
        return {"messages": [llm.invoke(context)], 'trials': trials+1, 'summary':summary}

def extractor(state: State):
    """Extract new memory and update the old memory"""
    sysmsg = SystemMessage(content=f"You are an information extractor! Follow the following schema:{schema}\n Then responde with schema only filled with information you find and default values if the information does not exist.")
    responde = llm.invoke([sysmsg, HumanMessage(content= input("User: "))]).content
    long_memory.put(("user_info",), config["configurable"]["user_id"], responde)
    return {"messages": [responde]}

extractor({})
print(long_memory.search(("user_info",config["configurable"]["user_id"])))