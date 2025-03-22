# def employeeRespond(state: State):
#     """It is responsible for recive the customer respond"""
#     context = [SystemMessage(content="You are a helpful customer service! Respond to each inquiry as best you can.")] + state["messages"]
#     return {"messages": [llm.invoke(context)], "trials": state['trials']}
from langgraph.store.memory import InMemoryStore
import json

with open('schema.json', 'r') as f:
    schema = json.load(f)

long_memory = InMemoryStore()
config = {"configurable": {"user_id": "1", "thread_id": "1", "recursion_limit": 30}}

long_memory.put(("user_info", config["configurable"]["user_id"]), config["configurable"]["user_id"], schema)

info = long_memory.search(("user_info", config["configurable"]["user_id"]))
print(info[0].value)