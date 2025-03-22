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

with open('schema.json', 'r') as f:
    schema = json.load(f)

long_memory = InMemoryStore()

config = {"configurable": {"user_id": "1", "thread_id": "1"}, "recursion_limit": 100}

# Set up the model
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Change model if needed
    temperature=0.0,
)

user_queries = []
query_index = 0 


def employee(state: State):
    """It is responsible for support the customer"""
    global query_index
    summary = state.get("summary", "")
    trials = state.get("trials", 0)
    info = long_memory.search(("user_info", config["configurable"]["user_id"]))
    if summary:
        sysmsg = SystemMessage(content=f"You are a helpful customer service! Respond to each inquiry as best you can. Consider the summary of earlier conversation: {summary}")
    else:
        sysmsg = SystemMessage(content="You are a helpful customer service! Respond to each inquiry as best you can.")
    
    if info:
        sysmsg.content += f"Consider the following information of the user if necessary: {info[0].value}"

    if trials == 0: # new inqury
        # inquery = HumanMessage(content=input("User: "))
        inquery = HumanMessage(user_queries[query_index]) # automate testing 
        query_index += 1
        context = [sysmsg] + state["messages"] + [inquery]
        if inquery.content == "Thanks, goodbye.":
            return {'messages': [inquery, AIMessage(content= "Good bye.")], 'summary': summary, "trials": -1}
        return {"messages": [inquery, llm.invoke(context)], 'trials': trials+1, 'summary':summary}
    else: # try to solve the same issue with another sugesstion
        context = state['messages'] + [HumanMessage(content="Give me more to solve it")]
        return {"messages": [llm.invoke(context)], 'trials': trials+1, 'summary':summary}

def isLong(state: State):
    return  "summarize" if len(state["messages"]) > 25  else "semantic"

def isEnd(state: State):
    return "End" if state["trials"] == -1 else "extract"

def summarizer(state: State):
    """It is responsible for summarize the conversation"""
    summary = state.get("summary", "")
    if summary:
        summary_message = HumanMessage(content=f"This is summary of the earlier conversation: {summary} \n\n Extend the summary by taking into account the new messages above.")
    else:
        summary_message = HumanMessage(content="Create a summary of the conversation above.")
    
    response = llm.invoke(state['messages'][:-10] + [summary_message])
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-10]]
    return {"messages": delete_messages, "summary": response.content, 'trials': state["trials"]}


def semanticAnalyzer(state: State):
    """It is responsible for making decision based on semantics of user"""
    global query_index
    #response = HumanMessage(content=input("User: "))
    response = HumanMessage(content=user_queries[query_index]) # Automate testing
    query_index += 1
    context = [SystemMessage(content="You are semantic analyzer! Analyze the final message, then provide one of the following outputs(Solved, Not solved) according to context of the conversation. Note you should not return anything other than given options")] + [response]
    semantic = llm.invoke(context).content
    summary = state.get("summary", "")
    trials = state.get("trials", 0)
    if(response.content == "Thanks, goodbye."):
        semantic = "End"
    #print(semantic)
    if semantic == "Solved":
        return {"messages": [AIMessage(content="Can I help you with anything else?")], 'summary': summary, 'trials': 0}
    elif semantic == "Not solved":
        if trials < 3:
            return {"messages": [response], 'summary': summary, 'trials': trials}
        else:
            return {'messages': [response, AIMessage(content= "As the problem is persisting, I will summarize and escalate it to support team.")], 'summary': summary, 'trials': trials}
    else:
        return {'messages': [response, AIMessage(content= "Good bye.")], 'summary': summary, "trials": -1}

def escalator(state: State):
    """Escalate the issue to human"""
    #print(state)
    return {"messages": [AIMessage(content= "Can I help you with anything else?")], "trials": 0, 'summary': state["summary"]}

def branching(state: State):
    if state["trials"] > 2:
        return "Escalate"
    elif state["trials"] == -1:
        return "End"
    else:
        return "Continue" 

def extractor(state: State):
    """Extract new memory and update the old memory"""
    global schema
    if state['trials'] == 1:
        sysmsg = SystemMessage(content=f"You are an information extractor! Follow the following schema:{schema}\n Then responde with schema filled with information you find and default values if the information does not exist. Don't write anything rather than filled schema")
        responde = llm.invoke([sysmsg, state["messages"][-2]]).content
        long_memory.put(("user_info", config["configurable"]["user_id"]), config["configurable"]["user_id"], responde)
        schema = responde
    return {"messages": [], "summary": state["summary"], "trails": state['trials']}

graph_builder = StateGraph(State)

graph_builder.add_node("employee", employee)
graph_builder.add_node("extractor", extractor)
graph_builder.add_node("semanticAnalyzer", semanticAnalyzer)
graph_builder.add_node("escalator", escalator)
graph_builder.add_node("summarizer",summarizer)

graph_builder.add_edge(START, "employee")
graph_builder.add_conditional_edges("employee", isEnd, {"extract":"extractor", "End": END})
graph_builder.add_edge("summarizer", "semanticAnalyzer")
graph_builder.add_edge("escalator", "employee")
graph_builder.add_conditional_edges("extractor", isLong, {"summarize": "summarizer", "semantic": "semanticAnalyzer"})
graph_builder.add_conditional_edges("semanticAnalyzer", branching, {"Escalate": "escalator", "Continue": "employee", "End": END})


graph = graph_builder.compile()


# Tests

def pretty_print_stream_chunk(chunk):
    #print(chunk.items())
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            if updates["messages"] == [] or isinstance(updates["messages"][-1], HumanMessage):
                continue
            updates["messages"][-1].pretty_print()
        else:
            print(updates)

        print("\n")

# Test Case 1: The bot resolve the issue 
# with open("Test1.txt", "r") as file:
#     user_queries = [line.strip() for line in file.readlines()]

# for chunk in graph.stream({"messages": []}, config=config):
#     pretty_print_stream_chunk(chunk)

# Test Case 2: The bot cannot resolve the issue so esclate it
# query_index = 0
# with open("Test2.txt", "r") as file:
#     user_queries = [line.strip() for line in file.readlines()]

# for chunk in graph.stream({"messages": []}, config=config):
#     pretty_print_stream_chunk(chunk)

# Test Case 3: The long term memory between sessions
# query_index = 0
# with open("Test3.txt", "r") as file:
#     user_queries = [line.strip() for line in file.readlines()]

# for chunk in graph.stream({"messages": []}, config=config):
#     pretty_print_stream_chunk(chunk)

# config["configurable"]["thread_id"] = "2"
# for chunk in graph.stream({"messages": []}, config=config):
#     pretty_print_stream_chunk(chunk)

# Test Case 4: Summarization
query_index = 0
with open("Test4.txt", "r") as file:
    user_queries = [line.strip() for line in file.readlines()]

for chunk in graph.stream({"messages": []}, config=config):
    pretty_print_stream_chunk(chunk)

# graph.get_graph().draw_mermaid_png(output_file_path="graph.png",draw_method=MermaidDrawMethod.API)

