from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_together import ChatTogether
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.graph import  MermaidDrawMethod
from langchain_community.vectorstores import Chroma
import os
import json

class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    trials: int

test_state = {"messages": [HumanMessage(content="I need to update my password how to do that?")], "trials":0, "summary": ""}

with open('schema.json', 'r') as f:
    schema = json.load(f)

long_memory = InMemoryStore()

config = {"configurable": {"user_id": "1", "thread_id": "1", "recursion_limit": 30}}

# Set up the model
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Change model if needed
    temperature=0.0,
)

def employee(state: State):
    """It is responsible for support the customer"""
    summary = state.get("summary", "")
    trials = state.get("trials", 0)
    info = long_memory.search(("user_info", config["configurable"]["user_id"]))
    if summary:
        sysmsg = SystemMessage(content=f"You are a helpful customer service! Respond to each inquiry as best you can. Consider the summary of earlier conversation: {summary}")
    else:
        sysmsg = SystemMessage(content="You are a helpful customer service! Respond to each inquiry as best you can.")
    
    if info:
        sysmsg.content += f"Consider the following information of the user if necessary: {info}"

    if trials == 0: # new inqury
        inquery = HumanMessage(content=input("User: "))
        context = [sysmsg] + state["messages"] + [inquery]
        return {"messages": [inquery, llm.invoke(context)], 'trials': trials+1, 'summary':summary}
    else: # try to solve the same issue with another sugesstion
        context = state['messages'] + [HumanMessage(content="Give me another method to solve it")]
        return {"messages": [llm.invoke(context)], 'trials': trials+1, 'summary':summary}

def isLong(state: State):
    return  "summarize" if len(state["messages"]) > 25  else "semantic"

def summarizer(state: State):
    """It is responsible for summarize the conversation"""
    summary = state.get("summary", "")
    if summary:
        summary_message = HumanMessage(content=f"This is summary of the earlier conversation: {summary} \n\n Extend the summary by taking into account the new messages below:")
    else:
        summary_message = HumanMessage(content="Create a summary of the conversation below:")
    
    response = llm.invoke([summary_message] + state['messages'][:-5])
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-5]]
    return {"messages": delete_messages, "summary": response.content, 'trials': state["trials"]}


def semanticAnalyzer(state: State):
    """It is responsible for making decision based on semantics of user"""
    response = HumanMessage(content=input("User: "))
    context = [SystemMessage(content="You are semantic analyzer! Analyze the final message, then provide one of the following outputs(Solved, Not solved or End conversation) according to context of the conversation. Note you should not return anything other than given options")] + state['messages'] + [response]
    semantic = llm.invoke(context).content
    summary = state.get("summary", "")
    trials = state.get("trials", 0)
    #print(semantic)
    if semantic == "Solved":
        return {"messages": [AIMessage(content="Can I help you with anything else?")], 'summary': summary, 'trials': trials}
    elif semantic == "Not solved":
        if trials < 3:
            return {"messages": [response], 'summary': summary, 'trials': trials}
        else:
            return {'messages': [response, AIMessage(content= "As the problem is persisting, I will summarize and escalate it to support team.")], 'summary': summary, 'trials': trials}
    else:
        return {'messages': [AIMessage(content= "Good bye.")], 'summary': summary, "trials": -1}

def escalator(state: State):
    """Escalate the issue to human"""
    print(state)
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
    sysmsg = SystemMessage(content=f"You are an information extractor! Follow the following schema:{schema}\n Then responde with schema only filled with information you find and default values if the information does not exist.")
    responde = llm.invoke([sysmsg, state["messages"][-1]]).content
    long_memory.put(("user_info", config["configurable"]["user_id"]), config["configurable"]["user_id"], responde)
    return {"messages": [], "summary": state["summary"], "trails": state['trials']}

graph_builder = StateGraph(State)

graph_builder.add_node("employee", employee)
graph_builder.add_node("extractor", extractor)
graph_builder.add_node("semanticAnalyzer", semanticAnalyzer)
graph_builder.add_node("escalator", escalator)
graph_builder.add_node("summarizer",summarizer)

graph_builder.add_edge(START, "employee")
graph_builder.add_edge("employee", "extractor")
graph_builder.add_edge("summarizer", "semanticAnalyzer")
graph_builder.add_edge("escalator", "employee")
graph_builder.add_conditional_edges("extractor", isLong, {"summarize": "summarizer", "semantic": "semanticAnalyzer"})
graph_builder.add_conditional_edges("semanticAnalyzer", branching, {"Escalate": "escalator", "Continue": "employee", "End": END})


graph = graph_builder.compile()

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


for chunk in graph.stream({"messages": []}, config=config):
    pretty_print_stream_chunk(chunk)

# graph.get_graph().draw_mermaid_png(output_file_path="graph.png",draw_method=MermaidDrawMethod.API)

#print(graph.invoke(test_state)['messages'])

#print(llm.invoke("Explain philosophy behind go?").content)
