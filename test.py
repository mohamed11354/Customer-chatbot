from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_together import ChatTogether
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import os

class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    trials: int

test_state = {"messages": [HumanMessage(content="I need to update my password how to do that?")], "trials":0, "summary": ""}

# Set up the model
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Change model if needed
    temperature=0.7,
)

def employee(state: State):
    """It is responsible for support the customer"""
    summary = state.get("summary", "")
    if summary:
        sysmsg = SystemMessage(content=f"You are a helpful customer service! Respond to each inquiry as best you can. Consider the summary of earlier conversation: {summary}")
    else:
        sysmsg = SystemMessage(content="You are a helpful customer service! Respond to each inquiry as best you can.")
    context = [sysmsg] + state["messages"]
    return {"messages": [llm.invoke(context)]}

def isLong(state: State):
    return  "summarize" if len(state["messages"]) > 25  else "semantic"

def summarizer(state: State):
    """It is responsible for summarize the conversation"""
    summary = state.get("summary", "")
    if summary:
        summary_message = HumanMessage(content=f"This is summary of the earlier conversation: {summary} \n\n Extend the summary by taking into account the new messages below:")
    else:
        summary_message = HumanMessage(content="Create a summary of the conversation below:")
    
    response = llm.invoke([summary_message] + state['messages'])
    return {"messages": state["messages"][-4:], "summary": response.content}


def semanticAnalyzer(state: State):
    """It is responsible for making decision based on semantics of user"""
    context = [SystemMessage(content="You are semantic analyzer! Analyze the following, then provide one of the following outputs: Solved, Not solved or End conversaion.")] + state["messages"][-1]
    semantic = llm.invoke(context).content
    if semantic == "Solved":
        return {"messages": AIMessage(content="Can I help you with anything else?")}
    elif semantic == "Not solved":
        if state["trials"] < 3:
            response = llm.invoke([state['messages'] + HumanMessage("Give me another method to solve it")])
            return {"messages": [response], "trials": state["trials"]+1}
        else:
            return {"trials": state["trials"]+1}
    else:
        return {"trials": -1}

def escalator(state: State):
    """Escalate the issue to human"""
    return {"messages": [AIMessage(content= "As the problem is presisting, I will sumarize and esclate it to support team. Can I help you with anything else?")], "trials": 0}

def branching(state: State):
    if state["trials"] > 3:
        return "Escalate"
    elif state["trials"] == -1:
        return "End"
    else:
        return "Continue" 
graph_builder = StateGraph(State)

graph_builder.add_node("employee", employee)
graph_builder.add_node("semanticAnalyzer", semanticAnalyzer)
graph_builder.add_node("escalator", escalator)
graph_builder.add_node("summarizer",summarizer)

graph_builder.add_edge(START, "employee")
graph_builder.add_edge("employee", "semanticAnalyzer")
graph_builder.add_edge("summarizer", "semanticAnalyzer")
graph_builder.add_edge("escalator", "employee")
graph_builder.add_conditional_edges("employee", isLong, {"summarize": "summarizer", "semantic": "semanticAnalyzer"})
graph_builder.add_conditional_edges("semanticAnalyzer", branching, {"Escalate": "escalator", "Continue": "employee", "End": END})


graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png",draw_method=MermaidDrawMethod.API)

#print(graph.invoke(test_state)['messages'])

#print(llm.invoke("Explain philosophy behind go?").content)
