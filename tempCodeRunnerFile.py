# def employeeRespond(state: State):
#     """It is responsible for recive the customer respond"""
#     context = [SystemMessage(content="You are a helpful customer service! Respond to each inquiry as best you can.")] + state["messages"]
#     return {"messages": [llm.invoke(context)], "trials": state['trials']}