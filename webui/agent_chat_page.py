import streamlit as st
from utils import PLATFORMS, get_models, get_chatllm
from typing import Literal
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools import weather_search


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    print(last_message)
    if last_message.tool_calls:
        return "tools"
    return END


def get_agent_graph(platform, model, temperature):
    tools = [weather_search]
    tool_node = ToolNode(tools)

    def call_model(state):
        messages = state['messages']
        llm = get_chatllm(platform, model, temperature=temperature).bind_tools(tools, parallel_tool_calls=False)
        response = llm.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

def graph_response(graph, input):
    for event in graph.stream(
        {"messages": input},
        config={"configurable": {"thread_id": 42}},
        stream_mode="messages",
    ):
        # st.write(event)

        if type(event[0]) == AIMessageChunk:
            yield event[0].content
        elif type(event[0]) == ToolMessage:
            status_placeholder = st.empty()
            with status_placeholder.status("Calling Tool...", expanded=True) as s:
                st.write("Called `", event[0].name, "` Tool")  # Show which tool is being called
                # st.write("Tool input: ")
                # st.code(event['data'].get('input'))  # Display the input data sent to the tool
                st.write("Tool output: ")
                st.code(event[0].content) # Placeholder for tool output that will be updated later below
                s.update(label="Completed Calling Tool!", expanded=False)


def get_agent_chat_response(platform, model, temperature, input):
    app = get_agent_graph(platform, model, temperature)
    return graph_response(graph=app, input=input)


def display_chat_history():
    for message in st.session_state["agent_chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def clear_chat_history():
    st.session_state["agent_chat_history"] = []


def agent_chat_page():
    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = []

    with st.sidebar:
        st.title("Chatbot")

    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_models(platform))
            temperature = st.slider("请选择历史消息长度", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["agent_chat_history"] += [{"role": 'user', "content": input}]

        print(st.session_state["agent_chat_history"][-history_len:])
        stream_response = get_agent_chat_response(
            platform,
            model,
            temperature,
            st.session_state["agent_chat_history"][-history_len:]
        )

        with st.chat_message("assistant"):
            response1 = st.write_stream(stream_response)
        st.session_state["agent_chat_history"] += [{"role": 'assistant', "content": response1}]

