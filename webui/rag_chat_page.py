import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Literal
from tools import get_naive_rag_tool


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # print(last_message)
    if last_message.tool_calls:
        return "tools"
    return END


def get_rag_graph(platform, model, temperature, selected_kbs, KBS):
    tools = [KBS[k] for k in selected_kbs]
    tool_node = ToolNode(tools)

    def call_model(state):
        messages = state['messages']
        llm = get_chatllm(platform, model, temperature=temperature).bind_tools(tools, parallel_tool_calls=False, tool_choice="any")
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
        # st.write(graph.get_state_history(config={"configurable": {"thread_id": 42}},))

        if type(event[0]) == AIMessageChunk:
            yield event[0].content
        elif type(event[0]) == ToolMessage:
            status_placeholder = st.empty()
            with status_placeholder.status("正在查询...", expanded=True) as s:
                st.write("已调用 `", event[0].name.replace("_knowledge_base_tool", ""), "` 知识库进行查询")  # Show which tool is being called
                # st.write("Tool input: ")
                # st.code(event['data'].get('input'))  # Display the input data sent to the tool
                st.write("知识库检索结果：")
                for cnum, content in enumerate(event[0].content.split("\n\n")):
                    st.write(f"- 已知信息 {cnum+1}:")
                    st.code(content, wrap_lines=True) # Placeholder for tool output that will be updated later below
                s.update(label="已完成知识库检索！", expanded=False)


def get_rag_chat_response(platform, model, temperature, input, selected_tools, KBS):
    app = get_rag_graph(platform, model, temperature, selected_tools, KBS)
    return graph_response(graph=app, input=input)

def display_chat_history():
    for message in st.session_state["rag_chat_history"]:
        with st.chat_message(message["role"], avatar=get_img_base64("chatchat_avatar.png") if message["role"] == "assistant" else None):
            st.write(message["content"])

def clear_chat_history():
    st.session_state["rag_chat_history"] = []


def rag_chat_page():
    kbs = get_kb_names()
    KBS = dict()
    for k in kbs:
        KBS[f"{k}"] = get_naive_rag_tool(k)

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": "你好，我是你的 Chatchat 智能助手，当前页面为`RAG 对话模式`，可以在对话让大模型基于左侧所选知识库进行回答，有什么可以帮助你的吗？"}
        ]

    with st.sidebar:
        selected_kbs = st.multiselect("请选择对话中可使用的知识库", kbs, default=kbs)

    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_llm_models(platform))
            temperature = st.slider("请选择历史消息长度", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["rag_chat_history"] += [{"role": 'user', "content": input}]

        stream_response = get_rag_chat_response(
            platform,
            model,
            temperature,
            st.session_state["rag_chat_history"][-history_len:],
            selected_kbs,
            KBS
        )

        with st.chat_message("assistant", avatar=get_img_base64("chatchat_avatar.png")):
            response = st.write_stream(stream_response)
        st.session_state["rag_chat_history"] += [{"role": 'assistant', "content": response}]