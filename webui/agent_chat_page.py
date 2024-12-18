import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import (
    weather_search_tool,
    get_naive_rag_tool,
    get_duckduckgo_search_tool,
    arxiv_search_tool,
    wikipedia_search_tool,
    daily_ai_papers_tool,
)

AGENT_PAGE_INTRODUCTION = "你好，我是你的 Chatchat 智能助手，当前页面为`Agent 对话模式`，可以在对话让大模型借助左侧所选工具进行回答，有什么可以帮助你的吗？"

def get_agent_graph(platform, model, temperature, selected_tools, TOOLS):
    tools = [TOOLS[k] for k in selected_tools]
    tool_node = ToolNode(tools=tools)

    def call_model(state):
        llm = get_chatllm(platform, model, temperature=temperature)
        llm_with_tools = llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")

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
            with status_placeholder.status("正在调用工具...", expanded=True) as s:
                st.write("已调用 `", event[0].name, "` 工具")  # Show which tool is being called
                # st.write("Tool input: ")
                # st.code(event['data'].get('input'))  # Display the input data sent to the tool
                st.write("工具输出：")
                st.code(event[0].content, wrap_lines=True) # Placeholder for tool output that will be updated later below
                s.update(label="已完成工具调用！", expanded=False)
            st.session_state["agent_tool_calls"].append(
                {
                    "status": "已完成工具调用！",
                    "tool": event[0].name,
                    "content": event[0].content
                })


def get_agent_chat_response(platform, model, temperature, input, selected_tools, TOOLS):
    app = get_agent_graph(platform, model, temperature, selected_tools, TOOLS)
    return graph_response(graph=app, input=input)


def display_chat_history():
    for message in st.session_state["agent_chat_history_with_tool_call"]:
        with st.chat_message(message["role"], avatar=get_img_base64("chatchat_avatar.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message.keys():
                for tool_call in message["tool_calls"]:
                    with st.status(tool_call["status"], expanded=False):
                        st.write("已调用 `", tool_call["tool"], "` 工具")
                        st.write("工具输出：")
                        st.code(tool_call["content"], wrap_lines=True)  # Placeholder for tool output that will be updated later below

            st.write(message["content"])

def clear_chat_history():
    st.session_state["agent_chat_history"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
        ]
    st.session_state["agent_chat_history_with_tool_call"] = [
        {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
    ]
    st.session_state["agent_tool_calls"] = []


def agent_chat_page():
    kbs = get_kb_names()
    duckduckgo_search_tool = get_duckduckgo_search_tool()
    TOOLS = {
        "天气查询": weather_search_tool,
        "Duckduckgo 搜索": duckduckgo_search_tool,
        "Arxiv 搜索": arxiv_search_tool,
        "Wikipedia 搜索": wikipedia_search_tool,
        "今日AI论文查询": daily_ai_papers_tool,
    }
    KBS = {}
    for k in kbs:
        KBS[f"{k} 知识库"] = get_naive_rag_tool(k)

    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
        ]
    if "agent_chat_history_with_tool_call" not in st.session_state:
        st.session_state["agent_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
        ]
    if "agent_tool_calls" not in st.session_state:
        st.session_state["agent_tool_calls"] = []

    with st.sidebar:
        selected_tools = st.pills("请选择对话中可使用的工具", list(TOOLS.keys()), selection_mode="multi")
        st.markdown(f"已选择工具：{selected_tools}")
        selected_kbs = st.pills("请选择对话中可使用的知识库", list(KBS.keys()), selection_mode="multi")
        st.markdown(f"已选择知识库：{selected_kbs}")
        # selected_tools = st.multiselect("请选择对话中可使用的工具", list(TOOLS.keys()), default=list(TOOLS.keys()))

    # selected_tools_kbs = selected_tools + selected_kbs
    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_llm_models(platform))
            temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["agent_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["agent_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        # print(st.session_state["agent_chat_history"][-history_len:])
        stream_response = get_agent_chat_response(
            platform,
            model,
            temperature,
            st.session_state["agent_chat_history"][-history_len:],
            selected_tools + selected_kbs,
            TOOLS
        )

        with st.chat_message("assistant", avatar=get_img_base64("chatchat_avatar.png")):
            response1 = st.write_stream(stream_response)
        st.session_state["agent_chat_history"] += [{"role": 'assistant', "content": response1}]
        st.session_state["agent_chat_history_with_tool_call"] += [{"role": 'assistant', "content": response1, "tool_calls": st.session_state["agent_tool_calls"]}]
        st.session_state["agent_tool_calls"] = []
